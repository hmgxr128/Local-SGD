from fileinput import filename
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from distributed_utils import is_main_process, reduce_value,  get_flat_params_from, set_flat_params_to, get_flat_buffers_from, set_flat_buffers_to
import numpy as np
import time
import os
from torch.cuda.amp import autocast




def avg_ps(client_list):
    flattened_list = []
    for client in client_list:
        flatten = get_flat_params_from(client.model)
        flattened_list.append(flatten)
    node_avg = torch.stack(flattened_list).mean(dim=0)
    flat_avg = reduce_value(node_avg, average=True)
    for client in client_list:
        set_flat_params_to(client.model, flat_avg)


def avg_buffer(client_list):
    model = client_list[0].model
    flatten = get_flat_buffers_from(model)
    flat_avg = reduce_value(flatten, average=True)
    for client in client_list:
        set_flat_buffers_to(client.model, flat_avg)







def Avg_Model(model):
    flat = get_flat_params_from(model)
    flat_avg = reduce_value(flat, average=True)
    set_flat_params_to(model, flat_avg)

def Split_list(lst, num_splits):
    splits = np.array_split(lst, num_splits)
    ret = [list(arr) for arr in splits]
    return ret


# The BN momentum should be set as 1
@torch.no_grad()
def Estimate_BN(model_list, trainset, sampler, args):
    # Estimate on non-augmented data
    data_loader = torch.utils.data.DataLoader(trainset,\
                                                sampler=sampler,\
                                                pin_memory=True,\
                                                num_workers=args.nw, batch_size=128, drop_last=True)
    # assert len(data_loader) == 1, "The dataloader should only have one batch"
    for model in model_list:
        model.train()
        for step, data in keep_trying(lambda: list(enumerate(data_loader))):
            images, labels = data
            adjust_bn_momentum(model, step)
            model(images.to(args.device))

def is_bn(m):
    return isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)

@torch.no_grad()
def Estimate_BN_single(model_list, trainset, sampler, args):
    # Estimate on augmented data
    lst = []
    data_loader = torch.utils.data.DataLoader(trainset,\
                                                sampler=sampler,\
                                                pin_memory=True,\
                                                num_workers=args.nw, batch_size=128, drop_last=True)
    # assert len(data_loader) == 1, "The dataloader should only have one batch"
    model = model_list[0]
    #Fetch the first model
    model.train()
    for step, data in keep_trying(lambda: list(enumerate(data_loader))):
        images, labels = data
        adjust_bn_momentum(model, step)
        model(images.to(args.device))
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            lst.append([layer.running_mean.to("cpu"), layer.running_var.to("cpu")])
    torch.save(lst, args.bn_param_pth + "bn_param_{}.pt".format(args.rank))
    print("Process {} finished calculate".format(args.rank))
    return lst



def change_BN_param(param, model_list, args):
    # Insert the running mean and variance into the model
    for model in model_list:
        idx = 0
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                mean_var = param[idx]
                layer.running_mean = mean_var[0].to(args.device)
                layer.running_var = mean_var[1].to(args.device)
                idx += 1

def adjust_bn_momentum(model, idx):
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.momentum = 1 / (1 + idx)

def Check_BN_param(args, model):
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            print("Process {}, Running mean {}".format(args.rank, layer.running_mean))
            print("Process {}, Running var {}".format( args.rank, layer.running_var))
            break
def check_model_param(model):
    for name, param in model.named_parameters():
        print(param.data[0, 0])
        break

def count_BN_layers(model):
    ctr = 0
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            ctr += 1
    return ctr


@torch.no_grad()

def evaluate(model, data_loader, device, loss_function):
    model.eval()

    # 用于存储预测正确的样本个数
    val_correct1_tot = torch.zeros(1).to(device)
    val_correct5_tot = torch.zeros(1).to(device)
    sum_loss = 0.0

    if is_main_process():
        print("Evaluate")
        

    with autocast():
        for step, data in enumerate(data_loader):
            images, targets = data

            output = model(images)
            sum_loss += loss_function(output, targets) * images.size(0)
            val_correct1, val_correct5 = count_correct(output=output, target=targets, topk=(1,5))
            val_correct1_tot += val_correct1
            val_correct5_tot += val_correct5
   
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    dist.barrier()
    # val_correct1_tot, val_correct5_tot, val_loss_tot
    val_correct1_tot = reduce_value(val_correct1_tot, average=False)
    val_correct5_tot = reduce_value(val_correct5_tot, average=False)
    val_loss_tot = reduce_value(sum_loss, average=False)

    return val_correct1_tot.item(), val_correct5_tot.item(), val_loss_tot.item()


@torch.no_grad()

def evaluate_local(model, data_loader, device, loss_function, verbose=False):
    model.eval()

    # to store the correct number of samples
    sum_num = torch.zeros(1).to(device)
    sum_loss = 0.0


    # for step, data in get_loader_enum(data_loader):
    for step, data in keep_trying(lambda: list(enumerate(data_loader))):
        images, labels = data
        if verbose and is_main_process() and step == 0:
            print("Evaluation", labels)
        
        pred = model(images.to(device))
        sum_loss += loss_function(pred, labels.to(device)) * images.size(0)
        
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # wait for all processes to finish computing
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)


    return sum_num.item(), sum_loss.item()


def adjust_lr(optimizer, model, ref_lr, ref_norm_sq):
    norm_sq = eval_param_norm(model)
    for g in optimizer.param_groups:
        temp = ref_lr * (norm_sq / ref_norm_sq)
        g['lr'] = temp
    return temp, norm_sq

def adjust_client_lr(client_list, gamma):
    for client in client_list:
        client.decay_lr(gamma)

def change_lr(optimizer_list, gamma):
    for optimizer in optimizer_list:
        for g in optimizer.param_groups:
            g['lr'] *= gamma


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def print_lr(step_ctr, optimizer):
    if is_main_process():
        print("Step {}, learning rate {}".format(step_ctr, optimizer.param_groups[0]['lr']))

def print_client_lr(optimizer):
    print(f"learning rate {optimizer.param_groups[0]['lr']}")



def keep_trying(f, w=10):
    while True:
        try:
            return f()
        except Exception as e:
            sys.stderr.write(str(e) + "\n")
            sys.stderr.write('Exception, sleep')
            time.sleep(w)


def get_loader_enum(loader):
    return keep_trying(lambda: enumerate(loader))

def eval_param_norm(model):
    norm_sq = 0.

    for name, param in model.named_parameters():
        with torch.no_grad():
            if param.requires_grad:
                norm_sq += torch.sum(param ** 2).item()
            
    return norm_sq

def mkdir(path):
    # remove space
    path=path.strip()
    # remove \ at the end
    path=path.rstrip("\\")
    # judge whether the paths exists
    isExists=os.path.exists(path)
    # judge the result
    if not isExists:
        '''
        differences between os.mkdir(path) and os.makedirs(path): os.mkdirs will create the parent directory but os.mkdir will not
        '''
        # use utf-8 encoding
        os.makedirs(path) 
        print(path + ' is successfully made')
        return True
    else:
        # if the path already exists
        print(path + 'already exists')
        return False

def count_correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k)
        return res

