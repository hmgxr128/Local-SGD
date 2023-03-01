import os
import torch.nn as nn
import torch
from distributed_utils import is_main_process
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


def group_weight(module):
    # do not add weight decay to normalization params
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

def is_bn(m):
    return isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d)

def print_client_lr(optimizer):
    print(f"learning rate {optimizer.param_groups[0]['lr']}")

def print_lr(step_ctr, optimizer):
    if is_main_process():
        print("Step {}, learning rate {}".format(step_ctr, optimizer.param_groups[0]['lr']))

def adjust_client_lr(client_list, gamma):
    for client in client_list:
        client.decay_lr(gamma)
