import copy
import torch
import torch.optim as optim
from train_utils import group_weight, count_correct, print_client_lr, is_bn
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast
import torch.nn as nn
from loss import LabelNoiseLoss
class Client(object):
    model: nn.Module

    def __init__(self, model, rank, idx, warmup, group_size):
        self.rank = rank
        self.model = copy.deepcopy(model)
        self.warmup = warmup
        self.group_size = group_size
        self.idx = idx

    def create_optimizer(self, refer_lr, momentum, wd, nesterov, if_group_weight):
        if if_group_weight:
            grouped_param = group_weight(self.model)
            if self.is_main_client():
                print("Grouping weight")
            self.optimizer = optim.SGD(
                grouped_param, lr=refer_lr, weight_decay=wd, momentum=momentum, nesterov=nesterov
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=refer_lr, weight_decay=wd, momentum=momentum, nesterov=nesterov
            )
    
    def create_warmup_scheduler(self, scale, warmup_steps):
        self.warmup_scheduler = LinearLR(
            self.optimizer, start_factor=1 / scale, end_factor = 1., total_iters = warmup_steps
        )
    
    def is_main_client(self):
        return self.idx == 0 and self.rank == 0

    def sgd_step(self, batch_image, batch_target, criterion, verbose=False):
        
        if not self.model.training: # optimize for speed
            self.model.train()

        self.optimizer.zero_grad()


        output = self.model(batch_image)
        loss_train = criterion(output, batch_target)
        loss_train.backward()
        self.optimizer.step()

        with torch.no_grad():
            train_correct1 = count_correct(
                output=output,
                target=batch_target,
                topk=(1,)
            )[0]
            cur_stats = torch.stack([
                train_correct1, loss_train * batch_image.shape[0],
                torch.as_tensor(batch_image.shape[0], dtype=loss_train.dtype, device=loss_train.device)
            ])

        # if self.is_main_client() and verbose:
        #     print_client_lr(self.optimizer)
        
        if self.warmup:
            self.warmup_scheduler.step()
        
        return cur_stats
    
    @torch.no_grad()
    def eval_step(self, batch_image, batch_target, criterion):
        if self.model.training: # optimize for speed
            self.model.eval()

        with autocast():
            output = self.model(batch_image)
            loss_val = criterion(output, batch_target)

            val_correct1 = count_correct(
                output=output,
                target=batch_target,
                topk=(1,)
            )[0]
            cur_stats = torch.stack([
                val_correct1, loss_val * batch_image.shape[0],
                torch.as_tensor(batch_image.shape[0], dtype=loss_val.dtype, device=loss_val.device)
            ])

        return cur_stats
    
    def decay_lr(self, gamma):
        for g in self.optimizer.param_groups:
            g['lr'] *= gamma

    def update_bn(self, idx, images):
        if not self.model.training: # optimize for speed
            self.model.train()

        for m in self.model.modules():
            if is_bn(m):
                m.momentum = 1 / (1 + idx)
        with torch.no_grad():
            with autocast():
                self.model(images)
    
    def buffers_to_average(self):
        for name, buffer in self.model.named_buffers():
            if name.endswith("mean") or name.endswith("var"):
                yield buffer
