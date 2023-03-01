from typing import List
import torch
import torch.nn as nn
import os
from prepare_data import get_loader
from distributed_utils import get_flat_tensor_from_tensor_sequence,  get_mean_flat_tensor_from_tensor_sequences, cleanup, reduce_value, is_main_process, set_flat_tensor_to_tensor_sequence
import torch.distributed as dist
from train_utils import print_lr, adjust_client_lr
import torchvision
from client import Client
import wandb
import math
from torch.optim.lr_scheduler import LinearLR
import ffcv

VAL_B = 128

class TrainerProcess:
    device: torch.device

    train_loader: ffcv.loader.Loader
    bn_loader: ffcv.loader.Loader
    val_loader: ffcv.loader.Loader

    num_train: int
    num_val: int

    epoch: int = 0

    client_list: List[Client] = []

    best_acc = 0

    phase_step_ctr: int = 0
    comm_round: int = 0
    total_step_ctr: int = 0
    

    def __init__(self, args):
        self.args = args

        self.device = torch.device(args.device)

        self.train_loader = get_loader(
            data_pth=args.train_pth, batch_size=args.batch_size,
            num_workers=args.nw, drop_last=True, rank=args.rank, train=1, seed=args.seed,
            distributed=1, res=224, in_memory=1
        )
        self.bn_loader = get_loader(
            data_pth=args.train_pth, batch_size=args.batch_size,
            num_workers=args.nw, drop_last=True, rank=args.rank, train=1, seed=args.seed,
            distributed=1, res=224, in_memory=1
        )
        self.val_loader = get_loader(
            data_pth=args.val_pth, batch_size=VAL_B,
            num_workers=args.nw, drop_last=False, rank=args.rank, train=0, seed=args.seed,
            distributed=1, res=224, in_memory=1
        )
        
        if self.args.steps_per_epoch == -1:
            steps_per_epoch = len(self.train_loader) // args.models_per_gpu
        else:
            steps_per_epoch = self.args.steps_per_epoch
        args.useful_batches = steps_per_epoch * args.models_per_gpu

        
        self.num_train = args.total_batch_size * steps_per_epoch
        self.num_val = 50000
        len_val = len(self.val_loader)

        if is_main_process():
            print(f"Number of training samples: {self.num_train}, length of val loader: {len_val}, steps per epoch: {steps_per_epoch}")
        
        #Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # correct1, correct5, loss
        self.train_stats = torch.zeros(4, device=self.device)


    def run(self):
        # step_change = 0
        # freq_change = 0

        # Use linear scaling rule
        scale = self.args.total_batch_size / 256
        refer_lr = self.args.lr * scale
        warmup_steps = (self.args.warmup_epochs * self.num_train) // self.args.total_batch_size

        if is_main_process():
            if self.args.warm_up:
                print(f"Reference_LR: {refer_lr}, warmup step: {warmup_steps}")
            else:
                print(f"lr: {refer_lr}, no warmup")
        

        
        if is_main_process():
            print("=> creating model '{}'".format(self.args.model))
        model = torchvision.models.__dict__[self.args.model]().to(self.device)

        # Initialize weights
        if self.args.resume_pth is not None:
            model.load_state_dict(torch.load(self.args.resume_pth, map_location=self.device))
        else:
            init_pth = os.path.join(self.args.log_pth, f"H={self.args.step1}-{self.args.step2}-{self.args.step3}_init.pt")
            if is_main_process():
                torch.save(model.state_dict(), init_pth)
                
            dist.barrier()
            model.load_state_dict(torch.load(init_pth, map_location=self.device))
        
        # initialize clients
        for i in range(self.args.models_per_gpu):
            client = Client(model=model, rank=self.args.rank, idx=i, warmup=self.args.warm_up, 
                            group_size=self.args.models_per_gpu)
            client.create_optimizer(refer_lr=refer_lr, momentum=self.args.momentum,
                                    wd=self.args.wd, nesterov=self.args.nesterov, if_group_weight=self.args.group_weight)
            if self.args.warm_up:
                client.create_warmup_scheduler(scale, warmup_steps)
            self.client_list.append(client)
        
        #images size ([?, 3, 224, 224])
        
        for nepoch in range(self.args.epochs):
            if self.args.resume_pth is not None:
                self.epoch = self.args.resume + nepoch + 1
            else:
                self.epoch = nepoch + 1
            # self.epoch = (self.args.resume + 1 if self.args.resume_pth is not None else 0) + nepoch

            if is_main_process():
                print(f"Entering epoch {self.epoch}")

            self.train_epoch()

        dist.barrier()
        cleanup()

    
    def train_epoch(self):
        
                
        train_loader_iter = iter(self.train_loader)
        for batch_idx, (images, targets) in enumerate(train_loader_iter):
            if batch_idx >= self.args.useful_batches:
                break

            if is_main_process() and batch_idx == 0:
                print(f"Dataloader {self.epoch}")

            self.local_step(batch_idx, images, targets)
            torch.cuda.synchronize()

            if batch_idx % self.args.models_per_gpu == self.args.models_per_gpu - 1:
                self.phase_step_ctr += 1
                self.total_step_ctr += 1

                # Check whether it is time to average
                if self.phase_step_ctr % self.args.local_steps == 0:
                    self.comm_round += 1
        
                    # if args.rank == 0:
                    #     print(f"{args.rank} before")
                    #     check_model_param(client_list[0].model)
                    #     check_model_param(client_list[1].model)

                    self.aggregation_step()
                    self.post_aggregation_step()
        
        train_loader_iter.close()

        # step wise decay
        if self.epoch == self.args.decay1 or self.epoch == self.args.decay2 or self.epoch == self.args.decay3:
            adjust_client_lr(self.client_list, self.args.gamma)
            if is_main_process():
                torch.save(self.client_list[0].model.state_dict(), os.path.join(self.args.log_pth, f"phase_epoch={self.epoch}.pt"))
                wandb.save(os.path.join(self.args.log_pth, f"phase_epoch={self.epoch}.pt"))
        if self.epoch == self.args.decay1:
            self.args.local_steps = self.args.step2
            self.args.eval_freq = self.args.eval_freq2
            self.args.save_freq = self.args.save_freq2
        
        if self.epoch == self.args.decay2:
            self.args.local_steps = self.args.step3
            self.args.eval_freq = self.args.eval_freq3
            self.args.save_freq = self.args.save_freq3
    
    def local_step(self, batch_idx, images, targets):
        client = self.client_list[batch_idx % self.args.models_per_gpu]
        self.train_stats += client.sgd_step(images, targets, self.criterion, verbose=self.phase_step_ctr <= 50)


    @torch.no_grad()
    def aggregation_step(self):
        flat = get_mean_flat_tensor_from_tensor_sequences([client.model.parameters() for client in self.client_list])
        flat = reduce_value(flat)
        for client in self.client_list:
            set_flat_tensor_to_tensor_sequence(flat, client.model.parameters())

        # if args.rank == 0:
        #     print(f"{args.rank} after")
        #     check_model_param(client_list[0].model)
        #     check_model_param(client_list[1].model)

    
    @torch.no_grad()
    def post_aggregation_step(self):
        if is_main_process() and self.comm_round <= 300:
            print(f"Round {self.comm_round}, phase step {self.phase_step_ctr}, total step {self.total_step_ctr}")
            print_lr(self.total_step_ctr, self.client_list[0].optimizer)

        if self.comm_round % self.args.eval_freq == 0:
            self.save_step()

            avg_train_stats = self.average_train_stats()

            # estimate BN params before evaluate
            self.estimate_BN_params()
            
            val_stats = self.eval_step()

            if is_main_process():
                wandb.log({
                    "epoch": self.epoch,
                    "comm_round": self.comm_round,
                    "train_acc1": avg_train_stats[0], 
                    "train_acc5": avg_train_stats[1],
                    "train_loss": avg_train_stats[2],
                    "val_acc1": val_stats[0], 
                    "val_acc5": val_stats[1], 
                    "val_loss": val_stats[2],
                    "best_acc": self.best_acc,
                    "iter": self.comm_round * self.args.local_steps,
                }, commit = True)
            
            self.train_stats.fill_(0)
    

    def average_train_stats(self):
        avg_train_stats = reduce_value(self.train_stats, average=False)
        samples_between_eval = avg_train_stats[3]
        avg_train_stats = avg_train_stats[:3] / avg_train_stats[3]

        if is_main_process():
            print(
                f"Samples between eval {samples_between_eval}, Epoch {self.epoch}, round {self.comm_round}, "
                f"train top1 {avg_train_stats[0]}, "
                f"train top5 {avg_train_stats[1]}, "
                f"train loss {avg_train_stats[2]}"
            )
        
        return avg_train_stats
    

    def estimate_BN_params(self):
        if is_main_process():
            print("Estimating BN")
        bn_loader_iter = iter(self.bn_loader)
        for idx, (images, targets) in enumerate(bn_loader_iter):
            if idx >= self.args.bn_batches // self.args.world_size:
                break
            self.client_list[0].update_bn(idx, images)
            torch.cuda.synchronize()
        bn_loader_iter.close()

        flat = get_flat_tensor_from_tensor_sequence(self.client_list[0].buffers_to_average())
        flat = reduce_value(flat)
        for client in self.client_list:
            set_flat_tensor_to_tensor_sequence(flat, client.buffers_to_average())


    def save_step(self):
        if is_main_process():
            torch.save(self.client_list[0].model.state_dict(), os.path.join(self.args.log_pth, "latest.pt"))
            # wandb.save(latest_pth)
            if self.epoch % self.args.save_freq == 0:
                torch.save(self.client_list[0].model.state_dict(), os.path.join(
                    self.args.log_pth,
                    f"H={self.args.step1}-{self.args.step2}-{self.args.step3}_epoch={self.epoch}.pt"
                ))
    

    def eval_step(self):
        val_stats = torch.zeros(4, device=self.device)
        for images, targets in self.val_loader:
            val_stats += self.client_list[0].eval_step(images, targets, self.criterion)
            torch.cuda.synchronize()
        avg_val_stats = reduce_value(val_stats, average=False)
        avg_val_stats = avg_val_stats[:3] / avg_val_stats[3]

        if avg_val_stats[0] > self.best_acc:
            self.best_acc = avg_val_stats[0]

        if is_main_process():
            print(f"Epoch {self.epoch}, round {self.comm_round}, "
                  f"val top1 {avg_val_stats[0]}, val top5 {avg_val_stats[1]}, best_top1 {self.best_acc}")
        
        return avg_val_stats
