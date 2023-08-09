
import torch
import torch.nn as nn
from prepare_data import get_loader
from distributed_utils import is_main_process, cleanup, get_mean_flat_tensor_from_tensor_sequences, get_flat_tensor_from_tensor_sequence, set_flat_tensor_to_tensor_sequence, reduce_value
from train_utils import print_lr, adjust_client_lr, print_client_lr
from models import resnets, vggnets, resnets_gn, resnets_gn2
import os
import wandb
import torch.distributed as dist
from client import Client
from loss import LabelNoiseLoss

class TrainerProcess:
    def __init__(self, args):
        self.client_list = []



        self.step_ctr = 0 #count the total number of steps
        self.phase_step_ctr = 0 #the number of steps after the latest lr decay
        self.comm_round = 0 #the total number of communication rounds
        self.phase_comm_round = 0

        self.epoch = args.resume

        self.sampler_seed = 0


        self.args = args

        self.device = torch.device(args.device)

        self.train_loader, self.train_sampler = get_loader(args, train=True)
        
        self.bn_loader, self.bn_sampler = get_loader(args, train=True)

        self.val_loader, _ = get_loader(args, train=False)

        if args.debug:
            print(f"process {args.rank}, train loader len {len(self.train_loader)}\n")

        self.num_train = args.total_batch_size * args.steps_per_epoch
        self.num_val = 10000

        if is_main_process() and args.debug:
            print(f"Number of training samples per epoch: {self.num_train}, length of val loader: {len(self.val_loader)}")
        
        #Define loss function
        if args.label_noise:
            self.criterion = LabelNoiseLoss(num_classes=10, p=args.noise_p)
            if is_main_process() and args.debug:
                print(f"Training with label noise")
        
        else:
            self.criterion = nn.CrossEntropyLoss()

        # train acc, loss, number of samples passed
        self.train_stats = torch.zeros(3, device=self.device)
        self.best_acc = 0
    
    def run(self):

        # Use linear scaling rule
        
        if self.args.warm_up:
            if self.args.max_lr < 0:
                refer_lr = self.args.start_lr * self.args.num_clients
            else:
                assert self.args.max_lr >= self.args.start_lr
                refer_lr = self.args.max_lr
        else:
            refer_lr = self.args.start_lr * self.args.num_clients
            

        warmup_steps = self.args.warmup_epochs * self.args.steps_per_epoch

        if is_main_process():
            if self.args.warm_up:
                print(f"Reference_LR: {refer_lr}, warmup step: {warmup_steps}")
            else:
                print(f"Reference_LR: {refer_lr}, no warmup")
        


        if is_main_process():
            print("=> creating model '{}'".format(self.args.model))
        if 'resnet' in self.args.model:
            if 'gn' in self.args.model:
                model = getattr(resnets_gn, self.args.model)(n_groups=self.args.n_groups).to(self.device)
            else:

                model = getattr(resnets, self.args.model)().to(self.device)
        else:
            model = vggnets.VGG(self.args.model).to(self.device)

        # Initialize weights
        if self.args.resume_pth is not None:
            model.load_state_dict(torch.load(self.args.resume_pth, map_location=self.device))
        else:
            init_pth = os.path.join(self.args.log_pth, f"H={self.args.step1}-{self.args.step2}-{self.args.step3}_init.pt")
            if is_main_process():
                torch.save(model.state_dict(), init_pth)
                
            dist.barrier()
            model.load_state_dict(torch.load(init_pth, map_location=self.device))


        
        # Initialize clients
        for i in range(self.args.models_per_gpu):
            client = Client(model=model, rank=self.args.rank, idx=i, warmup=self.args.warm_up, 
                            group_size=self.args.models_per_gpu)
            client.create_optimizer(refer_lr=refer_lr, momentum=self.args.momentum,
                                    wd=self.args.wd, nesterov=self.args.nesterov, if_group_weight=self.args.group_weight)
            
            if self.args.warm_up:
                client.create_warmup_scheduler(refer_lr / self.args.start_lr, warmup_steps)
            self.client_list.append(client)

            
        self.train()
        
        dist.barrier()
        cleanup()
    
    def train(self):
        self.total_steps = self.args.epochs * self.args.steps_per_epoch
        if self.args.eval_on_start:
            self.post_aggregation_step(train=False)
        
        while self.step_ctr < self.total_steps:
            # train_loader_iter = iter(self.train_loader)
            for batch_idx, (images, targets) in enumerate(self.train_loader):

                # if is_main_process() and self.args.debug and batch_idx == 0:
                #     print(f'sampler {list(self.train_sampler)[:10]}')
                
                
                self.local_step(batch_idx, images.to(self.device), targets.to(self.device))
                torch.cuda.synchronize()

                # if it is the last client, increase the counters by 1
                if batch_idx % self.args.models_per_gpu == self.args.models_per_gpu - 1:
                    self.phase_step_ctr += 1
                    self.step_ctr += 1
                    if is_main_process():
                        print(f"phase step {self.phase_step_ctr}, total step {self.step_ctr}")
                        print_client_lr(self.client_list[0].optimizer)
                
                    # Check whether it is time to average
                    if self.phase_step_ctr % self.args.local_steps == 0:
                        self.comm_round += 1
                        self.phase_comm_round += 1

                        self.epoch = self.step_ctr // self.args.steps_per_epoch + self.args.resume

                        # check whether it is time to decay the lr
                        dist.barrier()

                        self.aggregation_step()

                        self.change_lr()
                        self.post_aggregation_step()
                
                
                # reshuffle data among clients
                if batch_idx >= self.args.useful_batches - 1:
                    if not self.args.replacement:
                        self.sampler_seed += 1
                        self.train_sampler.set_epoch(self.sampler_seed)
                        self.bn_sampler.set_epoch(self.sampler_seed)
                    #     if is_main_process() and self.args.debug:
                    #         print('reset seed')
                    # if is_main_process() and self.args.debug:
                    #     print('break')

                    break
    
    def change_lr(self):
        if self.comm_round == self.args.decay1_round or self.comm_round == self.args.decay2_round:
            adjust_client_lr(self.client_list, self.args.gamma)
            self.phase_comm_round = 0
            self.phase_step_ctr = 0
            # if is_main_process():
            #     torch.save(self.client_list[0].model.state_dict(), os.path.join(self.args.log_pth, f"phase_epoch={self.epoch}_round={self.comm_round}.pt"))
            #     if self.args.wandb_save:
            #         wandb.save(os.path.join(self.args.log_pth, f"phase_epoch={self.epoch}_round={self.comm_round}.pt"))
        if self.comm_round == self.args.decay1_round:
            self.args.eval_freq = self.args.eval_freq2
            self.args.local_steps = self.args.step2
            self.args.save_freq = self.args.save_freq2
        
        if self.comm_round == self.args.decay2_round:
            self.args.eval_freq = self.args.eval_freq3
            self.args.local_steps = self.args.step3
            self.args.save_freq = self.args.save_freq3


    def local_step(self, batch_idx, images, targets):
        client = self.client_list[batch_idx % self.args.models_per_gpu]
        self.train_stats += client.sgd_step(images, targets, self.criterion, verbose=self.phase_step_ctr <= 50)


    @torch.no_grad()
    def aggregation_step(self):
        flat = get_mean_flat_tensor_from_tensor_sequences([client.model.parameters() for client in self.client_list]) #average model params on one gpu
        # average across gpus
        flat = reduce_value(flat)

        for client in self.client_list:
            set_flat_tensor_to_tensor_sequence(flat, client.model.parameters())

    
    @torch.no_grad()
    def post_aggregation_step(self, train=True):
        
        if is_main_process() and self.phase_comm_round <= 30 and self.args.debug:
            print(f"Round {self.comm_round}, phase step {self.phase_step_ctr}, total step {self.step_ctr}")
            print_lr(self.step_ctr, self.client_list[0].optimizer)
        
        # self.save_step()

        if self.phase_comm_round % self.args.eval_freq == 0:
            if train:
                avg_train_stats = self.average_train_stats()
            else:
                avg_train_stats = torch.zeros(3).to(self.device)
            if self.args.bn:
                # estimate BN params before evaluate
                self.estimate_BN_params()
            
            val_stats = self.eval_step()
            self.save_step()

            if is_main_process():
                wandb.log({
                    "epoch": self.epoch,
                    "comm_round": self.comm_round,
                    "train_acc1": avg_train_stats[0], 
                    "train_loss": avg_train_stats[1],
                    "val_acc1": val_stats[0], 
                    "val_loss": val_stats[1],
                    "best_acc": self.best_acc,
                    "iter": self.comm_round * self.args.local_steps,
                    "lr": self.client_list[0].optimizer.param_groups[0]['lr']
                }, commit = True)
            
            self.train_stats.fill_(0)

    def average_train_stats(self):
        avg_train_stats = reduce_value(self.train_stats, average=False)
        samples_between_eval = avg_train_stats[2]
        avg_train_stats = avg_train_stats[:2] / samples_between_eval

        if is_main_process():
            print(
                f"Samples between eval {samples_between_eval},\n Epoch {self.epoch}, round {self.comm_round}",
                "train top1 {:.4f}, ".format(avg_train_stats[0]),
                "train loss {:.4f}".format(avg_train_stats[1])
            )
            
        
        return avg_train_stats

    def estimate_BN_params(self):
        if is_main_process():
            print("Estimating BN")
        # bn_loader_iter = iter(self.bn_loader)
        bn_batches = (self.args.bn_batches // self.args.world_size) * self.args.world_size
        if self.args.debug and is_main_process():
            print(f"Number of batches used to compute bn param: {bn_batches}")
        for idx, (images, targets) in enumerate(self.bn_loader):
            
            if idx >= self.args.bn_batches // self.args.world_size:
                # bn_loader_iter.close()
                break
            self.client_list[0].update_bn(idx, images.to(self.device))
            torch.cuda.synchronize()

        flat = get_flat_tensor_from_tensor_sequence(self.client_list[0].buffers_to_average())
        flat = reduce_value(flat)
        for client in self.client_list:
            set_flat_tensor_to_tensor_sequence(flat, client.buffers_to_average())

    def eval_step(self):
        val_stats = torch.zeros(3, device=self.device)
        for images, targets in self.val_loader:
            val_stats += self.client_list[0].eval_step(images.to(self.device), targets.to(self.device), self.criterion)
            torch.cuda.synchronize()
        avg_val_stats = reduce_value(val_stats, average=False)
        # if is_main_process() and self.args.debug:
        #     print(f"avg val {avg_val_stats}")
        avg_val_stats = avg_val_stats[:2] / avg_val_stats[2]
        

        if avg_val_stats[0] > self.best_acc:
            self.best_acc = avg_val_stats[0]

        if is_main_process():
            print(f"Epoch {self.epoch}, round {self.comm_round}, "
                  "val top1 {:.4f}, best_top1 {:.4f}".format(avg_val_stats[0], self.best_acc))
            print_client_lr(self.client_list[0].optimizer)
        
        return avg_val_stats
    
    def save_step(self):
        if is_main_process():
            latest_pth = os.path.join(self.args.log_pth, "latest.pt")
            torch.save(self.client_list[0].model.state_dict(), latest_pth)
            if self.args.wandb_save:
                wandb.save(latest_pth)
            if self.phase_comm_round % self.args.save_freq == 0:
                pth = os.path.join(
                    self.args.log_pth,
                    f"H={self.args.step1}-{self.args.step2}-{self.args.step3}_epoch={self.epoch}_round={self.comm_round}_phase_round={self.phase_comm_round}.pt"
                )
                torch.save(self.client_list[0].model.state_dict(), pth)
                if self.args.wandb_save:
                    wandb.save(pth)





        






