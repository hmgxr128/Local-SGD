import argparse
import torch
from distributed_utils import init_distributed_mode, is_main_process
from train_utils import mkdir
from distributed_utils import is_main_process
import torchvision
#import resnet_gn
import wandb
from torch.distributed.elastic.multiprocessing.errors import record

from trainer_process import TrainerProcess

import getpass

USER = getpass.getuser()

if USER == 'klyu':
    TRAIN_PATH = "/scratch/imagenet_ffcv/train_500_0.50_90.ffcv"
    VAL_PATH = "/scratch/imagenet_ffcv/val_500_0.50_90.ffcv"
    WANDB_ENTITY = "gxr-team"
else:
    TRAIN_PATH = "/home/guxinran/ffcv_imagenet/train_500_0.50_90.ffcv"
    VAL_PATH = "/home/guxinran/ffcv_imagenet/val_500_0.50_90.ffcv"
    # WANDB_ENTITY = "hmgxr128"
    WANDB_ENTITY = "gxr-team"

@record
def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    # Initialize all processes
    init_distributed_mode(args)

    args.models_per_gpu = int(args.total_batch_size / (args.world_size * args.batch_size))
    args.batch_size_per_gpu = args.batch_size * args.models_per_gpu

    if is_main_process():
        print(f"Total batch size {args.total_batch_size}, models per gpu {args.models_per_gpu}, batch size per gpu {args.batch_size_per_gpu}")

    args.local_steps = args.step1
    args.eval_freq = args.eval_freq1
    args.save_freq = args.save_freq1
    if args.resume_pth is not None:
        print(f"Resume training on model {args.resume_pth}")
    if is_main_process():
        mkdir(args.log_pth)
        set_wandb(args)
    
    TrainerProcess(args).run()


def set_wandb(args):
    config = vars(args)

    wandb.init(
        mode=args.wandb,
        project="LocalSGD-imagenet",
        entity=WANDB_ENTITY,
        name=f"imagenet_B={args.total_batch_size}_H={args.step1}={args.step2}-{args.step3}_lr={args.lr}_m={args.model}",
        config=config
    )
    wandb.run.log_code(".")
    #settings=wandb.Settings(start_method='fork'),


if __name__ == '__main__':
    model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))


    parser = argparse.ArgumentParser()
    parser.add_argument('--train-pth', type=str, default=TRAIN_PATH)
    parser.add_argument('--val-pth', type=str, default=VAL_PATH)
    parser.add_argument('--nw', type=int, default=12)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps-per-epoch', type=int, default=-1)
    parser.add_argument('--total-batch-size', type=int, default=16384)
    parser.add_argument('--warm-up', type=int, default=0, help='Whether to use lr warmup')
    parser.add_argument('--gamma', type=float, default=0.1, help='The factor for learning rate decay')
    parser.add_argument('--model', type=str, default='resnet50', choices=model_names, help='The model to use')

    parser.add_argument('--momentum', type=float, default=0, help='Momentum value')
    parser.add_argument('--nesterov', type=int, default=0, help='Whether to use nesterov momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay value')
    parser.add_argument('--decay1', type=int, default=150, help='first time to decay')
    parser.add_argument('--decay2', type=int, default=225, help='second time to decay')
    parser.add_argument('--decay3', type=int, default=225, help='second time to decay')
    parser.add_argument('--resume', type=int, default=0, help='The epoch to continue training')
    parser.add_argument('--resume-pth', type=str, default=None, help='The path to load the model to resume')
    parser.add_argument('--log-pth', type=str, default=None, help='The path to save model params')
    
    parser.add_argument('--wandb', type=str, default=None, help='wandb mode')
    
    
    parser.add_argument('--step1', type=int, default=1)
    parser.add_argument('--step2', type=int, default=1)
    parser.add_argument('--step3', type=int, default=1)
    parser.add_argument('--eval-freq1', type=int, default=1)
    parser.add_argument('--eval-freq2', type=int, default=1)
    parser.add_argument('--eval-freq3', type=int, default=1)
    parser.add_argument('--save-freq1', type=int, default=50)
    parser.add_argument('--save-freq2', type=int, default=50)
    parser.add_argument('--save-freq3', type=int, default=50)
    parser.add_argument('--group-weight', type=int, default=1)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--bn-batches', type=int, default=100)


    args = parser.parse_args()

    if args.resume_pth == 'None':
        args.resume_pth = None
    
    if args.wandb == 'None':
        args.wandb = None


    main(args)