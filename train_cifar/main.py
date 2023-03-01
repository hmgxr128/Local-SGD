from trainer_process import TrainerProcess
import wandb
import argparse
import torch
from distributed_utils import init_distributed_mode, is_main_process
from train_utils import mkdir

WANDB_ENTITY = 'hmgxr128'
PROJ_NAME = 'LocalSGD-cifar-label-noise'
NUM_TRAIN = 50000
def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    # Initialize all processes
    init_distributed_mode(args)

    assert args.total_batch_size * args.steps_per_epoch <= NUM_TRAIN, \
        "Number of train samples per ecpoh should not be greater than the trainset size"

    assert args.total_batch_size % args.batch_size == 0, \
        "Total batch size should be divisible by per client batch size"
    args.num_clients = args.total_batch_size // args.batch_size
    assert args.total_batch_size % (args.world_size * args.batch_size) == 0, \
        "The number of clients on each gpu should be equal"
    args.models_per_gpu = args.total_batch_size // (args.world_size * args.batch_size)

    args.batch_size_per_gpu = args.batch_size * args.models_per_gpu
    args.useful_batches = args.steps_per_epoch * args.models_per_gpu

    args.decay1_round = args.decay1 * args.steps_per_epoch // args.step1
    args.decay2_round = (args.decay2 - args.decay1) * args.steps_per_epoch // args.step2 + args.decay1_round

    if is_main_process() and args.debug:
        print(f"decay 1 epoch: {args.decay1}, round {args.decay1_round}. decay 2 epoch: {args.decay2}, round {args.decay2_round}")

    if is_main_process():
        print(f"Total batch size {args.total_batch_size}, models per gpu {args.models_per_gpu}, batch size per gpu {args.batch_size_per_gpu}")

    args.local_steps = args.step1
    args.eval_freq = args.eval_freq1
    args.save_freq = args.save_freq1
    if args.resume_pth is not None and is_main_process():
        print(f"Resume training on model {args.resume_pth}")
    if is_main_process():
        mkdir(args.log_pth)
        set_wandb(args)
    
    TrainerProcess(args).run()


def set_wandb(args):
    config = vars(args)
    wandb.init(
        mode=args.wandb,
        project=PROJ_NAME,
        entity=WANDB_ENTITY,
        name=f"cifar_B={args.total_batch_size}_H={args.step1}={args.step2}-{args.step3}_slr={args.start_lr}_m={args.model}",
        config=config
    )
    wandb.run.log_code(".")
    #settings=wandb.Settings(start_method='fork'),








# model = getattr(models, 'resnet66')()
# print(model)

if __name__ == '__main__':
    model_names_resenet = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
    model_names_vgg = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    model_names = [name + '_gn' for name in model_names_resenet]
    model_names.extend(model_names_resenet)
    model_names.extend(model_names_vgg)
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-pth', type=str, default="./data")
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--nw', type=int, default=2, help='number of workers')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, help='random seed')

    parser.add_argument('--model', type=str, default='resnet56', choices=model_names, help='The model to use')
    parser.add_argument('--bn', type=int, default=1, help='whether the model uses batch normalization')
    parser.add_argument('--group-weight', type=int, default=0, help='whether we remove wd on normalization layers')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size on each client')
    

    parser.add_argument('--start-lr', type=float, default=0.01, \
                        help='learning rate before scaling, true lr = start-lr * num_clients if max_lr is not set')
    parser.add_argument('--max-lr', type=float, default=-1, \
                        help='max_lr to warm up from start_lr, if set negative, will be set to start_lr * num_clients')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay value')
    parser.add_argument('--momentum', type=float, default=0, help='momentum value')
    parser.add_argument('--nesterov', type=float, default=0, help='param for nesterov momentum')


    # schedule
    parser.add_argument('--epochs', type=int, default=300, help='total number of epochs')
    parser.add_argument('--warm-up', type=int, default=0, help='whether to use lr warmup')
    parser.add_argument('--warmup-epochs', type=int, default=5)


    parser.add_argument('--bn-batches', type=int, default=100)

    parser.add_argument('--wandb', type=str, default=None, help='wandb mode')
    parser.add_argument('--wandb-save', type=int, default=0, help='whether we let wandb to save the model')

    parser.add_argument('--total-batch-size', type=int, default=128, help='batch size on each client')
    parser.add_argument('--steps-per-epoch', type=int, default=-1, \
                    help='designate the number of steps per epoch to aviod division problem')

    parser.add_argument('--decay1', type=int, default=250, help='first decay at the decay1-th epoch.')
    parser.add_argument('--decay2', type=int, default=500, help='second decay at the decay2-th epoch.')
    parser.add_argument('--gamma', type=float, default=0.1, help='multiply the learning rate by gamma when the learning rate should decay')

    parser.add_argument('--step1', type=int, default=1)
    parser.add_argument('--step2', type=int, default=1)
    parser.add_argument('--step3', type=int, default=1)
    parser.add_argument('--eval-freq1', type=int, default=1)
    parser.add_argument('--eval-freq2', type=int, default=1)
    parser.add_argument('--eval-freq3', type=int, default=1)
    parser.add_argument('--save-freq1', type=int, default=100)
    parser.add_argument('--save-freq2', type=int, default=100)
    parser.add_argument('--save-freq3', type=int, default=100)

    parser.add_argument('--debug', type=int, default=0, help='whether to turn on debug mode')

    parser.add_argument('--log-pth', type=str, default=None, help='the checkpoint path to save models')

    # resume training params
    parser.add_argument('--resume', type=int, default=0, help='the epoch of the model to resume from')
    parser.add_argument('--resume-pth', type=str, default=None, help='the checkpoint path to resume from')
    parser.add_argument('--eval-on-start', type=int, default=0, help='whether to evaluate the initial model')

    # whether to use sampling with replacement
    parser.add_argument('--replacement', type=int, default=0, help='whether to use sampling with replacement')

    # whether to use data augmentation
    parser.add_argument('--aug', type=int, default=1, help='whether to use data augmentation')

    # label noise setup
    parser.add_argument('--label-noise', type=int, default=0, help="Whether to add label noise")
    parser.add_argument('--noise-p', type=float, default=0.1, help="The corruption probability in label noise")

    # number of groups for group normalization
    parser.add_argument('--n-groups', type=int, default=-1, help="number of groups for group normalization")




    args = parser.parse_args()

    if args.resume_pth == 'None':
        args.resume_pth = None
    
    if args.wandb == 'None':
        args.wandb = None

    main(args)








