# Why (and When) does Local SGD Generalize Better than SGD?

This repository provides the code for the paper "Why (and When) does Local SGD Generalize Better than SGD" by Xinran Gu, Kaifeng Lyu, Longbo Huang and Sanjeev Arora, published in ICLR 2023.

## Code Structure

This repository relies on ```torch.distributed``` to implement parallel training among multiple GPUs. Our implementation allows you to simulate more clients than the number of GPUs you have. For example, if you have 8 GPUs in total and want to experiment with 32 workers with local batch size 128, then  this implementation will assign 4 models to each GPU. Each single GPU will train the 4 models in serial, compute the average parameter among the 4 models after local updates, and finally communicate with all other GPUs to obtain the global average. 

We use [wandb](https://wandb.ai/) to log training statistics. Modify ```WANDB_ENTITY``` and ```PROJ_NAME``` in ```main.py``` to specify you wandb account and project name.

### CIFAR-10 Code

The CIFAR-10 code integrates the training of Local SGD w/ and w/o label noise, learning rate warmup, data augmentation. You can also flexibly choose from multiple model structures and sampling schemes (w/ and w/o replacement). We  list the meaning of command line arguments below.

- ```--data-pth```: data directory.
- ```--log-pth```: the directory to save model checkpoints. A new directory will be automatically created if it does not exist. Model checkpoints will be automatically saved to this directory at initialization and the time of every learning rate decay. The model at the latest aggregation will also be saved.
- ```--model```: the architecture to use.
- ``--resume``: the epoch index of the model to resume from.
- ```--resume-pth```: the directory of the checkpoint to resume from.  Remove this argument or set it as None if you want to train from random initialization.
- ```--batch-size```: local batch size.
- ```--total-batch-size```: total batch size.
- ```--steps-per-epoch```: the number of steps per epoch. For example, for experiments with total batch size 4096, you can set this argument as 12 since $\lfloor 50000/4096\rfloor=12$. We use this argument to keep the total number of samples passed by experiments with different batch sizes the same. For example, when we train with total batch size 512,  the default number of steps per epoch by segmenting the training data will be 97. Then, the total number of samples passed for the same number of epochs will be different for experiments with total batch size 4096 and 512.
- ```--debug```: whether to turn on the debug mode, which will generate more console outputs.
- ```--warm-up```: whether to turn on linear warmup of the 
rate.
- ```--warmup-epochs```: the number of epochs for learning rate warmup.
- ```--start-lr```: If warmup is turned on, this argument specifies the initial learning rate. If warmup is turned off, the learning rate will be set directly as ```start_lr```*```number_of_clients```.
- ```--max-lr```: This argument will only be used when learning rate warmup is turned on. The learning rate at the end of the warmup will be set as this argument if it is non-negative and as ```start_lr```*```number_of_clients``` if it is negative.
- ```--wd```: weight decay factor.
- ``--momentum``: momentum factor.
- ```--nesterov```: whether to enable nesterov momentum.
- ```--bn```: whether the model uses BatchNorm. If this argument is turned on, the workers will pass some additional batches to estimate the BN statistics for evaluation.
- ```--bn-batches```: the number of batches used to estimate the BN statistics. It will only be useful if ```--bn``` is turned on.
- ```--group-weight```: whether to remove weight decay on parameters for normalization layers and bias.
- ```--n-groups```: the number of groups for group normalization. It will only be useful if you choose architectures that use group normalization.
- ```--decay1```: the epoch for first learning rate decay. You should set it as a very large number if no learning rate decay is needed. 
- ```--decay2```: the epoch for second learning rate decay. You should set it as a very large number if the training does not require a second learning rate decay. 
- ```--gamma```: the learning rate will be multiplied by ```gamma``` at each decay.
- ```--step1```: the number of local steps in the first phase.
- ```--step2```: the number of local steps in the second phase.
- ```--step3```: the number of local steps in the third phase.
- ```--eval-freq1```: Test statistics will be evaluated every ```eval_freq1``` *communication rounds* in the first phase.
- ```--eval-freq2```: Test statistics will be evaluated every ``eval_freq2`` *communication rounds* in the second phase.
- ```--eval-freq3```: Test statistics will be evaluated every ``eval_freq3`` *communication rounds* in the third phase.
- ```--save-freq1```: Model checkpoints will be saved every ``save_freq1`` *communication rounds* in the first phase. Set it as a large value if you do not want to save checkpoints.
- ```--save-freq2```:Model checkpoints will be saved every `save_freq2` *communication rounds* in the second phase. Set it as a large value if you do not want to save checkpoints.
- ```--save-freq3```:Model checkpoints will be saved every ```save_freq3``` *communication rounds* in the third phase. Set it as a large value if you do not want to save checkpoints.
- ```--eval-on-start```: whether to evaluate test statistics of the initial model.
- ```--replacement```: whether to use sampling with replacement.
- ```--aug```: whether to use data augmentation
- ```--label-noise```: whether to add label noise
- ``--noise-p``: the corruption probability for label noise. It will only be useful when ```--label-noise``` is turned on.
- ```--wandb```: wandb mode. Set it as "None" if you want wandb to log the training statistics.  Set it as "disabled" if you want to turn off  wandb.
- ```--wandb-save```: whether to save checkpoints to wandb.

### ImageNet Code

We explain the learning rate argument below.The  rest of the command line arguments have the same meaning as those of cifar-10 code. 

- ``--lr``: initial learning rate. If learning rate warmup is turned on, the learning rate will ramp up linearly from the initial learning rate to ```total-batch-size``` * ```lr``` /256. This scaling rule is recommended by [Goyal et al., 2017](https://arxiv.org/abs/1706.02677). If learning rate warmup is turned off, the learning rate is directly set as ```total-batch-size``` * ```lr``` /256.

## Reproducing Our Results

We provide the model checkpoints and sample shell scripts to reproduce our experimental results.

### CIFAR-10 Experiments

The shell scripts for cifar-10 experiments are provided in ```train_cifar``` directory. You can modify ```--step1``` to experiment with different numbers of local steps.

- Run ```run-post-local.sh``` to reproduce Figure 1(a). 
- Run ```run-resnet-label-noise.sh``` and ```run-vgg-label-noise.sh``` to reproduce label noise experiments in Figure 7.
- To reproduce the experiments on reducing the diffusion term in Figure 3(a), you should change the starting chekpoint to ```./checkpoint/postlocal/diffusion_common_start.pt```.

## ImageNet Experiments

The sample shell script of ImageNet experiments is provided in ```train_imagenet``` directory. You can execute ```run.sh```  to reproduce our post-local SGD experiments.

## Requirements

python = 3.9

torch = 1.12.1

torchvision = 0.13.1

wandb = 0.13.5

ImageNet experiments additionally require [ffcv](https://github.com/libffcv/ffcv) 0.0.3 to accelerate data loading. Please follow the instructions on their website to download and generate data.

## References

The implementation of multi-GPU training is adapted from [Zhe Wu's repository](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/train_multi_GPU).

The implementation of label noise loss is adapted from [Alex Damian's repository](https://github.com/adamian98/LabelNoiseFlatMinimizers).

The implementation of model architectures is adapted from [Wei Yang's repository](https://github.com/bearpaw/pytorch-classification). 






