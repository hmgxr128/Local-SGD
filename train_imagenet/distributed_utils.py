import os

import torch
import torch.distributed as dist
import numpy as np


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ: # Add --use_env in the command line
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK']) # Is equal to rank in single machine setting

    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu) # distribute automatically to gpu
    args.dist_backend = 'nccl'  # use nccl backend
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    print("Initializing the {}th process\n".format(args.rank))
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """check if distributed training is supported"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


@torch.no_grad()
def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # single gpu
        return value
    torch.cuda.synchronize()
    dist.all_reduce(value)
    if average:
        value /= world_size
    return value


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_buffers_from(model):
    buffers = []
    for name, buffer in model.named_buffers():
        if name.endswith("mean") or name.endswith("var"):
            buffers.append(buffer.data.view(-1))
    flat_buffers = torch.cat(buffers)
    return flat_buffers

def set_flat_buffers_to(model, flat_buffers):
    prev_ind = 0
    for name, buffer in model.named_buffers():
        if name.endswith("mean") or name.endswith("var"):
            flat_size = int(np.prod(list(buffer.size())))
            buffer.data.copy_(flat_buffers[prev_ind:prev_ind + flat_size].view(buffer.size()))
            prev_ind += flat_size

def get_flat_tensor_from_tensor_sequence(seq):
    all = []
    for p in seq:
        all.append(p.view(-1))
    return torch.cat(all)

def get_mean_flat_tensor_from_tensor_sequences(seqs):
    all = []
    for ps in zip(*seqs):
        all.append(torch.stack(ps).mean(dim=0).view(-1))
    return torch.cat(all)

def set_flat_tensor_to_tensor_sequence(flat, seq):
    idx = 0
    for p in seq:
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n
