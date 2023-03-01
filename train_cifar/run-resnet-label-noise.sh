CUDA_VISIBLE_DEVICES=04,5,6,7 torchrun --nproc_per_node=4\
    --master_port 25040 main.py --seed $(($RANDOM * 32768 + $RANDOM))\
    --data-pth /home/guxinran/localsgd/post_local/data\
    --log-pth ./checkpoint/test2/\
    --model resnet56_gn\
    --resume 500 --resume-pth ./checkpoint/label_noise/resnet_ckpt/resnet_common_start.pt\
    --batch-size 128 --total-batch-size 4096 --steps-per-epoch 12\
    --debug 1 --wandb-save 0\
    --epochs 500000\
    --start-lr 0.1 --max-lr 0.1 --wd 0.0005\
    --momentum 0 --nesterov 0 --warm-up 1 --warmup-epochs 1\
    --bn 0 --bn-batches 0 --group-weight 1 --n-groups 8\
    --decay1 5000000 --decay2 5000000 --gamma 0.1\
    --step1 384 --step2 384 --step3 384\
    --eval-freq1 1 --eval-freq2 1 --eval-freq3 1\
    --save-freq1 1000 --save-freq2 1000 --save-freq3 1000\
    --eval-on-start 1\
    --replacement 1 --aug 0\
    --label-noise 1 --noise-p 0.1\
    # --wandb disabled