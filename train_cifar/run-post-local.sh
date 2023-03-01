CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4\
    --master_port 25047 main.py --seed $(($RANDOM * 32768 + $RANDOM))\
    --data-pth /home/guxinran/localsgd/post_local/data\
    --log-pth ./checkpoint/test2\
    --model resnet56\
    --resume 250 --resume-pth ./checkpoint/postlocal/phase1_common.pt\
    --batch-size 128 --total-batch-size 4096 --steps-per-epoch 12\
    --debug 1 --wandb-save 0\
    --epochs 250\
    --start-lr 0.01 --max-lr -1 --wd 0.0005\
    --momentum 0 --nesterov 0 --warm-up 0 --warmup-epochs 50\
    --bn 1 --bn-batches 100 --group-weight 1 --n-groups -1\
    --decay1 25000 --decay2 25000 --gamma 0.1\
    --step1 600 --step2 600 --step3 600\
    --eval-freq1 1 --eval-freq2 1 --eval-freq3 1\
    --save-freq1 6000 --save-freq2 6000 --save-freq3 6000\
    --eval-on-start 1\
    --replacement 0 --aug 1\
    --label-noise 0 --noise-p 0\
    # --wandb disabled
