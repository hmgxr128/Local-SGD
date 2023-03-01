CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4\
    --master_port 25076 main.py --seed $(($RANDOM * 32768 + $RANDOM))\
    --data-pth /home/guxinran/localsgd/post_local/data\
    --log-pth ./checkpoint/test2\
    --model vgg16\
    --resume 500 --resume-pth ./checkpoint/label_noise/vgg_ckpt/vgg_common_start.pt\
    --batch-size 128 --total-batch-size 4096 --steps-per-epoch 12\
    --debug 1 --wandb-save 0\
    --epochs 50000\
    --start-lr 0.1 --max-lr 0.1 --wd 0\
    --momentum 0 --nesterov 0 --warm-up 1 --warmup-epochs 1\
    --bn 0 --bn-batches 100 --group-weight 0 --n-groups -1\
    --decay1 5000000 --decay2 5000000 --gamma 0.1\
    --step1 384 --step2 384 --step3 384\
    --eval-freq1 1 --eval-freq2 1 --eval-freq3 1\
    --save-freq1 1000 --save-freq2 1000 --save-freq3 1000\
    --eval-on-start 1\
    --replacement 1 --aug 0\
    --label-noise 1 --noise-p 0.1
    # --resume-pth /home/guxinran/localsgd/post_local/check_point/phase1_common.pt\
