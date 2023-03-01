seed=$(($RANDOM * 32768 + $RANDOM))
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    --master_port 25027 main.py --seed $seed \
    --nw 6\
    --batch-size 32 --total-batch-size 8192 --steps-per-epoch 156\
    --lr 0.005 --epochs 50 \
    --warm-up 0 --gamma 0.1 --model resnet50\
    --save-freq1 50 --step1 26 --eval-freq1 6\
    --momentum 0 --wd 0.0001 --decay1 1000 --decay2 1000 --decay3 1000\
    --log-pth ./checkpoint/SGD150_lr${LR}_bs32_1024_ls${STEP}_seed${seed}/\
    --resume 100 --resume-pth ./checkpoint/phase_epoch=100.pt