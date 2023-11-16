set -ex
python train.py --model ua_cycle_gan \
--netG ua_resnet_9blocks \
--load_pretrain \
--netD basic \
--gpu_ids 3 \
--batch_size 32 \
--num_threads 0 \
--norm instance \
--dataroot ./datasets/Phase2HE \
--name PhaseHE_uacyclegan \
--pool_size 50 \
--no_dropout \
--n_epochs 1 \
--n_epochs_decay 100

# --continue --epoch 35 --epoch_count 35 \
# --verbose \
# --input_nc 1 --lambda_identity 0.0 \
# --freeze \
# --lr 0.0002 \

