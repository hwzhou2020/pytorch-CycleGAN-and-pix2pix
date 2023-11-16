set -ex
python train.py --verbose --gpu_ids 3 --batch_size 32 --num_threads 0 --norm instance --dataroot ./datasets/Phase2HE --name PhaseHE_cyclegan --model cycle_gan --pool_size 50 --no_dropout --n_epochs 80 --n_epochs_decay 100
# --continue --epoch 35 --epoch_count 35