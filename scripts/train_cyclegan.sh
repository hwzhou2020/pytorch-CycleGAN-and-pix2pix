set -ex
python train.py --gpu_ids 0,1,2,3 --batch_size 32 --norm instance --dataroot ./datasets/Phase2HE --name PhaseHE_cyclegan --model cycle_gan --pool_size 50 --no_dropout --n_epochs 80 --n_epochs_decay 100
