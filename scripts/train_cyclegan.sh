set -ex
python train.py --gpu_ids 0,1,2,3 --batch_size 16 --norm instance --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
