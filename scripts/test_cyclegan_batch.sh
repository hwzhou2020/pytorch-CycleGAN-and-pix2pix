set -ex
python test.py --dataroot ./datasets/Phase2HE --name PhaseHE_cyclegan --model cycle_gan --phase test --no_dropout --num_test 200 
