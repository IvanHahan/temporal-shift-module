# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
python main.py nandos RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.002 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 8 -j 1 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb
     --tune_from checkpoint/TSM_nandos_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar
