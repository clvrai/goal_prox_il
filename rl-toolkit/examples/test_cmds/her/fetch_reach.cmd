# 50 epochs
# Each epoch 38 (processes) * 50 (episodes) * 50 (steps)
python tests/test_cmds/her/def.py --prefix 'her-test' --env-name "HandReach-v0" --log-smooth-len 1 --save-interval 1 --lr 3e-4 --trans-buffer-size 1e6 --linear-lr-decay False --max-grad-norm -1 --num-processes 38 --update-every 1 --log-interval 1 --eval-interval 1 --gamma 0.98 --normalize-env True --num-env-steps 4.75e6 --num-steps 2500 --tau 0.05 --batch-size 256  --eval-num-processes 10 --num-eval 1 --warmup-step 0 --rnd-prob 0.3
