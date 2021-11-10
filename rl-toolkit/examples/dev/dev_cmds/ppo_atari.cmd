python tests/run_alg.py --env-name "BreakoutNoFrameskip-v4" --alg ppo --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --linear-lr-decay True --entropy-coef 0.01 --prefix ppo-test --eval-interval -1 --vid-fps 30.0

