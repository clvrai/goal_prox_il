## Hopper Steps
1. Train expert policy: 

```
python tests/run_alg.py --prefix ppo-test --use-proper-time-limits --linear-lr-decay True --lr 3e-4 --entropy-coef 0 --num-env-steps 3000000 --num-mini-batch 32 --num-epochs 10 --num-steps 64 --alg ppo --env-name Hopper-v3 --env-log-dir /home/aszot/tmp --eval-interval -1 --log-smooth-len 10 --cuda False --seed 41
```

2. Generate 5 demonstration from expert policy. Note that they take so long to generate because the episodes are so long. 

```
python tests/run_alg.py --prefix ppo-test --use-proper-time-limits --linear-lr-decay True --lr 3e-4 --entropy-coef 0 --num-env-steps 3000000 --num-mini-batch 32 --num-epochs 10 --num-steps 64 --alg ppo --env-name Hopper-v3 --env-log-dir /home/aszot/tmp --eval-interval -1 --log-smooth-len 10 --cuda False --load-file ./data/trained_models/Hopper-v3/529-H-31-IN-ppo-test/model_899.pt --eval-only --eval-save --eval-num-processes 5 --num-eval 1
--
```

3. Train GAIfO
```
py -m rlf --cfg tests/config.yaml --cmd gaifo/hopper --traj-load-path ./data/traj/Hopper-v3/529-H-31-PQ-ppo-test/trajs.pt
```
