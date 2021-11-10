Instructions on how the full process of first training a policy to get
demonstrations and then performing IL.

## HalfCheetah Instructions
Steps: 
1. Train expert `python -m rlf --cmd ppo/halfcheetah --cd 0 --cfg ./tests/config.yaml --cuda False --save-interval 10000000 --sess-id 0 --seed "31"`

## CartPole Instructions
Steps: 
1. Train expert `py -m rlf --cfg tests/config.yaml --cuda False --cmd ppo/cartpole  --seed "31"  --sess-id 0 --save-interval 10000000`
2. Evaluate expert and save expert dataset `py -m rlf --cfg tests/config.yaml
   --cuda False --cmd ppo/cartpole  --seed "31"  --eval-only --load-file
   ./data/trained_models/CartPole-v1/519-CP-31-T8-ppo-cartpole-test/model_389.pt --eval-save` 
3. Run BCO with the path to your expert dataset `py -m rlf --cfg tests/config.yaml --cmd bco/cartpole --traj-load-path ./data/traj/CartPole-v1/520-CP-31-AE-ppo-cartpole-test/trajs.pt`
