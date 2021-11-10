# Pretending like num-cpu=16 and num-envs=2 and batch-size 128
# num-steps needs to be set to the episode length (50 for reacher)
# each update needs 16 episodes (for 8 threads) so 16 * 8 * 50
python tests/dev/her/def.py --prefix 'her-test' --num-env-steps 5e6 --env-name "FetchReach-v1" --log-smooth-len 10 --save-interval -1 --lr 0.001 --critic-lr 0.001 --tau 0.05 --warmup-steps 0 --update-every 1 --trans-buffer-size 1000000 --batch-size 128 --linear-lr-decay False --max-grad-norm -1 --noise-std 0.1 --noise-type gaussian --num-processes 32 --num-steps 200 --updates-per-batch 40 --log-interval 1 --n-rnd-steps 0 --rnd-prob 0.2 --num-render 0 --eval-interval 25 --gamma 0.98
