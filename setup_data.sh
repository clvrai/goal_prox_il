read -p "Name of server [username@address]: " server_name


copy_traj() {
    copy_dir=`dirname $1`
    #ssh -i ../neurips2020.pem $server_name "mkdir -p rl-framework/$copy_dir"
    #scp -i ../neurips2020.pem $1 $server_name:~/rl-framework/$1
    ssh $server_name "mkdir -p p-goal-prox/$copy_dir"
    #scp ~/rl-framework/$1 $server_name:~/p-goal-prox/$1
    scp ~/p-goal-prox/$1 $server_name:~/p-goal-prox/$1
    echo "Copied $1"
}

#copy_traj ./data/traj/91cd009e/FetchPickAndPlaceDiffHoldout-v0/trajs.pt
#copy_traj ./data/traj/912dbeb1/FetchPickAndPlaceDiffHoldout-v0/trajs.pt 

#copy_traj ./data/traj/f086fb49/FetchPickAndPlaceDiffHoldout-v0/trajs.pt
#copy_traj ./data/traj/9cc80baf/FetchPickAndPlaceDiffHoldout-v0/trajs.pt
#copy_traj ./data/traj/e296aeba/FetchPickAndPlaceDiffHoldout-v0/trajs.pt
#copy_traj ./data/traj/a473429e/FetchPickAndPlaceDiffHoldout-v0/trajs.pt 
#
copy_traj ./data/traj/b947c898/FetchPushEnvCustom-v0/trajs.pt
#copy_traj ./data/traj/6a3f0768/FetchPushEnvCustom-v0/trajs.pt 
#copy_traj ./data/traj/32730637/FetchPushEnvCustom-v0/trajs.pt
#copy_traj ./data/traj/2369ce46/FetchPushEnvCustom-v0/trajs.pt
#
# RND 50
copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-9Q-gw-exp/trajs.pt
copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-2P-gw-exp/trajs.pt
#copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-C8-gw-exp/trajs.pt

# RND 25
#copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-9K-gw-exp/trajs.pt
#copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-MW-gw-exp/trajs.pt
#copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-GI-gw-exp/trajs.pt

# RND 75
#copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-N3-gw-exp/trajs.pt
#copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-8U-gw-exp/trajs.pt
copy_traj data/traj/MiniGrid-FourRooms-v0/810-MGFR-31-1W-gw-exp/trajs.pt

#copy_traj data/traj/1ecac60e/FetchPickAndPlaceDiff-v0/trajs.pt
#copy_traj data/traj/610e04a3/FetchPickAndPlaceDiff-v0/trajs.pt
#copy_traj data/traj/b40f194e/FetchPushEnvCustom-v0/trajs.pt
#copy_traj data/traj/a9f5473b/FetchPushEnvCustom-v0/trajs.pt
copy_traj data/traj/0834022b/FetchPickAndPlaceDiffHoldout-v0/trajs.pt
#copy_traj data/traj/3673a4af/FetchPickAndPlaceDiffHoldout-v0/trajs.pt
#copy_traj data/traj/4e10a31f/FetchPushEnvCustom-v0/trajs.pt
#copy_traj data/traj/487d5bef/FetchPushEnvCustom-v0/trajs.pt
# pick 75
#copy_traj data/traj/de995151/FetchPickAndPlaceDiffHoldout-v0/trajs.pt
# pick 25
#copy_traj data/traj/b8a262b0/FetchPickAndPlaceDiffHoldout-v0/trajs.pt
# push 75
#copy_traj data/traj/69307151/FetchPushEnvCustom-v0/trajs.pt
# push 25 
#copy_traj data/traj/96257b4c/FetchPushEnvCustom-v0/trajs.pt
#
copy_traj data/traj/MiniGrid-FourRooms-v0/429-MGFR-31-0T-gw-exp/trajs.pt
# gw 75
#copy_traj data/traj/MiniGrid-FourRooms-v0/51-MGFR-31-RN-gw-exp/trajs.pt
#copy_traj data/traj/MiniGrid-FourRooms-v0/51-MGFR-31-DL-gw-exp/trajs.pt
# gw 25
#copy_traj data/traj/MiniGrid-FourRooms-v0/51-MGFR-31-KZ-gw-exp/trajs.pt
#

#copy_traj ./data/traj/dm.ball_in_cup.catch/915-dbc-31-H2-ppo/trajs.pt

#copy_traj ./data/traj/885183b7/FetchPushEnvCustom-v0/trajs.pt
#copy_traj ./data/traj/dffc9afb/FetchPushEnvCustom-v0/trajs.pt
#copy_traj ./data/traj/f720075e/FetchPushEnvCustom-v0/trajs.pt
#copy_traj ./data/traj/0ff129f1/FetchPushEnvCustom-v0/trajs.pt

#copy_traj ./data/traj/AntGoal-v0/929-AG-1-DG-ppo/trajs.pt
#copy_traj ./data/traj/AntGoal-v0/929-AG-1-NK-ppo/trajs.pt 
#copy_traj ./data/traj/AntGoal-v0/929-AG-1-B0-ppo/trajs.pt


#copy_traj ./data/traj/AntGoal-v0/929-AG-1-SR-ppo/trajs.pt
#copy_traj ./data/traj/AntGoal-v0/929-AG-1-SO-ppo/trajs.pt 
#copy_traj ./data/traj/AntGoal-v0/929-AG-1-V8-ppo/trajs.pt
