# Instruction-conditioned semantic video acquisition Dec-POMDP.
# Agent 0 is SUT; agents 1..N are UAVs.

python train.py --algo happo --env uav_escs_sc --exp_name IC_HAPPO_semantic_video --cuda False
python train.py --algo mappo --env uav_escs_sc --exp_name IC_MAPPO_semantic_video --cuda False

tensorboard --logdir "/home/king/Downloads/Projects/TON/HARL/HARL/examples/results/uav_escs_sc/" --port 6006
