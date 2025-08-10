#!/bin/bash

# gymnasium mujoco-py
nohup python demos/sac.py --env.env-type gymnasium --env.env-name Hopper-v4 --env.seed 0 --agent.seed 0 --logger.use-wandb --agent.device cuda:0 &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name Hopper-v4 --env.seed 1 --agent.seed 1 --logger.use-wandb --agent.device cuda:0 &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name Hopper-v4 --env.seed 2 --agent.seed 2 --logger.use-wandb --agent.device cuda:1 &

nohup python demos/sac.py --env.env-type gymnasium --env.env-name Ant-v4 --env.seed 0 --agent.seed 0 --logger.use-wandb --agent.device cuda:1 &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name Ant-v4 --env.seed 1 --agent.seed 1 --logger.use-wandb --agent.device cuda:2 &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name Ant-v4 --env.seed 2 --agent.seed 2 --logger.use-wandb --agent.device cuda:2 &

nohup python demos/sac.py --env.env-type gymnasium --env.env-name HalfCheetah-v4 --env.seed 0 --agent.seed 0 --logger.use-wandb --agent.device cuda:3 &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name HalfCheetah-v4 --env.seed 1 --agent.seed 1 --logger.use-wandb --agent.device cuda:3 &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name HalfCheetah-v4 --env.seed 2 --agent.seed 2 --logger.use-wandb --agent.device cuda:4 &

nohup python demos/sac.py --env.env-type gymnasium --env.env-name HumanoidStandup-v4 --env.seed 0 --agent.seed 0 --logger.use-wandb --agent.device cuda:4 &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name HumanoidStandup-v4 --env.seed 1 --agent.seed 1 --logger.use-wandb --agent.device cuda:5 &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name HumanoidStandup-v4 --env.seed 2 --agent.seed 2 --logger.use-wandb --agent.device cuda:5 &

nohup python demos/sac.py --env.env-type gymnasium --env.env-name Humanoid-v4 --env.seed 0 --agent.seed 0 --logger.use-wandb --agent.device cuda:6  &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name Humanoid-v4 --env.seed 1 --agent.seed 1 --logger.use-wandb --agent.device cuda:6  &
nohup python demos/sac.py --env.env-type gymnasium --env.env-name Humanoid-v4 --env.seed 2 --agent.seed 2 --logger.use-wandb --agent.device cuda:7  &
