import os
import sys
import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import argparse
import toy_envs
from agents import *
from modules import flatten_cfg, outputdir_make_and_add
import hydra
from omegaconf import DictConfig


def maybe_rescale_action(envs):
    try:
        return RescaleAction(envs, min_action=-1, max_action=1)
    except AssertionError:
        return envs


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg : DictConfig) -> None:
    # parse args
    cfg = flatten_cfg(cfg) # flatten the nested Dict structure from hydra
    args = argparse.Namespace(**cfg)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # logger init
    save_path = os.path.join('ckpts', args.env, args.algo, args.description)
    os.makedirs(save_path, exist_ok=True)
    outputdir = outputdir_make_and_add(outputdir=save_path, title=f'seed{args.seed}')
    args.save_path = outputdir
    figdir = os.path.join(args.save_path, 'figures')
    os.makedirs(figdir, exist_ok=True)

    # environment init
    env_kwargs = {}
    if args.env == "NonStationaryMultiGoal-v0":
        env_kwargs = {
            "drift_pattern": args.drift_pattern,
            "total_steps": args.steps,
            "drift_fraction": args.drift_fraction,
        }
    train_envs = gym.make_vec(args.env, num_envs=1, **env_kwargs)
    test_envs = gym.make_vec(args.env, num_envs=args.test_num, **env_kwargs)
    train_envs.action_space.seed(args.seed)
    train_envs = maybe_rescale_action(train_envs) # rescale tanh action (-1~1) to env action space when supported
    test_envs = maybe_rescale_action(test_envs) # rescale tanh action (-1~1) to env action space when supported
    args.state_sizes = train_envs.observation_space.shape[1]
    args.action_sizes = train_envs.action_space.shape[1]
    print("Args:", args)
    print("Observation space:", train_envs.observation_space)
    print("Action space:", train_envs.action_space)
    
    # model
    AGENT = getattr(sys.modules[__name__], args.algo.upper())
    agent = AGENT(args)
    agent.train(train_envs, test_envs)


if __name__ == '__main__':
    main()
