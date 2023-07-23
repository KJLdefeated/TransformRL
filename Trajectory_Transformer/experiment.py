import gym
import numpy as np
import torch
import torch.nn as nn

import argparse
import inspect
import pickle
import random
import sys
import time
import random
import logging
import pickle
import argparse
import os
from torch.nn import functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from GPT_Module.nanogpt import GPT, GPTConfig

class TrajectoryTransformer(GPT):
    def __init__(self, config: GPTConfig):
        super().__init__(config)

        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())
        