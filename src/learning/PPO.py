import logging, warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from torch import multiprocessing
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

from ..environments import EnvironmentTorch
from .agent import agent, LAYER_SIZE

class PPOTrainer:
    def __init__(self,
        env: EnvironmentTorch,
        lr: float,
        max_grad_norm: float,
        frames_per_batch: int,
        total_frames: int,
        sub_batch_size: int,
        num_epochs: int,
        clip_epsilon: float,
        gamma: float,
        lmbda: float,
        entropy_eps: float
    ):

        self.env = env

        self.device = (
            torch.device(0)
            if torch.cuda.is_available() and multiprocessing.get_start_method() != "fork"
            else torch.device("cpu")
        )

        self.lr = lr
        self.max_grad_norm = max_grad_norm

        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames

        self.sub_batch_size = sub_batch_size
        self.num_epochs = num_epochs
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps



