"""
Predictive Inference Environments

This module provides gym-style environments for change-point and oddball
predictive inference tasks based on the helicopter/bag positioning paradigm.
"""

import copy
from typing import Tuple, Optional

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class PIE_CP_OB_v2:
    """
    Predictive Inference Environment for Change-Point and Oddball tasks (v2).

    This environment implements the helicopter/bag positioning task where an agent
    must predict where a bag will fall. The task has two conditions:
    - Change-point: The helicopter position changes suddenly (hazard event)
    - Oddball: The bag occasionally falls from a random position

    Parameters
    ----------
    condition : str, optional
        Task condition: "change-point" or "oddball". Default is "change-point".
    total_trials : int, optional
        Number of trials per epoch. Default is 200.
    max_time : int, optional
        Maximum time steps per trial. Default is 300.
    train_cond : bool, optional
        If True, helicopter position is visible to agent. Default is False.
    max_displacement : float, optional
        Maximum bucket movement per action. Default is 15.
    reward_size : float, optional
        Standard deviation for Gaussian reward function. Default is 7.5.
    step_cost : float, optional
        Cost per time step (penalty for inaction). Default is 0.0.
    alpha : float, optional
        Velocity smoothing factor. Default is 1.

    Attributes
    ----------
    action_space : spaces.Discrete
        Discrete action space with 3 actions (left, right, confirm).
    observation_space : spaces.Box
        Observation space with 6 dimensions.
    task_type : str
        Current task condition.
    context : np.ndarray
        One-hot encoding of task condition.
    trial : int
        Current trial number.

    Examples
    --------
    >>> env = PIE_CP_OB_v2(condition="change-point", total_trials=200)
    >>> obs, done = env.reset()
    >>> action = 2  # confirm action
    >>> next_obs, reward, done = env.step(action)
    """

    def __init__(
        self,
        condition: str = "change-point",
        total_trials: int = 200,
        max_time: int = 300,
        train_cond: bool = False,
        max_displacement: float = 15,
        reward_size: float = 7.5,
        step_cost: float = 0.0,
        alpha: float = 1,
    ):
        super(PIE_CP_OB_v2, self).__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -301, 0, 0]),
            high=np.array([301, 301, 301, 301, 301, 301]),
            dtype=np.float32,
        )

        # Environment parameters
        self.max_time = max_time
        self.min_obs_size = 1
        self.max_obs_size = 301
        self.bound_helicopter = 30
        self.total_trials = total_trials
        self.hide_variable = 0

        # Task parameters
        self.max_disp = max_displacement
        self.reward_size = reward_size
        self.step_cost = step_cost
        self.alpha = alpha

        # Task type
        self.task_type = condition
        self.train_cond = train_cond

        # Set context encoding
        if condition == "change-point":
            self.context = np.array([1, 0])
        elif condition == "oddball":
            self.context = np.array([0, 1])
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Hazard rates
        self.change_point_hazard = 0.125
        self.oddball_hazard = 0.125

        # Initialize state
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.helicopter_pos = np.random.randint(
            self.min_obs_size + self.bound_helicopter,
            self.max_obs_size - self.bound_helicopter,
        )
        self.bucket_pos = np.random.randint(
            self.min_obs_size + self.bound_helicopter,
            self.max_obs_size - self.bound_helicopter,
        )
        self.bag_pos = self._generate_bag_position(self.helicopter_pos)

        self.prev_bag_pos = 0
        self.prev_bucket_pos = 0
        self.pred_error = self.prev_bag_pos - self.prev_bag_pos
        self.reward = 0
        self.velocity = 0

        # Trial counter and data storage
        self.trial = 0
        self.trials = []
        self.bucket_positions = []
        self.bag_positions = []
        self.helicopter_positions = []
        self.hazard_triggers = []

    def normalize_states(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize states to [0, 1] range.

        Parameters
        ----------
        x : np.ndarray
            State vector.

        Returns
        -------
        np.ndarray
            Normalized state vector.
        """
        return x / 300

    def reset(self) -> Tuple[np.ndarray, bool]:
        """
        Reset environment for new trial.

        Returns
        -------
        obs : np.ndarray
            Initial observation.
        done : bool
            Episode termination flag (always False after reset).
        """
        self.time = 0
        self.hazard_trigger = 0
        self.velocity = 0

        # Update positions based on task type
        if self.task_type == "change-point":
            if np.random.rand() < self.change_point_hazard:
                self.helicopter_pos = np.random.randint(
                    self.min_obs_size + self.bound_helicopter,
                    self.max_obs_size - self.bound_helicopter,
                )
                self.hazard_trigger = 1
            self.bag_pos = self._generate_bag_position(self.helicopter_pos)

        else:  # oddball
            # Slow drift in helicopter position
            slow_shift = int(np.random.normal(0, 7.5))
            self.helicopter_pos += slow_shift
            self.helicopter_pos = np.clip(
                self.helicopter_pos,
                self.min_obs_size + self.bound_helicopter,
                self.max_obs_size - self.bound_helicopter,
            )

            if np.random.rand() < self.oddball_hazard:
                self.bag_pos = np.random.randint(0, 300)
                self.hazard_trigger = 1
            else:
                self.bag_pos = self._generate_bag_position(self.helicopter_pos)

        # Construct observation
        if self.train_cond:
            self.obs = np.array(
                [
                    self.helicopter_pos,
                    self.bucket_pos,
                    copy.copy(self.hide_variable),
                    self.pred_error,
                    self.prev_bucket_pos,
                    self.prev_bag_pos,
                ],
                dtype=np.float32,
            )
        else:
            self.obs = np.array(
                [
                    copy.copy(self.hide_variable),
                    self.bucket_pos,
                    copy.copy(self.hide_variable),
                    self.pred_error,
                    self.prev_bucket_pos,
                    self.prev_bag_pos,
                ],
                dtype=np.float32,
            )

        self.done = False
        return self.obs, self.done

    def step(
        self,
        action: int,
        direct_action: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one time step in the environment.

        Parameters
        ----------
        action : int
            Action to take: 0=left, 1=right, 2=confirm/stay.
        direct_action : float, optional
            Direct displacement value (for Bayesian agent).

        Returns
        -------
        obs : np.ndarray
            New observation.
        reward : float
            Reward received.
        done : bool
            Whether trial is complete.
        """
        self.time += 1

        # Phase 1: Update bucket position
        if action == 0:  # Move left
            self.gt = -self.max_disp
        elif action == 1:  # Move right
            self.gt = self.max_disp
        elif action == 2:  # Stay/confirm
            self.gt = 0
            self.velocity = 0
        elif direct_action is not None:
            self.gt = direct_action
        else:
            self.gt = 0

        # Apply velocity dynamics
        self.velocity += self.alpha * (-self.velocity + self.gt)
        newbucket_pos = copy.copy(self.bucket_pos) + self.velocity

        # Boundary checking
        if newbucket_pos > self.max_obs_size or newbucket_pos < self.min_obs_size:
            self.velocity = 0
            newbucket_pos = copy.copy(self.bucket_pos)

        self.bucket_pos = np.clip(
            copy.copy(newbucket_pos),
            a_min=self.min_obs_size,
            a_max=self.max_obs_size,
        )

        # Update observation
        self.obs = copy.copy(self.obs)
        self.obs[1] = self.bucket_pos
        self.reward = self.step_cost

        # Phase 2: Bag drop (if confirmed or timeout)
        if action == 2 or self.time >= self.max_time - 1 or direct_action is not None:
            # Show bag drop
            if self.train_cond:
                self.obs = np.array(
                    [
                        self.helicopter_pos,
                        self.bucket_pos,
                        self.bag_pos,
                        self.pred_error,
                        self.prev_bucket_pos,
                        self.prev_bag_pos,
                    ],
                    dtype=np.float32,
                )
            else:
                self.obs = np.array(
                    [
                        copy.copy(self.hide_variable),
                        self.bucket_pos,
                        self.bag_pos,
                        self.pred_error,
                        self.prev_bucket_pos,
                        self.prev_bag_pos,
                    ],
                    dtype=np.float32,
                )

            # Calculate Gaussian reward
            df = ((self.bag_pos - self.bucket_pos) / self.reward_size) ** 2
            self.reward = np.exp(-0.5 * df)

            # Update prediction error for next trial
            self.prev_bag_pos = copy.copy(self.bag_pos)
            self.prev_bucket_pos = copy.copy(self.bucket_pos)
            self.pred_error = self.prev_bag_pos - self.prev_bucket_pos

            # Penalize timeout
            if self.time >= self.max_time - 1:
                self.reward = self.step_cost

            # Store trial data
            self.trial += 1
            self.done = True
            self.trials.append(self.trial)
            self.bucket_positions.append(self.bucket_pos)
            self.bag_positions.append(self.bag_pos)
            self.helicopter_positions.append(self.helicopter_pos)
            self.hazard_triggers.append(self.hazard_trigger)

        return self.obs, self.reward, self.done

    def _generate_bag_position(self, helicopter_pos: float) -> int:
        """
        Generate bag position around helicopter.

        Parameters
        ----------
        helicopter_pos : float
            Current helicopter position.

        Returns
        -------
        int
            Bag position (clipped to bounds).
        """
        bag_pos = int(np.random.normal(helicopter_pos, 20))
        return np.clip(bag_pos, self.min_obs_size, self.max_obs_size)

    def render(self, epoch: int = 0) -> np.ndarray:
        """
        Render and return trial history.

        Parameters
        ----------
        epoch : int, optional
            Current epoch number.

        Returns
        -------
        np.ndarray
            Array containing [trials, bucket_pos, bag_pos, heli_pos, hazards].
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.trials,
            self.bag_positions,
            label='Bag Position',
            color='red',
            marker='o',
            linestyle='-.',
            alpha=0.5,
        )
        plt.plot(
            self.trials,
            self.helicopter_positions,
            label='Helicopter',
            color='green',
            linestyle='--',
        )
        plt.plot(
            self.trials,
            self.bucket_positions,
            label='Bucket Position',
            color='b',
            marker='o',
            linestyle='-.',
            alpha=0.5,
        )

        plt.ylim(-10, 310)
        plt.xlabel('Trial')
        plt.ylabel('Position')
        plt.title(f"Task: {self.task_type.capitalize()} Condition - Epoch: {epoch}")
        plt.legend()
        plt.show()

        return np.array(
            [
                self.trials,
                self.bucket_positions,
                self.bag_positions,
                self.helicopter_positions,
                self.hazard_triggers,
            ]
        )

    def get_state_history(self) -> Tuple:
        """
        Get complete state history for analysis.

        Returns
        -------
        tuple
            (trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers)
        """
        return (
            self.trials,
            self.bucket_positions,
            self.bag_positions,
            self.helicopter_positions,
            self.hazard_triggers,
        )

    @classmethod
    def from_config(cls, config) -> "PIE_CP_OB_v2":
        """
        Create environment from configuration object.

        Parameters
        ----------
        config : TaskConfig
            Configuration object with task parameters.

        Returns
        -------
        PIE_CP_OB_v2
            Instantiated environment.
        """
        return cls(
            condition=config.condition,
            total_trials=config.total_trials,
            max_time=config.max_time,
            train_cond=config.train_cond,
            max_displacement=config.max_displacement,
            reward_size=config.reward_size,
            step_cost=config.step_cost,
            alpha=config.alpha,
        )
