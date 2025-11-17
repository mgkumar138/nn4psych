"""
Actor-Critic RNN Model

This module provides the canonical implementation of the ActorCritic model
used for predictive inference tasks. This consolidates the 8 duplicate
implementations into a single source of truth.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import init


class ActorCritic(nn.Module):
    """
    Actor-Critic RNN model for predictive inference tasks.

    This model uses a vanilla RNN with tanh activation to process sequential
    inputs and produces both action logits (actor) and value estimates (critic).

    Parameters
    ----------
    input_dim : int
        Dimension of input features (observation space size).
    hidden_dim : int
        Number of hidden units in the RNN layer.
    action_dim : int
        Number of possible actions.
    gain : float, optional
        Scaling factor for recurrent weight initialization. Default is 1.5.
    noise : float, optional
        Noise variance for hidden state (currently unused, reserved for future).
        Default is 0.0.
    bias : bool, optional
        Whether to include bias terms in RNN and output layers. Default is False.

    Attributes
    ----------
    rnn : nn.RNN
        Recurrent neural network layer.
    actor : nn.Linear
        Linear layer mapping hidden state to action logits.
    critic : nn.Linear
        Linear layer mapping hidden state to value estimate.

    Examples
    --------
    >>> model = ActorCritic(input_dim=9, hidden_dim=64, action_dim=3)
    >>> x = torch.randn(1, 1, 9)  # (batch, seq_len, input_dim)
    >>> hx = torch.zeros(1, 1, 64)  # (num_layers, batch, hidden_dim)
    >>> actor_logits, value, new_hx = model(x, hx)
    >>> actor_logits.shape
    torch.Size([1, 3])
    >>> value.shape
    torch.Size([1, 1])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        gain: float = 1.5,
        noise: float = 0.0,
        bias: bool = False,
    ):
        super(ActorCritic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.gain = gain
        self.noise = noise

        # RNN layer
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            batch_first=True,
            nonlinearity='tanh',
            bias=bias,
        )

        # Actor head (policy network)
        self.actor = nn.Linear(hidden_dim, action_dim, bias=bias)

        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1, bias=bias)

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize model weights using scaled normal distributions.

        Input-to-hidden weights are scaled by 1/sqrt(input_dim).
        Hidden-to-hidden weights are scaled by gain/sqrt(hidden_dim).
        Output weights are scaled by 1/hidden_dim.
        """
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                init.normal_(param, mean=0, std=1.0 / (self.input_dim ** 0.5))
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights (recurrent)
                init.normal_(param, mean=0, std=self.gain / (self.hidden_dim ** 0.5))
            elif 'bias' in name:
                init.constant_(param, 0)

        # Initialize output layer weights
        for layer in [self.actor, self.critic]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.normal_(param, mean=0, std=1.0 / self.hidden_dim)
                elif 'bias' in name:
                    init.constant_(param, 0)

    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Actor-Critic network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).
        hx : torch.Tensor
            Hidden state tensor of shape (num_layers, batch_size, hidden_dim).

        Returns
        -------
        actor_logits : torch.Tensor
            Action logits of shape (batch_size, action_dim).
        critic_value : torch.Tensor
            Value estimate of shape (batch_size, 1).
        new_hx : torch.Tensor
            Updated hidden state of shape (num_layers, batch_size, hidden_dim).
        """
        # Process through RNN
        rnn_out, new_hx = self.rnn(x, hx)

        # Squeeze sequence dimension (assuming seq_len=1 for step-by-step processing)
        rnn_out = rnn_out.squeeze(1)

        # Compute actor logits and critic value
        actor_logits = self.actor(rnn_out)
        critic_value = self.critic(rnn_out)

        return actor_logits, critic_value, new_hx

    def get_initial_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get initial hidden state (zeros).

        Parameters
        ----------
        batch_size : int, optional
            Batch size. Default is 1.
        device : torch.device, optional
            Device to create tensor on. Default is CPU.

        Returns
        -------
        torch.Tensor
            Initial hidden state of shape (1, batch_size, hidden_dim).
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        preset_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Reset hidden state, optionally with preset value.

        Parameters
        ----------
        batch_size : int, optional
            Batch size. Default is 1.
        device : torch.device, optional
            Device to create tensor on. Default is CPU.
        preset_value : float, optional
            Value to initialize hidden state with. Default is 0.0.

        Returns
        -------
        torch.Tensor
            Reset hidden state of shape (1, batch_size, hidden_dim).
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.full(
            (1, batch_size, self.hidden_dim),
            preset_value,
            device=device,
        )

    @classmethod
    def from_config(cls, config) -> "ActorCritic":
        """
        Create model from configuration object.

        Parameters
        ----------
        config : ModelConfig
            Configuration object with model parameters.

        Returns
        -------
        ActorCritic
            Instantiated model.
        """
        return cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            gain=config.gain,
            noise=config.noise,
            bias=config.bias,
        )
