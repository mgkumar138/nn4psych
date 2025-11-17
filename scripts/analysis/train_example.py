#!/usr/bin/env python3
"""
Example training script using the modular nn4psych package.

This script demonstrates how to use the new package structure for training
RNN actor-critic models on predictive inference tasks.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Import from the new modular package
from nn4psych.models import ActorCritic
from nn4psych.envs import PIE_CP_OB_v2
from nn4psych.training.configs import ExperimentConfig, create_default_config
from nn4psych.utils.io import save_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RNN Actor-Critic on Predictive Inference Task")

    # Config file option
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')

    # Override options
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=None, help='RNN hidden dimension')
    parser.add_argument('--gamma', type=float, default=None, help='Discount factor')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save models')

    return parser.parse_args()


def train(config: ExperimentConfig) -> None:
    """
    Train the model using configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    """
    # Set random seeds
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    # Setup device
    device = torch.device(config.training.device)
    print(f"Using device: {device}")

    # Create model
    model = ActorCritic.from_config(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Create environments for both conditions
    env_cp = PIE_CP_OB_v2.from_config(config.task)
    config.task.condition = "oddball"
    env_ob = PIE_CP_OB_v2.from_config(config.task)
    config.task.condition = "change-point"  # Reset

    contexts = ["change-point", "oddball"]
    envs = [env_cp, env_ob]

    # Training tracking
    all_rewards = []
    save_dir = Path(config.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(config.training.epochs):
        epoch_rewards = []

        # Alternate between conditions
        for context_idx, (context, env) in enumerate(zip(contexts, envs)):
            # Reset hidden state
            h = model.reset_hidden(
                batch_size=1,
                device=device,
                preset_value=config.training.preset_memory,
            )

            # Reset environment
            env._reset_state()

            # Collect rollout data
            log_probs = []
            values = []
            rewards = []

            for trial in range(config.task.total_trials):
                obs, done = env.reset()
                norm_obs = env.normalize_states(obs)
                state = np.concatenate([norm_obs, env.context])

                while not done:
                    # Get action from model
                    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    actor_logits, value, h = model(x, h)

                    # Sample action
                    action_probs = F.softmax(actor_logits, dim=-1)
                    dist = Categorical(action_probs)
                    action = dist.sample()

                    # Store for training
                    log_probs.append(dist.log_prob(action))
                    values.append(value)

                    # Take action
                    obs, reward, done = env.step(action.item())
                    rewards.append(reward)

                    # Update state
                    norm_obs = env.normalize_states(obs)
                    state = np.concatenate([norm_obs, env.context])

            # Compute returns and advantages
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + config.training.gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            log_probs = torch.stack(log_probs).squeeze()
            values = torch.stack(values).squeeze()

            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Compute loss
            advantage = returns - values.detach()
            actor_loss = -(log_probs * advantage).mean()
            critic_loss = F.mse_loss(values, returns)
            loss = actor_loss + 0.5 * critic_loss

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_rewards.append(np.sum(rewards))

        # Track progress
        mean_reward = np.mean(epoch_rewards)
        all_rewards.append(mean_reward)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{config.training.epochs}, Mean Reward: {mean_reward:.4f}")

        # Save model periodically
        if epoch % config.training.save_frequency == 0 and epoch > 0:
            model_path = save_dir / f"model_epoch_{epoch}.pth"
            save_model(model, model_path, optimizer, metadata={'epoch': epoch, 'reward': mean_reward})

    # Save final model
    final_model_path = save_dir / f"{mean_reward:.2f}_{config.get_filename()}.pth"
    save_model(
        model,
        final_model_path,
        optimizer,
        metadata={
            'epochs': config.training.epochs,
            'final_reward': mean_reward,
            'config': config.to_dict(),
        }
    )

    print(f"\nTraining complete!")
    print(f"Final mean reward: {mean_reward:.4f}")
    print(f"Model saved to: {final_model_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    if args.config:
        config = ExperimentConfig.from_yaml(Path(args.config))
    else:
        config = create_default_config()

    # Apply command-line overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.gamma is not None:
        config.training.gamma = args.gamma
    if args.seed is not None:
        config.training.seed = args.seed
    if args.save_dir is not None:
        config.training.save_dir = args.save_dir

    print("=" * 60)
    print("NN4Psych Training Script")
    print("=" * 60)
    print(f"Configuration: {config.name}")
    print(f"Description: {config.description}")
    print()

    # Train
    train(config)


if __name__ == "__main__":
    main()
