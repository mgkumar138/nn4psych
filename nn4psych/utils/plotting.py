"""
Visualization utilities for behavior analysis.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_behavior(
    states: Tuple,
    context: str,
    epoch: int,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Plot trial behavior showing bag, helicopter, and bucket positions.

    Parameters
    ----------
    states : tuple
        State tuple containing:
        - [0]: trials
        - [1]: bucket_positions
        - [2]: bag_positions
        - [3]: helicopter_positions
        - [4]: hazard_triggers
    context : str
        Task context name (e.g., "change-point", "oddball").
    epoch : int
        Training epoch number.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    show : bool, optional
        Whether to call plt.show(). Default is False.

    Returns
    -------
    plt.Figure or None
        Figure object if created, None if using existing axes.

    Examples
    --------
    >>> states = (trials, bucket_pos, bag_pos, heli_pos, hazards)
    >>> plot_behavior(states, "change-point", epoch=50)
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

    trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers = states

    # Plot positions
    ax.plot(
        trials,
        bag_positions,
        label='Bag',
        color='red',
        marker='o',
        linestyle='-.',
        alpha=0.5,
        ms=2,
    )
    ax.plot(
        trials,
        helicopter_positions,
        label='Heli',
        color='green',
        linestyle='--',
        ms=2,
    )
    ax.plot(
        trials,
        bucket_positions,
        label='Bucket',
        color='b',
        marker='o',
        linestyle='-.',
        alpha=0.5,
        ms=2,
    )

    ax.set_ylim(-10, 310)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Position')
    ax.set_title(f"{context}, E:{epoch}")
    ax.legend(fontsize=6)

    if show:
        plt.show()

    return fig


def plot_learning_rate_curve(
    prediction_errors: np.ndarray,
    learning_rates: np.ndarray,
    label: str = "",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[plt.Figure]:
    """
    Plot learning rate as a function of prediction error.

    Parameters
    ----------
    prediction_errors : np.ndarray
        Prediction error values.
    learning_rates : np.ndarray
        Corresponding learning rates.
    label : str, optional
        Label for the plot line.
    ax : plt.Axes, optional
        Axes to plot on.
    **kwargs
        Additional keyword arguments passed to plot.

    Returns
    -------
    plt.Figure or None
        Figure if created.
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

    ax.plot(prediction_errors, learning_rates, label=label, **kwargs)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate vs Prediction Error')

    if label:
        ax.legend()

    return fig


def plot_hazard_triggers(
    states: Tuple,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Plot hazard/change-point events overlaid on position data.

    Parameters
    ----------
    states : tuple
        State tuple.
    ax : plt.Axes, optional
        Axes to plot on.
    show : bool, optional
        Whether to display the plot.

    Returns
    -------
    plt.Figure or None
        Figure if created.
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

    trials, bucket_positions, bag_positions, helicopter_positions, hazard_triggers = states

    # Plot positions
    ax.plot(trials, helicopter_positions, label='Helicopter', color='green', alpha=0.7)
    ax.plot(trials, bag_positions, label='Bag', color='red', alpha=0.5)

    # Mark hazard events
    hazard_indices = np.where(np.array(hazard_triggers) == 1)[0]
    for idx in hazard_indices:
        ax.axvline(x=trials[idx], color='orange', linestyle='--', alpha=0.5)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Position')
    ax.set_title('Hazard Events')
    ax.legend()

    if show:
        plt.show()

    return fig


def create_summary_figure(
    states_cp: Tuple,
    states_ob: Tuple,
    epoch: int,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Create a summary figure comparing change-point and oddball conditions.

    Parameters
    ----------
    states_cp : tuple
        State data for change-point condition.
    states_ob : tuple
        State data for oddball condition.
    epoch : int
        Epoch number.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    plt.Figure
        Summary figure with multiple subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top row: behavior plots
    plot_behavior(states_cp, "Change-Point", epoch, ax=axes[0, 0])
    plot_behavior(states_ob, "Oddball", epoch, ax=axes[0, 1])

    # Bottom row: hazard events
    plot_hazard_triggers(states_cp, ax=axes[1, 0])
    axes[1, 0].set_title("Change-Point: Hazard Events")

    plot_hazard_triggers(states_ob, ax=axes[1, 1])
    axes[1, 1].set_title("Oddball: Hazard Events")

    plt.suptitle(f"Epoch {epoch} Summary", fontsize=14)
    plt.tight_layout()

    return fig
