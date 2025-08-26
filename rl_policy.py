"""
Reinforcement‑learning based position sizing for the Spot‑AI Agent.

This module implements a simple Q‑learning policy to adjust the
position sizing multiplier based on recent trading outcomes.  The goal
is to allocate more capital to high‑reward states and less to
underperforming ones.  While the basic implementation uses only the
previous trade outcome (``win``/``loss``/``neutral``) as the state, it
also supports richer context such as volatility regimes.  The action
space consists of discrete multipliers (e.g., ``[0.5, 1.0, 1.5, 2.0]``),
each representing a fraction of the default position size computed by
the agent.

This lightweight RL approach is an approximation to more sophisticated
actor–critic methods.  It learns a Q‑table mapping states and actions
to expected rewards.  After each trade, the Q‑value for the
corresponding state–action pair is updated.  Over time, the policy
tends to favour the multiplier that yields higher returns.

Usage::

    from rl_policy import RLPositionSizer
    from trade_utils import get_rl_state
    rl_sizer = RLPositionSizer()
    # during trade entry combine last outcome with volatility percentile
    state = get_rl_state(vol_percentile)
    multiplier = rl_sizer.select_multiplier(state)
    position_size = base_size * multiplier
    ...
    # after trade exits
    rl_sizer.update(state, multiplier, reward)  # reward = trade profit in percent

The learned Q‑table is stored in ``rl_policy.json`` in the module
directory, ensuring persistence across sessions.
"""

import os
import json
from typing import Dict, List


ROOT_DIR = os.path.dirname(__file__)
POLICY_FILE = os.path.join(ROOT_DIR, "rl_policy.json")


class RLPositionSizer:
    """Q‑learning based position sizing agent."""

    def __init__(self,
                 actions: List[float] | None = None,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1) -> None:
        # Allowed multipliers
        self.actions = actions or [0.5, 1.0, 1.5, 2.0]
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration probability
        self.q_table: Dict[str, Dict[float, float]] = {}
        self._load_policy()

    def _load_policy(self) -> None:
        """Load the Q‑table from disk if it exists."""
        if os.path.exists(POLICY_FILE):
            try:
                with open(POLICY_FILE, 'r') as f:
                    data = json.load(f)
                    # convert keys back to float for actions
                    self.q_table = {state: {float(a): v for a, v in actions.items()} for state, actions in data.items()}
            except Exception:
                self.q_table = {}

    def _save_policy(self) -> None:
        """Persist the Q‑table to disk."""
        try:
            with open(POLICY_FILE, 'w') as f:
                # convert float keys to strings for JSON serialization
                serializable = {state: {str(a): v for a, v in actions.items()} for state, actions in self.q_table.items()}
                json.dump(serializable, f, indent=2)
        except Exception:
            pass

    def _init_state(self, state: str) -> None:
        """Ensure a state has entries in the Q‑table."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

    def select_multiplier(self, state: str) -> float:
        """
        Choose a position size multiplier given the current state.

        With probability ``epsilon`` the agent explores a random action.  Otherwise
        it exploits the current best known action for the state.
        """
        self._init_state(state)
        import random
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Select the action with the maximum Q‑value
        action_values = self.q_table[state]
        return max(action_values, key=action_values.get)

    def update(self, state: str, action: float, reward: float) -> None:
        """Update the Q‑table for the action actually taken.

        Parameters
        ----------
        state : str
            The state used when the multiplier was chosen.
        action : float
            The position size multiplier applied to the trade.
        reward : float
            The realised reward (e.g. PnL percentage).
        """
        self._init_state(state)
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        current_q = self.q_table[state][action]
        next_max = max(self.q_table[state].values())
        updated_q = current_q + self.alpha * (reward + self.gamma * next_max - current_q)
        self.q_table[state][action] = updated_q
        self._save_policy()
