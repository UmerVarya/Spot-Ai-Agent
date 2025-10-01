"""Adaptive reinforcement-learning based position sizing for Spot-AI.

Compared to the original Q-learning stub this module now tracks
state/action visit counts, adapts exploration based on recent reward
history, and exposes human-readable policy summaries so the sizing
layer can be monitored in production.
"""

import os
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional


ROOT_DIR = os.path.dirname(__file__)
POLICY_FILE = os.path.join(ROOT_DIR, "rl_policy.json")


class RLPositionSizer:
    """Q‑learning based position sizing agent."""

    def __init__(self,
                 actions: List[float] | None = None,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.02,
                 reward_clip: float = 5.0) -> None:
        # Allowed multipliers
        self.actions = actions or [0.5, 1.0, 1.5, 2.0]
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration probability
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.reward_clip = reward_clip
        self.q_table: Dict[str, Dict[float, float]] = {}
        self.state_action_visits: Dict[tuple[str, float], int] = defaultdict(int)
        self.reward_history: deque[float] = deque(maxlen=200)
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

    def _adaptive_epsilon(self) -> None:
        if not self.reward_history:
            return
        avg_reward = sum(self.reward_history) / len(self.reward_history)
        if avg_reward > 0.5:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        elif avg_reward < -0.5:
            self.epsilon = min(0.9, self.epsilon / self.epsilon_decay)

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
        best_action = max(action_values, key=action_values.get)
        self._adaptive_epsilon()
        return best_action

    def update(self, state: str, action: float, reward: float, next_state: Optional[str] = None) -> None:
        """Update the Q‑table for the action actually taken.

        Parameters
        ----------
        state : str
            The state used when the multiplier was chosen.
        action : float
            The position size multiplier applied to the trade.
        reward : float
            The realised reward (e.g. PnL percentage).
        next_state : str, optional
            When provided the update bootstraps from the value of the next
            observable state, enabling multi-step Q-learning in contexts
            where state transitions are available.
        """
        self._init_state(state)
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        bounded_reward = max(-self.reward_clip, min(self.reward_clip, reward))
        self.reward_history.append(bounded_reward)
        current_q = self.q_table[state][action]
        lookahead_state = next_state or state
        self._init_state(lookahead_state)
        next_max = max(self.q_table[lookahead_state].values())
        updated_q = current_q + self.alpha * (bounded_reward + self.gamma * next_max - current_q)
        self.q_table[state][action] = updated_q
        self.state_action_visits[(state, action)] += 1
        if self.epsilon > self.min_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self._save_policy()

    def policy_summary(self, top_n: int = 3) -> Dict[str, List[tuple[float, float]]]:
        """Return the top multipliers per state for introspection."""

        summary: Dict[str, List[tuple[float, float]]] = {}
        for state, actions in self.q_table.items():
            ranked = sorted(actions.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            summary[state] = ranked
        return summary
