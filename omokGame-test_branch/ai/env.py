import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.Board import Board
from ai.rule_based_ai import get_candidate_moves


class GomokuEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent=None, board_size=15):
        super().__init__()

        self.board_size = board_size
        self.board = Board(board_size)
        self.opponent = opponent

        self.action_space = spaces.Discrete(board_size * board_size)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(4, board_size, board_size),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board.reset()

        return self._get_obs(), {}

    def step(self, action):
        r = action // self.board_size
        c = action % self.board_size

        reward = 0.0
        done = False

        if not self.action_masks()[action]:
            return self._get_obs(), -1.0, True, False, {}

        success = self.board.make_move(r, c)

        if not success:
            return self._get_obs(), -1.0, True, False, {}

        if self.board.is_over:
            reward = 1.0 if self.board.winner == 1 else -1.0
            return self._get_obs(), reward, True, False, {}

        reward += self._shape_reward(1)

        if self.opponent is not None:
            move = self.opponent.select_move(self.board.board, 2)

            if move:
                self.board.make_move(move[0], move[1])

        if self.board.is_over:
            reward = 1.0 if self.board.winner == 1 else -1.0
            done = True

        return self._get_obs(), reward, done, False, {}

    def _shape_reward(self, player):
        reward = 0.0
        center = self.board_size // 2

        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board.board[r][c] == player:
                    dist = abs(r - center) + abs(c - center)
                    reward += max(0.0, 0.01 * (7 - dist))

        return reward

    def _get_obs(self):
        current_player = self.board.current_player
        opponent = 3 - current_player

        ch1 = (self.board.board == current_player).astype(np.float32)
        ch2 = (self.board.board == opponent).astype(np.float32)
        ch3 = np.full_like(ch1, current_player - 1, dtype=np.float32)

        mask = self.action_masks().reshape(self.board_size, self.board_size)
        ch4 = mask.astype(np.float32)

        return np.stack([ch1, ch2, ch3, ch4], axis=0)

    def action_masks(self):
        mask = np.zeros(self.board_size * self.board_size, dtype=bool)

        for r, c in get_candidate_moves(self.board.board):
            idx = r * self.board_size + c
            mask[idx] = True

        return mask

    def render(self):
        print(self.board.board)
