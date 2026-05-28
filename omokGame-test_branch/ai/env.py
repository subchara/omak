import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.Board import Board
from ai.rule_based_ai import get_candidate_moves


class GomokuEnv(gym.Env):

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
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.board.reset()

        return self._get_obs(), {}

    def step(self, action):

        r = action // self.board_size
        c = action % self.board_size

        reward = 0
        done = False

        if not self.action_masks()[action]:
            return self._get_obs(), -1.0, True, False, {}

        ok = self.board.make_move(r, c)

        if not ok:
            return self._get_obs(), -1.0, True, False, {}

        if self.board.is_over:

            if self.board.winner == 1:
                reward = 1.0
            else:
                reward = -1.0

            return self._get_obs(), reward, True, False, {}

        reward += self.shape_reward(1)

        if self.opponent is not None:

            move = self.opponent.select_move(self.board.board, 2)

            if move:
                self.board.make_move(move[0], move[1])

        if self.board.is_over:

            if self.board.winner == 1:
                reward = 1.0
            else:
                reward = -1.0

            done = True

        return self._get_obs(), reward, done, False, {}

    def shape_reward(self, player):

        reward = 0.0

        center = self.board_size // 2

        for r in range(self.board_size):
            for c in range(self.board_size):

                if self.board.board[r][c] == player:

                    dist = abs(r - center) + abs(c - center)

        return mask
