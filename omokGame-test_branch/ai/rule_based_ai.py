import random
import numpy as np

BOARD_SIZE = 15

WIN_SCORE = 100000
BLOCK_WIN = 90000
OPEN_FOUR = 50000
DOUBLE_THREE = 10000
OPEN_THREE = 3000
CENTER_BONUS = 50
CONNECT_BONUS = 10


DIRECTIONS = [
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
]


def get_candidate_moves(board, radius=2):

    stones = np.argwhere(board != 0)

    if len(stones) == 0:
        return [(BOARD_SIZE // 2, BOARD_SIZE // 2)]

    candidates = set()

    for r, c in stones:

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):

                nr = r + dr
                nc = c + dc

                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:

                    if board[nr][nc] == 0:
                        candidates.add((nr, nc))

    return list(candidates)


class RandomAI:

    def select_move(self, board, player):
        return random.choice(get_candidate_moves(board))


class WeakRuleAI:

    def select_move(self, board, player):

        moves = get_candidate_moves(board)

        best_score = -1e9
        best_move = None

        for move in moves:

            score = evaluate_move(board, move, player)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


class StrongRuleAI:

    def select_move(self, board, player):

        moves = get_candidate_moves(board)

        best_score = -1e9
        best_move = None

        opponent = 3 - player

        for move in moves:

            r, c = move

    return False
