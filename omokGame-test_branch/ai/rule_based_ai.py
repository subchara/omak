import random
import numpy as np

WIN_SCORE = 100000
BLOCK_WIN = 90000
OPEN_THREE = 3000
CENTER_BONUS = 50
CONNECT_BONUS = 10


def get_candidate_moves(board, radius=2):
    size = len(board)

    stones = np.argwhere(board != 0)

    if len(stones) == 0:
        return [(size // 2, size // 2)]

    candidates = set()

    for r, c in stones:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc

                if 0 <= nr < size and 0 <= nc < size:
                    if board[nr][nc] == 0:
                        candidates.add((nr, nc))

    return list(candidates)


class RandomAI:
    def select_move(self, board, player):
        return random.choice(get_candidate_moves(board))


class WeakRuleAI:
    def select_move(self, board, player):
        moves = get_candidate_moves(board)

        best_move = None
        best_score = -1e9

        for move in moves:
            score = evaluate_move(board, move, player)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


class StrongRuleAI(WeakRuleAI):
    pass


def evaluate_move(board, move, player):
    r, c = move

    if board[r][c] != 0:
        return -1e9

    score = 0

    center = len(board) // 2
    dist = abs(r - center) + abs(c - center)

    score += CENTER_BONUS - dist

    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    ]

    temp = board.copy()
    temp[r][c] = player

    for dr, dc in directions:
        nr = r + dr
        nc = c + dc

        if 0 <= nr < len(board) and 0 <= nc < len(board):
            if temp[nr][nc] == player:
                score += CONNECT_BONUS

    return score
