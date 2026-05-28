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

            temp = board.copy()
            temp[r][c] = player

            if is_five(temp, r, c, player):
                return move

            temp2 = board.copy()
            temp2[r][c] = opponent

            if is_five(temp2, r, c, opponent):
                return move

            score = evaluate_move(board, move, player)

            if creates_open_four(temp, r, c, player):
                score += OPEN_FOUR

            if creates_open_three(temp, r, c, player):
                score += OPEN_THREE

            if creates_open_four(temp2, r, c, opponent):
                score += BLOCK_WIN

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
    
def evaluate_move(board, move, player):

    r, c = move

    if board[r][c] != 0:
        return -1e9

    score = 0

    center = BOARD_SIZE // 2

    dist = abs(r - center) + abs(c - center)

    score += CENTER_BONUS - dist

    temp = board.copy()
    temp[r][c] = player

    for dr, dc in DIRECTIONS:

        count = count_line(temp, r, c, dr, dc, player)

        score += count * CONNECT_BONUS

    return score

def count_line(board, r, c, dr, dc, player):

    count = 1

    nr = r + dr
    nc = c + dc

    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:

        if board[nr][nc] != player:
            break

        count += 1

        nr += dr
        nc += dc

    nr = r - dr
    nc = c - dc

    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:

        if board[nr][nc] != player:
            break

        count += 1

        nr -= dr
        nc -= dc

    return count

def is_five(board, r, c, player):

    for dr, dc in DIRECTIONS:

        if count_line(board, r, c, dr, dc, player) >= 5:
            return True

    return False


def creates_open_four(board, r, c, player):

    for dr, dc in DIRECTIONS:

        count = count_line(board, r, c, dr, dc, player)

        if count == 4:
            return True

    return False


def creates_open_three(board, r, c, player):

    for dr, dc in DIRECTIONS:

        count = count_line(board, r, c, dr, dc, player)

        if count == 3:
            return True

    return False
