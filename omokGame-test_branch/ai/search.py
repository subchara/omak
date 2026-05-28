import math

from ai.rule_based_ai import evaluate_move
from ai.rule_based_ai import get_candidate_moves


class MinimaxSearch:

    def __init__(self, depth=2):
        self.depth = depth

    def search(self, board, player, candidates):

        best_move = None
        best_score = -math.inf

        for move in candidates:

            r, c = move

            temp = board.copy()
            temp[r][c] = player

            score = self.minimax(
                temp,
                self.depth - 1,
                -math.inf,
                math.inf,
                False,
                player
            )

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing, player):

        if depth == 0:
            return self.evaluate(board, player)

        current = player if maximizing else 3 - player

        moves = get_candidate_moves(board)

        if maximizing:

            value = -math.inf

            for move in moves:

                r, c = move

                temp = board.copy()
                temp[r][c] = current

                value = max(
                    value,
                    self.minimax(temp, depth - 1, alpha, beta, False, player)
                )

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        else:

            value = math.inf

            for move in moves:

                r, c = move

                temp = board.copy()
                temp[r][c] = current

                value = min(
        return score
