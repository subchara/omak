from sb3_contrib import MaskablePPO

from ai.rule_based_ai import get_candidate_moves
from ai.rule_based_ai import evaluate_move
from ai.search import MinimaxSearch


class HybridAgent:
    def __init__(self, model_path, top_k=8):
        self.model = MaskablePPO.load(model_path)
        self.top_k = top_k
        self.search_engine = MinimaxSearch(depth=2)

    def select_move(self, board, player):
        candidates = get_candidate_moves(board)

        scored = []

        for move in candidates:
            score = evaluate_move(board, move, player)
            scored.append((score, move))

        scored.sort(reverse=True)

        top_moves = [x[1] for x in scored[:self.top_k]]

        return self.search_engine.search(board, player, top_moves)
