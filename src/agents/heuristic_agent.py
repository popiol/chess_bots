from __future__ import annotations

import logging
from typing import List

import chess

from src.agents.heuristic_evaluator import HeuristicEvaluator
from src.agents.trainable_agent import PredictionResult, TrainableAgent

logger = logging.getLogger(__name__)


class HeuristicAgent(TrainableAgent):
    """Simple heuristic agent that evaluates moves by material balance.

    For each legal move, it applies the move and computes total piece values
    for white and black. The evaluation returned is the post-move material
    advantage for the side that was to move, normalized to [-1, 1].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = HeuristicEvaluator()

    def _predict(self, fen: str, our_squares: list[str]) -> List[PredictionResult]:
        board = chess.Board(fen)
        is_white = board.turn  # True if agent to move is white, False if black

        results: List[PredictionResult] = []

        # If no legal moves, return empty list
        for move in board.legal_moves:
            b2 = board.copy()
            b2.push(move)

            eval_val, decisive = self.evaluator.evaluate_position(b2, is_white)

            uci = move.uci()
            from_sq = uci[0:2]
            to_sq = uci[2:4]

            results.append(
                PredictionResult(
                    from_sq=from_sq,
                    to_sq=to_sq,
                    evaluation=float(eval_val),
                    decisive=float(decisive),
                )
            )

        # Sort best moves first (higher eval better for agent)
        results.sort(key=lambda r: r.evaluation, reverse=True)
        return results[: self.prediction_count]
