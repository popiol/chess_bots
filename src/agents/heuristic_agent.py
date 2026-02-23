from __future__ import annotations

import logging
import random
import time
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
        mover_is_white = board.turn  # True if agent to move is white, False if black

        results: List[PredictionResult] = []
        start = time.perf_counter()

        # Fast pre-evaluation: score all legal moves with evaluate_fast,
        # pick top-N candidates, then fully evaluate those with evaluate_position.
        fast_rows: list[tuple[chess.Move, float]] = []
        for move in board.legal_moves:
            b2 = board.copy()
            b2.push(move)
            fast_eval, _ = self.evaluator.evaluate_fast(b2, not mover_is_white)
            fast_eval = -fast_eval
            fast_rows.append((move, fast_eval))

        fast_rows.sort(key=lambda r: r[1], reverse=True)
        top_n = min(10, len(fast_rows))
        candidates = [r[0] for r in fast_rows[:top_n]]

        for move in candidates:
            b2 = board.copy()
            b2.push(move)

            # Full evaluation from opponent perspective then invert
            eval_val, decisive = self.evaluator.evaluate_position(
                b2, not mover_is_white
            )
            eval_val = -eval_val

            eval_val += random.gauss(0, 0.01)
            decisive += random.gauss(0, 0.01)

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

        duration = time.perf_counter() - start
        logger.info(
            "HeuristicAgent._predict: %.4fs (fast pre-eval + full eval) for %d candidates",
            duration,
            len(results),
        )

        return results[: self.prediction_count]
