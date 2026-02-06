from __future__ import annotations

import logging
import random
from typing import List

import chess
import numpy as np
from stockfish import Stockfish

from src.agents.trainable_agent import PredictionResult, TrainableAgent

logger = logging.getLogger(__name__)


class StockfishAgent(TrainableAgent):
    """Agent that queries Stockfish to produce move candidates."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stockfish_path = "stockfish"
        self._sf = Stockfish(
            path=self.stockfish_path, parameters={"Hash": 8, "Threads": 1}
        )
        self._sf.set_depth(1)

    def _predict(self, fen: str, our_squares: List[str]) -> List[PredictionResult]:
        assert our_squares, "Our squares must be provided"

        # Use python-chess to get all legal moves
        board = chess.Board(fen)
        candidates = []

        for move in board.legal_moves:
            uci = move.uci()

            # Set position and make move to evaluate resulting position
            self._sf.set_fen_position(fen)
            self._sf.make_moves_from_current_position([uci])

            # Get evaluation of the position AFTER the move
            # This evaluation is from the perspective of the side to move (the opponent)
            eval_info = self._sf.get_evaluation()

            eval_type = eval_info.get("type")
            eval_val_raw = eval_info.get("value")
            assert isinstance(eval_val_raw, int)

            if eval_type == "mate":
                raw_score = 100000 if eval_val_raw > 0 else -100000
            else:
                raw_score = eval_val_raw

            # Invert score to get value for US (the side that made the move)
            my_raw_score = -1 * raw_score

            # Convert to -1..1 scale
            base_eval = self._convert_stockfish_eval(my_raw_score)

            eval_val = base_eval + random.gauss(0, 0.05)
            decisive = abs(base_eval) + random.gauss(0, 0.05)

            candidates.append(
                {
                    "from_sq": uci[0:2],
                    "to_sq": uci[2:4],
                    "evaluation": eval_val,
                    "decisive": decisive,
                }
            )

        # Sort moves by evaluation (descending)
        candidates.sort(key=lambda x: x["evaluation"], reverse=True)

        results: List[PredictionResult] = []
        for c in candidates[: self.prediction_count]:
            results.append(
                PredictionResult(
                    from_sq=c["from_sq"],
                    to_sq=c["to_sq"],
                    evaluation=c["evaluation"],
                    decisive=c["decisive"],
                )
            )

        return results

    def _convert_stockfish_eval(self, raw: int | None) -> float:
        """Convert Stockfish evaluation to -1..1 scale.

        - If raw is None, return 0.0
        - If raw is centipawns (int), map via logarithmic scale clamped to [-1,1]
        - Large mate values are mapped to +/-1.0
        """
        if raw is None:
            return 0.0

        # Logarithmic scaling used to reduce impact of large values
        max_cp = 1000.0
        val = np.sign(raw) * (np.log1p(abs(raw)) / np.log1p(max_cp))
        return float(np.clip(val, -1.0, 1.0))
