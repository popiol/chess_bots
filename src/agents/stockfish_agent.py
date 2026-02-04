from __future__ import annotations

import logging
import random
from typing import List

import numpy as np
from stockfish import Stockfish

from src.agents.trainable_agent import PredictionResult, TrainableAgent

logger = logging.getLogger(__name__)


class StockfishAgent(TrainableAgent):
    """Agent that queries Stockfish to produce move candidates."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stockfish_path = "stockfish"
        self._sf = Stockfish(path=self.stockfish_path)
        self._sf.set_depth(1)

    def _predict(self, fen: str, our_squares: List[str]) -> List[PredictionResult]:
        assert our_squares, "Our squares must be provided"

        # Set position
        self._sf.set_fen_position(fen)

        # Get top moves
        moves_info = self._sf.get_top_moves(self.prediction_count, verbose=False)

        results: List[PredictionResult] = []

        # moves_info is a list of dicts with keys like 'Move', 'Centipawn', 'Mate'
        for item in moves_info[: self.prediction_count]:
            # 'Move' should contain the UCI move string (e.g., 'e2e4' or a pv string)
            uci = item.get("Move")
            assert isinstance(uci, str)

            from_sq = uci[0:2]
            to_sq = uci[2:4]

            # Centipawn or Mate (one will be None)
            cp = item.get("Centipawn")
            mate = item.get("Mate")
            if cp is not None:
                eval_raw = int(cp)
            elif mate is not None:
                # Mate in N -> large value with sign
                eval_raw = 100000 if int(mate) > 0 else -100000
            else:
                eval_raw = None

            eval_val = self._convert_stockfish_eval(eval_raw) + random.gauss(0, 0.4)
            decisive = abs(eval_val) + random.gauss(0, 0.4)

            results.append(
                PredictionResult(
                    from_sq=from_sq,
                    to_sq=to_sq,
                    evaluation=eval_val,
                    decisive=decisive,
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
