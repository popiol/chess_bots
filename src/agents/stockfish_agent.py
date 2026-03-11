from __future__ import annotations

import logging
import random
import time
from typing import List

import chess
import numpy as np
from stockfish import Stockfish

from src.agents.trainable_agent import PredictionResult, TrainableAgent

logger = logging.getLogger(__name__)


# Module-level shared Stockfish instances per depth (lazy-initialized).
# Key: depth -> Stockfish instance
_SHARED_STOCKFISH_BY_DEPTH: dict[int, Stockfish] = {}


def _get_shared_stockfish(depth: int = 1) -> Stockfish:
    """Return a shared Stockfish instance for the requested depth.

    Creates and caches a Stockfish instance per depth so multiple depths can
    coexist without killing/recreating a single global engine.
    """
    global _SHARED_STOCKFISH_BY_DEPTH
    if depth not in _SHARED_STOCKFISH_BY_DEPTH:
        _SHARED_STOCKFISH_BY_DEPTH[depth] = Stockfish(
            path="stockfish", depth=depth, parameters={"Threads": 2, "Hash": 8}
        )
    return _SHARED_STOCKFISH_BY_DEPTH[depth]


class StockfishAgent(TrainableAgent):
    """Agent that queries Stockfish to produce move candidates."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Stockfish instance is initialized in load_state (depends on strength)
        self._sf: Stockfish | None = None
        self._strength: float | None = None

    def snapshot_state(self) -> dict:
        state = super().snapshot_state()
        state["strength"] = self._strength
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        if "strength" in state:
            self._strength = state["strength"]
        else:
            self._strength = random.random()
        assert self._strength is not None
        depth = 1 + round(self._strength * 2)
        self._sf = _get_shared_stockfish(depth)
        logger.info(
            "StockfishAgent loaded with strength=%.3f (depth=%d)", self._strength, depth
        )

    def _predict(self, fen: str, our_squares: List[str]) -> List[PredictionResult]:
        assert our_squares, "Our squares must be provided"

        start = time.time()
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        candidates = []

        assert self._strength is not None
        assert self._sf is not None

        eval_count = max(
            1, min(1 + int((1 - self._strength) * 4 - 0.5), len(legal_moves))
        )
        top_uci: set[str] = set()
        self._sf.set_fen_position(fen)
        try:
            top = self._sf.get_top_moves(eval_count)
            for item in top:
                assert isinstance(item["Move"], str)
                top_uci.add(item["Move"])
        except Exception:
            logger.exception("Exception while querying Stockfish for top moves")

        for move in legal_moves:
            uci = move.uci()
            if uci in top_uci:
                # Assign random evaluation to top moves
                eval_val = random.gauss(0.2, 0.2)
                decisive = random.random()
            else:
                eval_val = -1
                decisive = random.random()

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

        duration = time.time() - start
        logger.debug(
            "StockfishAgent._predict: moves_considered=%d samples=%d moves_returned=%d time=%.3fs",
            len(candidates),
            eval_count,
            len(results),
            duration,
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
        max_cp = 10000.0
        val = np.sign(raw) * (np.log1p(abs(raw)) / np.log1p(max_cp))
        return float(np.clip(val, -1.0, 1.0))
