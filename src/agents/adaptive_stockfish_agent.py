from __future__ import annotations

import logging
import random
import time
from typing import List

import chess

from src.agents.stockfish_agent import StockfishAgent, _get_shared_stockfish
from src.agents.trainable_agent import PredictionResult

logger = logging.getLogger(__name__)


class AdaptiveStockfishAgent(StockfishAgent):
    """Adaptive Stockfish-based agent."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sf = _get_shared_stockfish()

    def _predict(self, fen: str, our_squares: List[str]) -> List[PredictionResult]:
        """Adaptive Stockfish agent: choose strength from position evaluation."""
        assert our_squares, "Our squares must be provided"

        start = time.time()
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        candidates = []
        assert self._sf is not None, "Stockfish instance must be initialized"

        # Quick evaluation using the default shared engine
        self._sf.set_fen_position(fen)
        ev = self._sf.get_evaluation()

        # Extract raw centipawn-like value from evaluation dict
        raw_cp = None
        t = str(ev["type"])
        v = int(ev["value"])
        if t == "cp":
            raw_cp = v
        elif t == "mate":
            # map mate to a large centipawn value with sign
            raw_cp = 10000 if v > 0 else -10000

        converted = self._convert_stockfish_eval(raw_cp)
        perspective = converted if board.turn == chess.WHITE else -converted
        strength = -perspective

        # Choose depth based on strength (match StockfishAgent mapping)
        depth = 1 + round(strength * 2)
        sf = _get_shared_stockfish(depth)

        # Determine how many top moves to ask for (fewer when strength is high)
        eval_count = max(1, min(1 + int((1 - strength) * 4 - 0.5), len(legal_moves)))
        top_uci: set[str] = set()
        sf.set_fen_position(fen)
        top = sf.get_top_moves(eval_count)
        for item in top:
            if isinstance(item.get("Move"), str):
                assert isinstance(item["Move"], str)
                top_uci.add(item["Move"])

        for move in legal_moves:
            uci = move.uci()
            if uci in top_uci:
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
            "AdaptiveStockfishAgent._predict: depth=%d strength=%.3f moves_considered=%d samples=%d moves_returned=%d time=%.3fs",
            depth,
            strength,
            len(candidates),
            eval_count,
            len(results),
            duration,
        )

        return results
