from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import List

import chess
import lightgbm as lgb
import numpy as np

from src.agents.heuristic_evaluator import HeuristicEvaluator
from src.agents.trainable_agent import (
    PredictionResult,
    TrainableAgent,
)

logger = logging.getLogger(__name__)


class DecisionTreeAgent(TrainableAgent):
    """Agent that uses LightGBM gradient-boosted decision trees for move prediction.

    Two models are maintained:
    - eval_model: regression model predicting position evaluation (-1..1)
    - decisive_model: regression model predicting decisiveness (0..1)

    Each model is trained incrementally after every game via warm-starting
    (init_model parameter).  Models are persisted as a single pickle file.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_dir = Path("models")
        self.model_file = self.model_dir / f"{self.username}.lgbm"
        self.eval_model: lgb.Booster | None = None
        self.decisive_model: lgb.Booster | None = None
        self.heuristic = HeuristicEvaluator()

    # ── persistence ──────────────────────────────────────────────────

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._load_models()

    def _load_models(self) -> None:
        # Try user-specific model first, then fall back to shared base model.
        try:
            if self.model_file.exists():
                with self.model_file.open("rb") as f:
                    data = pickle.load(f)
                self.eval_model = data.get("eval")
                self.decisive_model = data.get("dec")
                logger.info(
                    "Loaded LightGBM models from %s",
                    self.model_file,
                    extra={"username": self.username},
                )
                return
        except Exception:
            logger.exception(
                "Failed to load user LightGBM model; trying base model",
                extra={"username": self.username},
            )

        try:
            base_file = self.model_dir / "base.lgbm"
            with base_file.open("rb") as f:
                data = pickle.load(f)
            self.eval_model = data.get("eval")
            self.decisive_model = data.get("dec")
            logger.info(
                "Loaded base LightGBM models from %s",
                base_file,
                extra={"username": self.username},
            )
            return
        except Exception:
            logger.exception(
                "Failed to load base LightGBM model; starting fresh",
                extra={"username": self.username},
            )

        self.eval_model = None
        self.decisive_model = None

    def _save_models(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with self.model_file.open("wb") as f:
            pickle.dump({"eval": self.eval_model, "dec": self.decisive_model}, f)

    # ── encoding ─────────────────────────────────────────────────────

    def _extract_metrics(self, board: chess.Board, is_white: bool) -> np.ndarray:
        """Extract the 12 heuristic metrics for a position as a flat array."""
        h = self.heuristic
        material = h._material_eval(board, is_white)
        mobility = h._mobility_eval(board, is_white)
        king = h._king_safety_eval(board, is_white)
        castling = h._castling_bonus(board, is_white)
        check = h._check_eval(board, is_white)
        our_atk, opp_atk = h._profitable_attack_eval(board, is_white)
        center = h._center_control_eval(board, is_white)
        undeveloped = h._undeveloped_pieces_eval(board, is_white)
        doubled = h._doubled_pawns_eval(board, is_white)
        isolated = h._isolated_pawns_eval(board, is_white)
        passed = h._passed_pawns_eval(board, is_white)
        return np.array(
            [
                material,
                mobility,
                king,
                castling,
                check,
                our_atk,
                opp_atk,
                center,
                undeveloped,
                doubled,
                isolated,
                passed,
            ],
            dtype=np.float32,
        )

    def _encode_move_features(
        self, board_before: chess.Board, move: chess.Move, is_white: bool
    ) -> np.ndarray:
        """Encode a (position, move) pair as a 24-dim feature vector.

        12 heuristic metrics for the position before the move, plus
        12 heuristic metrics for the position after the move.
        The tree model can learn both absolute values and deltas.
        """
        before = self._extract_metrics(board_before, is_white)
        board_after = board_before.copy()
        board_after.push(move)
        after = self._extract_metrics(board_after, is_white)
        return np.concatenate([before, after])

    # ── prediction ───────────────────────────────────────────────────

    def _predict(self, fen: str, our_squares: List[str]) -> List[PredictionResult]:
        assert our_squares
        start = time.time()

        board = chess.Board(fen)
        is_white = board.turn

        # Build feature matrix for all legal moves using heuristic metrics
        candidates: list[tuple[str, str, chess.Move]] = []
        rows: list[np.ndarray] = []
        for move in board.legal_moves:
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            rows.append(self._encode_move_features(board, move, is_white))
            candidates.append((from_sq, to_sq, move))

        if not candidates:
            return []

        X = np.asarray(rows, dtype=np.float32)

        # Require trained models for prediction. Let any exceptions surface.
        if self.eval_model is None or self.decisive_model is None:
            raise RuntimeError(
                "DecisionTreeAgent models are not initialized. Load or train models before calling _predict"
            )

        # Ensure predictions are numpy arrays (avoid sparse-matrix indexing errors)
        model_evals = np.asarray(self.eval_model.predict(X))
        model_decs = np.asarray(self.decisive_model.predict(X))

        results: list[PredictionResult] = []
        for i, (from_sq, to_sq, move) in enumerate(candidates):
            final_eval = float(model_evals[i])
            final_dec = float(model_decs[i])
            results.append(
                PredictionResult(
                    from_sq=from_sq,
                    to_sq=to_sq,
                    evaluation=final_eval,
                    decisive=final_dec,
                )
            )

        results.sort(key=lambda r: r.evaluation, reverse=True)

        duration = time.time() - start
        logger.info(
            "DecisionTreeAgent._predict: moves=%d returned=%d time=%.3fs",
            len(results),
            min(len(results), self.prediction_count),
            duration,
            extra={"username": self.username},
        )
        return results[: self.prediction_count]

    # ── training ─────────────────────────────────────────────────────

    _LGB_PARAMS_EVAL = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "verbose": -1,
    }

    _LGB_PARAMS_DEC = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "verbose": -1,
    }

    def _on_game_end(self, score: int | None, reason: str | None) -> None:
        if score is None:
            logger.info(
                "Game ended without score (reason=%s); skipping training.",
                reason,
                extra={"username": self.username},
            )
            return

        if self._moves_made < 2 or (
            reason in ["timeout", "agreement"] and self._moves_made < 10
        ):
            logger.info(
                "Skipping training: moves_made=%d reason=%s",
                self._moves_made,
                reason,
                extra={"username": self.username},
            )
            return

        if not self._decision_history:
            return

        # Build training data from our decisions
        X_rows: list[np.ndarray] = []
        y_eval: list[float] = []
        y_dec: list[float] = []
        for d in self._decision_history:
            board = chess.Board(d.fen)
            move = chess.Move.from_uci(d.from_sq + d.to_sq)
            is_white = board.turn
            X_rows.append(self._encode_move_features(board, move, is_white))
            y_eval.append(float(score))
            y_dec.append(float(abs(score)))

        # Opponent moves (if tracked)
        for d in getattr(self, "_opponent_move_history", []):
            board = chess.Board(d.fen)
            move = chess.Move.from_uci(d.from_sq + d.to_sq)
            is_white = board.turn
            X_rows.append(self._encode_move_features(board, move, is_white))
            y_eval.append(-float(score))
            y_dec.append(float(abs(score)))

        if not X_rows:
            return

        X = np.asarray(X_rows, dtype=np.float32)
        y_e = np.asarray(y_eval, dtype=np.float32)
        y_d = np.asarray(y_dec, dtype=np.float32)

        try:
            ds_eval = lgb.Dataset(X, label=y_e)
            self.eval_model = lgb.train(
                self._LGB_PARAMS_EVAL,
                ds_eval,
                num_boost_round=20,
                init_model=self.eval_model,
            )

            ds_dec = lgb.Dataset(X, label=y_d)
            self.decisive_model = lgb.train(
                self._LGB_PARAMS_DEC,
                ds_dec,
                num_boost_round=20,
                init_model=self.decisive_model,
            )

            self._save_models()
            logger.info(
                "Trained LightGBM on game end (score=%s, samples=%d) and saved to %s",
                score,
                len(X),
                self.model_file,
                extra={"username": self.username},
            )
        except Exception:
            logger.exception(
                "Failed to train LightGBM models",
                extra={"username": self.username},
            )

    def _handle_move_failure(self, fen: str, from_square: str, to_square: str) -> None:
        """Train the model that this move is bad by giving it a negative eval target."""
        board = chess.Board(fen)
        move = chess.Move.from_uci(from_square + to_square)
        is_white = board.turn
        X = self._encode_move_features(board, move, is_white).reshape(1, -1)
        y_e = np.array([-1.0], dtype=np.float32)
        y_d = np.array([0.0], dtype=np.float32)

        try:
            ds_eval = lgb.Dataset(X, label=y_e)
            self.eval_model = lgb.train(
                self._LGB_PARAMS_EVAL,
                ds_eval,
                num_boost_round=5,
                init_model=self.eval_model,
            )

            ds_dec = lgb.Dataset(X, label=y_d)
            self.decisive_model = lgb.train(
                self._LGB_PARAMS_DEC,
                ds_dec,
                num_boost_round=5,
                init_model=self.decisive_model,
            )

            logger.info(
                "Trained LightGBM on invalid move %s->%s",
                from_square,
                to_square,
                extra={"username": self.username},
            )
        except Exception:
            logger.exception(
                "Failed to train LightGBM on move failure",
                extra={"username": self.username},
            )
