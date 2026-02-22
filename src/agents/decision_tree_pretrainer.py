from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterator

import chess
import lightgbm as lgb
import numpy as np

from src.agents.decision_tree_agent import DecisionTreeAgent

logger = logging.getLogger(__name__)


class DecisionTreePretrainer:
    """Utility class to pretrain a DecisionTreeAgent from CSV game data.

    Expects training data in CSV format at data/lichess.csv.

    The CSV format is:
            FEN,Evaluation,Move

    - FEN: Forsyth-Edwards Notation string for the position.
    - Evaluation: engine evaluation in centipawns (e.g. "+56", "-10", "0")
            or mate notation (e.g. "#+2", "#-1").
    - Move: best move in UCI notation (e.g. "e2e4", "d3g6").

    Features are the 24-dim heuristic vector from
    DecisionTreeAgent._encode_move_features (12 metrics before + 12 after the move).
    Labels are normalized evaluation [-1, 1] and decisiveness [0, 1].

    Training calls lgb.train with init_model each epoch so trees are added
    on top of previous rounds (incremental warm-start).
    """

    def __init__(
        self,
        agent: DecisionTreeAgent,
        data_path: str = "data/lichess.csv",
        test_split: float = 0.2,
        shuffle: bool = True,
        random_seed: int = 42,
        max_samples: int = 500_000,
        read_limit: int = 100_000,
    ) -> None:
        self._agent = agent
        self._data_path = Path(data_path)
        self._test_split = test_split
        self._shuffle = shuffle
        self._random_seed = random_seed
        self._max_samples = max_samples
        self._read_limit = read_limit
        self._train_indices: list[int] | None = None
        self._test_indices: list[int] | None = None

    # ── row parsing ──────────────────────────────────────────────────

    def _parse_row(self, row: list[str]) -> tuple[np.ndarray, float, float] | None:
        """Parse a single CSV row into (features, eval, decisive) or None."""
        if len(row) < 3:
            return None

        fen = row[0].strip()
        raw_eval = row[1].strip()
        raw_move = row[2].strip()

        if not fen or not raw_eval or not raw_move:
            return None

        # Normalise evaluation to [-1, 1]
        if raw_eval.startswith("#"):
            norm_eval = 1.0 if "-" not in raw_eval else -1.0
        else:
            try:
                cp = float(raw_eval)
            except ValueError:
                return None
            norm_eval = max(-1.0, min(1.0, cp / 1000.0))

        # Parse move — accept optional promotion suffix (e.g. "e7e8q")
        if len(raw_move) < 4:
            return None
        try:
            board = chess.Board(fen)
            # Try with full UCI first (handles promotions); fall back to 4-char base
            for candidate in (raw_move, raw_move[:4]):
                try:
                    move = chess.Move.from_uci(candidate)
                    if move in board.legal_moves:
                        break
                except ValueError:
                    continue
            else:
                return None
        except ValueError:
            return None

        is_white = board.turn
        # Convert label from absolute (white-centric) to side-to-move-centric.
        # This keeps target semantics consistent with feature extraction and
        # prediction ranking (higher is better for player to move).
        if not is_white:
            norm_eval = -norm_eval
        try:
            feats = self._agent._encode_move_features(board, move, is_white)
        except Exception:
            return None

        return feats.astype(np.float32), float(norm_eval), abs(float(norm_eval))

    # ── index preparation ────────────────────────────────────────────

    def _prepare_indices(self) -> None:
        """Scan CSV to count valid samples and build train/test splits."""
        if not self._data_path.exists():
            raise FileNotFoundError(f"Training data not found at {self._data_path}")

        logger.info("Scanning data from %s", self._data_path)
        valid_count = 0
        with self._data_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or len(header) < 3:
                raise ValueError("Expected CSV header: FEN,Evaluation,Move")
            rows_read = 0
            for row in reader:
                rows_read += 1
                if self._parse_row(row) is not None:
                    valid_count += 1
                if rows_read >= self._read_limit:
                    break

        if valid_count == 0:
            raise ValueError("No valid rows found in CSV")

        indices = np.arange(valid_count)
        if self._shuffle:
            rng = np.random.default_rng(self._random_seed)
            rng.shuffle(indices)

        if self._max_samples and self._max_samples < valid_count:
            indices = indices[: self._max_samples]

        split_idx = int(len(indices) * (1.0 - self._test_split))
        self._train_indices = indices[:split_idx].tolist()
        self._test_indices = indices[split_idx:].tolist()

    # ── sample iteration ─────────────────────────────────────────────

    def _iter_samples(
        self, indices: list[int]
    ) -> Iterator[tuple[np.ndarray, float, float]]:
        """Yield (features, eval, decisive) tuples for the given row indices."""
        index_set = set(indices)
        current_idx = 0
        with self._data_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            rows_read = 0
            for row in reader:
                rows_read += 1
                result = self._parse_row(row)
                if result is not None:
                    if current_idx in index_set:
                        yield result
                    current_idx += 1
                if rows_read >= self._read_limit:
                    break

    # ── training ─────────────────────────────────────────────────────

    def train(self, epochs: int = 1) -> None:
        """Train LightGBM models on parsed CSV data.

        Each epoch collects all training samples and calls lgb.train with
        init_model so new trees are stacked on previous ones.
        """
        if self._train_indices is None:
            self._prepare_indices()
        assert self._train_indices is not None

        for epoch in range(1, epochs + 1):
            logger.info("Epoch %d/%d — collecting samples…", epoch, epochs)
            X_list: list[np.ndarray] = []
            y_eval_list: list[float] = []
            y_dec_list: list[float] = []

            for feats, y_e, y_d in self._iter_samples(self._train_indices):
                X_list.append(feats)
                y_eval_list.append(y_e)
                y_dec_list.append(y_d)

            if not X_list:
                logger.warning("No training samples collected; skipping epoch")
                continue

            X = np.asarray(X_list, dtype=np.float32)
            y_e_arr = np.asarray(y_eval_list, dtype=np.float32)
            y_d_arr = np.asarray(y_dec_list, dtype=np.float32)

            logger.info("Epoch %d — training on %d samples", epoch, len(X))

            ds_eval = lgb.Dataset(X, label=y_e_arr)
            self._agent.eval_model = lgb.train(
                self._agent._LGB_PARAMS_EVAL,
                ds_eval,
                num_boost_round=100,
                init_model=self._agent.eval_model,
            )

            ds_dec = lgb.Dataset(X, label=y_d_arr)
            self._agent.decisive_model = lgb.train(
                self._agent._LGB_PARAMS_DEC,
                ds_dec,
                num_boost_round=100,
                init_model=self._agent.decisive_model,
            )

            self._agent._save_models()
            logger.info(
                "Epoch %d complete — models saved to %s",
                epoch,
                self._agent.model_file,
            )

    # ── evaluation ───────────────────────────────────────────────────

    def evaluate(self) -> dict[str, float]:
        """Evaluate trained models on the test set (MAE for eval and decisive)."""
        if self._test_indices is None:
            self._prepare_indices()
        assert self._test_indices is not None

        if self._agent.eval_model is None or self._agent.decisive_model is None:
            raise RuntimeError("Models not trained. Call train() first.")

        X_list: list[np.ndarray] = []
        y_eval_list: list[float] = []
        y_dec_list: list[float] = []

        for feats, y_e, y_d in self._iter_samples(self._test_indices):
            X_list.append(feats)
            y_eval_list.append(y_e)
            y_dec_list.append(y_d)

        if not X_list:
            logger.warning("No test samples found")
            return {}

        X = np.asarray(X_list, dtype=np.float32)
        y_e_arr = np.asarray(y_eval_list, dtype=np.float32)
        y_d_arr = np.asarray(y_dec_list, dtype=np.float32)

        pred_e = np.asarray(self._agent.eval_model.predict(X))
        pred_d = np.asarray(self._agent.decisive_model.predict(X))

        eval_mae = float(np.mean(np.abs(pred_e - y_e_arr)))
        dec_mae = float(np.mean(np.abs(pred_d - y_d_arr)))

        logger.info(
            "Test set evaluation — eval_mae=%.4f  dec_mae=%.4f", eval_mae, dec_mae
        )
        return {"eval_mae": eval_mae, "dec_mae": dec_mae}

    def predict_starting_position(self) -> None:
        """Predict and display move evaluations for the starting chess position.

        This does not attempt to predict move validity — it evaluates the
        legal moves from the starting FEN using the trained LightGBM models.
        """
        # Ensure models are loaded
        self._agent._load_models()

        if self._agent.eval_model is None or self._agent.decisive_model is None:
            raise RuntimeError(
                "Models are not initialized. Load or train models first."
            )

        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        logger.info("Predicting for starting position: %s", starting_fen)

        board = chess.Board(starting_fen)
        legal_moves = []
        actual_legal_moves = set()
        for m in board.legal_moves:
            uci = m.uci()
            from_sq = uci[:2]
            to_sq = uci[2:4]
            legal_moves.append((from_sq, to_sq, m))
            actual_legal_moves.add((from_sq, to_sq))

        print(f"\n=== Actual Legal Moves (count: {len(actual_legal_moves)}) ===")
        for from_sq, to_sq in sorted(actual_legal_moves):
            print(f"{from_sq}{to_sq}")

        if not legal_moves:
            print("No legal moves found for starting position.")
            return

        # Build feature matrix for legal moves
        X_list = []
        for from_sq, to_sq, move in legal_moves:
            try:
                feats = self._agent._encode_move_features(board, move, board.turn)
            except Exception:
                feats = self._agent._encode_move_features(board, move, board.turn)
            X_list.append(feats)

        X = np.asarray(X_list, dtype=np.float32)

        pred_e = np.asarray(self._agent.eval_model.predict(X))
        pred_d = np.asarray(self._agent.decisive_model.predict(X))

        # Report specific moves of interest
        for mv in ("d2d4", "e2e4"):
            from_sq = mv[:2]
            to_sq = mv[2:4]
            found = False
            for i, (f, t, _) in enumerate(legal_moves):
                if f == from_sq and t == to_sq:
                    print(
                        f"\nModel prediction for {mv}: eval={float(pred_e[i]):+.4f}, decisive={float(pred_d[i]):.4f}"
                    )
                    found = True
                    break
            if not found:
                print(
                    f"\nModel prediction for {mv}: not available (illegal or not present)"
                )

        # Top N moves by evaluation
        model_list = []
        for i, (from_sq, to_sq, _) in enumerate(legal_moves):
            model_list.append((float(pred_e[i]), from_sq, to_sq, float(pred_d[i])))

        model_list.sort(key=lambda x: -x[0])
        top = model_list[:10]

        print("\n=== Top 10 Model-Predicted Moves by Evaluation ===")
        for i, (eval_score, from_sq, to_sq, dec_score) in enumerate(top):
            print(
                f"{i + 1:2d}. {from_sq}{to_sq}: eval={eval_score:+.4f}, decisive={dec_score:.4f}"
            )


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain a DecisionTreeAgent")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train LightGBM models from CSV")
    train_p.add_argument("--username", required=True, help="Agent username")
    train_p.add_argument("--epochs", type=int, default=1, help="Training epochs")

    eval_p = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_p.add_argument("--username", required=True, help="Agent username")

    predict_p = subparsers.add_parser("predict", help="Predict starting position")
    predict_p.add_argument("--username", required=True, help="Agent username")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    agent = DecisionTreeAgent(
        username=args.username,
        password="",
        email="",
        classpath="",
        web_client=None,
    )
    # Load existing models if present so training can continue from a checkpoint
    agent._load_models()

    pretrainer = DecisionTreePretrainer(agent=agent)

    try:
        if args.command == "train":
            logger.info("Starting training…")
            pretrainer.train(epochs=args.epochs)
            logger.info("Training complete")
            return 0
        elif args.command == "evaluate":
            logger.info("Starting evaluation…")
            results = pretrainer.evaluate()
            logger.info("Results: %s", results)
            return 0
        elif args.command == "predict":
            logger.info("Predicting starting position…")
            pretrainer.predict_starting_position()
            return 0
    except Exception:
        logger.error("Error during %s", args.command, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
