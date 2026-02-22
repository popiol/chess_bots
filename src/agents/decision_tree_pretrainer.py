from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
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

    Features are the 46-dim heuristic vector from
    DecisionTreeAgent._encode_move_features (23 metrics before + 23 after the move).
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
        group_by_fen: bool = True,
    ) -> None:
        self._agent = agent
        self._data_path = Path(data_path)
        self._test_split = test_split
        self._shuffle = shuffle
        self._random_seed = random_seed
        self._max_samples = max_samples
        self._read_limit = read_limit
        self._group_by_fen = group_by_fen
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

        norm_eval *= 0.5  # scale down to avoid resignations

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
        valid_fens: list[str] = []
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
                    valid_fens.append(row[0].strip())
                if rows_read >= self._read_limit:
                    break

        if valid_count == 0:
            raise ValueError("No valid rows found in CSV")

        indices: np.ndarray
        if valid_count == 0:
            indices = np.array([], dtype=int)
        else:
            # initial indices in original (valid-row) order
            base_indices = list(range(valid_count))

            if self._group_by_fen:
                # group contiguous indices by FEN, preserving samples for the
                # same position together. Optionally shuffle the order of
                # positions (groups) instead of shuffling individual samples.
                fen_to_idxs: dict[str, list[int]] = defaultdict(list)
                for idx, fen in enumerate(valid_fens):
                    fen_to_idxs[fen].append(idx)

                groups = list(fen_to_idxs.values())
                if self._shuffle:
                    rng = np.random.default_rng(self._random_seed)
                    rng.shuffle(groups)

                flattened: list[int] = []
                for g in groups:
                    flattened.extend(g)

                indices = np.asarray(flattened, dtype=int)
            else:
                indices = np.asarray(base_indices, dtype=int)
                if self._shuffle:
                    rng = np.random.default_rng(self._random_seed)
                    rng.shuffle(indices)

        if self._max_samples and self._max_samples < len(indices):
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

    def _iter_samples_with_fen(
        self, indices: list[int]
    ) -> Iterator[tuple[str, str, np.ndarray, float, float]]:
        """Yield (fen, move_uci, features, eval, decisive) tuples for the given row indices.

        This keeps the original CSV move string alongside parsed features so
        evaluators can display move ordering information per position.
        """
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
                        fen = row[0].strip()
                        move_uci = row[2].strip()
                        yield fen, move_uci, result[0], result[1], result[2]
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
                num_boost_round=300,
            )

            ds_dec = lgb.Dataset(X, label=y_d_arr)
            self._agent.decisive_model = lgb.train(
                self._agent._LGB_PARAMS_DEC,
                ds_dec,
                num_boost_round=300,
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

    def evaluate_ranking(self) -> dict[str, float]:
        """Evaluate move ranking accuracy per position.

        Groups test samples by FEN.  For each position with at least two
        moves, computes the pairwise concordance ratio: the fraction of
        move pairs whose relative predicted ordering matches the true
        ordering.  Tied true evaluations are skipped.

        Returns a dict with ``avg_concordance`` (mean concordance across
        positions) and ``positions_evaluated`` (number of positions used).
        """
        if self._test_indices is None:
            self._prepare_indices()
        assert self._test_indices is not None

        if self._agent.eval_model is None:
            raise RuntimeError("Models not trained. Call train() first.")

        # Collect samples grouped by FEN, keeping original move strings
        groups: dict[str, list[tuple[str, np.ndarray, float]]] = defaultdict(list)
        for fen, move_uci, feats, y_e, _y_d in self._iter_samples_with_fen(
            self._test_indices
        ):
            groups[fen].append((move_uci, feats, y_e))

        # Keep only positions with >= 2 moves
        multi = {k: v for k, v in groups.items() if len(v) >= 2}

        if not multi:
            logger.warning("No positions with multiple moves found in test set")
            return {}

        concordant_counts: list[int] = []
        total_pairs_list: list[int] = []
        per_position_concordance: dict[str, float] = {}
        per_position_data: dict[str, dict] = {}
        for fen, samples in multi.items():
            move_strs = [s[0] for s in samples]
            X = np.asarray([s[1] for s in samples], dtype=np.float32)
            true_evals = np.array([s[2] for s in samples])
            pred_evals = np.asarray(self._agent.eval_model.predict(X))

            # Pairwise concordance
            n = len(samples)
            concordant = 0
            total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    true_diff = true_evals[i] - true_evals[j]
                    pred_diff = pred_evals[i] - pred_evals[j]
                    if true_diff == 0:
                        continue  # skip ties in ground truth
                    total += 1
                    if (true_diff > 0 and pred_diff > 0) or (
                        true_diff < 0 and pred_diff < 0
                    ):
                        concordant += 1

            if total > 0:
                ratio = concordant / total
                concordant_counts.append(concordant)
                total_pairs_list.append(total)
                per_position_concordance[fen] = ratio
                per_position_data[fen] = {
                    "moves": move_strs,
                    "true": true_evals,
                    "pred": pred_evals,
                }

        if not per_position_concordance:
            logger.warning("No valid position groups for ranking evaluation")
            return {}

        # Weighted average concordance: sum(concordant pairs) / sum(total pairs)
        total_concordant = sum(concordant_counts)
        total_pairs = sum(total_pairs_list)
        avg_concordance = (
            float(total_concordant / total_pairs) if total_pairs > 0 else 0.0
        )
        positions_evaluated = len(per_position_concordance)

        # Average number of samples per multi-move position
        avg_group_size = float(np.mean([len(v) for v in multi.values()]))

        logger.info(
            "Ranking evaluation — avg_concordance=%.4f  across %d positions  avg_group_size=%.2f",
            avg_concordance,
            positions_evaluated,
            avg_group_size,
        )

        # Find and display the worst concordance group (lowest ratio) among
        # positions with at least 9 samples. Fall back to the global worst
        # if no position meets the minimum size.
        candidate_fens = [
            f for f, s in multi.items() if len(s) >= 9 and f in per_position_concordance
        ]
        if candidate_fens:
            worst_fen = min(candidate_fens, key=lambda f: per_position_concordance[f])
        else:
            # use items() to satisfy static type checkers
            worst_fen = min(per_position_concordance.items(), key=lambda kv: kv[1])[0]
        worst_ratio = per_position_concordance[worst_fen]
        worst_data = per_position_data[worst_fen]

        logger.info(
            "Worst-position concordance=%.4f  FEN=%s",
            worst_ratio,
            worst_fen,
        )

        # Display correct (true) ordering and predicted ordering for the worst group
        true_order_idx = list(np.argsort(-worst_data["true"]))
        pred_order_idx = list(np.argsort(-worst_data["pred"]))

        logger.info("Correct order (best -> worst):")
        for rank, idx in enumerate(true_order_idx, start=1):
            mv = worst_data["moves"][idx]
            tval = worst_data["true"][idx]
            pval = float(worst_data["pred"][idx])
            logger.info("  %2d. %s  true=%.4f  pred=%.4f", rank, mv, tval, pval)

        logger.info("Predicted order (best -> worst):")
        for rank, idx in enumerate(pred_order_idx, start=1):
            mv = worst_data["moves"][idx]
            tval = worst_data["true"][idx]
            pval = float(worst_data["pred"][idx])
            logger.info("  %2d. %s  true=%.4f  pred=%.4f", rank, mv, tval, pval)
        return {
            "avg_concordance": avg_concordance,
            "positions_evaluated": positions_evaluated,
            "avg_group_size": avg_group_size,
        }

    def predict_position(self, fen: str) -> None:
        """Predict and display move evaluations for the given FEN.

        This does not attempt to predict move validity — it evaluates the
        legal moves from `fen` using the trained LightGBM models.
        """
        # Ensure models are loaded
        self._agent._load_models()

        if self._agent.eval_model is None or self._agent.decisive_model is None:
            raise RuntimeError(
                "Models are not initialized. Load or train models first."
            )

        logger.info("Predicting for position: %s", fen)

        board = chess.Board(fen)
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

    rank_p = subparsers.add_parser("ranking", help="Evaluate move ranking accuracy")
    rank_p.add_argument("--username", required=True, help="Agent username")

    predict_p = subparsers.add_parser("predict", help="Predict starting position")
    predict_p.add_argument("--username", required=True, help="Agent username")
    predict_p.add_argument(
        "--fen",
        required=False,
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="FEN string for the position to predict (default: starting position)",
    )

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
        elif args.command == "ranking":
            logger.info("Starting ranking evaluation…")
            results = pretrainer.evaluate_ranking()
            logger.info("Ranking results: %s", results)
            return 0
        elif args.command == "predict":
            logger.info("Predicting position…")
            pretrainer.predict_position(args.fen)
            return 0
    except Exception:
        logger.error("Error during %s", args.command, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
