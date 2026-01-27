from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterator

import chess
import numpy as np
import tensorflow as tf

from src.agents.neural_network_agent import NeuralNetworkAgent

logger = logging.getLogger(__name__)


class NeuralNetworkPretrainer:
    """Utility class to train and evaluate a NeuralNetworkAgent.

    Expects training data in CSV format at data/tactic_evals.csv.

    The CSV format is:
            FEN,Evaluation,Move

    - FEN: Forsyth-Edwards Notation string for the position.
    - Evaluation: engine evaluation for the given move in centipawns
            (e.g. "+56", "-10", "0") or mate notation (e.g. "#+2", "#-1").
    - Move: best move in long algebraic from-to format without promotion
            suffix, e.g. "d3g6" (from d3 to g6).

    For training, FEN is encoded using TrainableAgent._encode_fen to a
    768-dimensional feature vector. The evaluation is normalized to [-1, 1].
    Targets are (4032, 2) tensors where only the entry corresponding to the
    labeled move is set to [evaluation, decisive] and all other moves are
    set to [0.0, 0.0]. Decisive is the absolute value of the normalized
    evaluation (0-1).
    """

    def __init__(
        self,
        agent: NeuralNetworkAgent,
        data_path: str = "data/lichess.csv",
        test_split: float = 0.2,
        shuffle: bool = True,
        random_seed: int = 42,
        max_samples: int = 50_000,
        read_limit: int = 500_000,
    ) -> None:
        self._agent = agent
        self._data_path = Path(data_path)
        self._test_split = float(test_split)
        self._shuffle = shuffle
        self._random_seed = random_seed
        self._max_samples = max_samples
        self._read_limit = read_limit
        self._train_indices: list[int] | None = None
        self._test_indices: list[int] | None = None
        self._total_samples: int = 0

    def _parse_row(self, row: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
        """Parse a single CSV row into (features, targets) pair.

        Returns None if the row is invalid or should be skipped.
        """
        if len(row) < 3:
            return None

        fen = row[0].strip()
        raw_eval = row[1].strip()
        raw_move = row[2].strip()
        if not fen or not raw_eval or not raw_move:
            return None

        # Normalize evaluation string to [-1, 1]
        norm_eval: float
        if raw_eval.startswith("#"):
            # Mate score, treat as max win/loss
            if "+" in raw_eval:
                norm_eval = 1.0
            elif "-" in raw_eval:
                norm_eval = -1.0
            else:
                norm_eval = 1.0
        else:
            try:
                cp = float(raw_eval)
            except ValueError:
                return None
            # Map centipawns to [-1, 1], clamped (e.g. +/-1000cp -> +/-1)
            norm_eval = max(-1.0, min(1.0, cp / 1000.0))

        # Support promotion moves like "e7e8Q" by ignoring the
        # trailing promotion piece and using only the from/to squares.
        if len(raw_move) < 4:
            return None

        base_move = raw_move[:4]
        from_square = base_move[0:2]
        to_square = base_move[2:4]
        try:
            from_idx = self._agent._square_to_index(from_square)
            to_idx = self._agent._square_to_index(to_square)
        except ValueError:
            return None

        if from_idx == to_idx:
            return None

        # Encode FEN to features
        feats = self._agent._encode_fen(fen)

        # Build (4032, 2) target: only the labeled move has non-zero values
        eval_val = float(norm_eval)
        decisive_val = abs(eval_val)
        move_targets = np.zeros((4032, 2), dtype=np.float32)

        # Map (from_idx, to_idx) with from_idx != to_idx into [0, 4031]
        if to_idx < from_idx:
            local = to_idx
        else:
            local = to_idx - 1

        move_id = from_idx * 63 + local
        move_targets[move_id, 0] = eval_val
        move_targets[move_id, 1] = decisive_val

        return (feats.astype(np.float32), move_targets)

    def _prepare_indices(self) -> None:
        """Scan the CSV to count samples and prepare train/test indices."""
        if not self._data_path.exists():
            raise FileNotFoundError(f"Training data not found at {self._data_path}")

        logger.info("Scanning training data from %s", self._data_path)

        valid_count = 0
        max_samples = self._max_samples
        with self._data_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip header
            header = next(reader, None)
            if header is None or len(header) < 3:
                raise ValueError(
                    "Expected header with at least three columns: FEN,Evaluation,Move"
                )

            rows_read = 0
            for row in reader:
                rows_read += 1
                if self._parse_row(row) is not None:
                    valid_count += 1
                    if rows_read >= self._read_limit:
                        break

        if valid_count == 0:
            raise ValueError("No valid data rows found in CSV")

        self._total_samples = valid_count
        indices = np.arange(valid_count)

        if self._shuffle:
            rng = np.random.default_rng(self._random_seed)
            rng.shuffle(indices)

        # If max_samples is smaller than available valid rows, take a random subset
        if max_samples is not None and max_samples > 0 and max_samples < valid_count:
            indices = indices[:max_samples]

        # Split into train/test
        split_idx = int(len(indices) * (1.0 - self._test_split))
        train_indices = indices[:split_idx].tolist()
        test_indices = indices[split_idx:].tolist()

        self._train_indices = train_indices
        self._test_indices = test_indices

        logger.info(
            "Data split: %d train samples, %d test samples",
            len(train_indices),
            len(test_indices),
        )

    def _data_generator(
        self, indices: list[int], mode: str = "moves"
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate (features, targets) pairs for the given indices.

        Args:
            indices: List of row indices to include.
            mode: "moves" for standard evaluation training, "validity" for legal move training.
        """
        if not self._data_path.exists():
            raise FileNotFoundError(f"Training data not found at {self._data_path}")

        index_set = set(indices)
        current_idx = 0

        with self._data_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)

            rows_read = 0
            for row in reader:
                rows_read += 1
                if mode == "moves":
                    result = self._parse_row(row)
                elif mode == "validity":
                    result = self._parse_row_validity(row)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                if result is not None:
                    if current_idx in index_set:
                        yield result
                    current_idx += 1

                if rows_read >= self._read_limit:
                    break

    def _create_dataset(self, indices: list[int], batch_size: int) -> tf.data.Dataset:
        """Create a TensorFlow Dataset from the given indices."""

        def generator():
            return self._data_generator(indices)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(768,), dtype=tf.float32),
                tf.TensorSpec(shape=(4032, 2), dtype=tf.float32),
            ),
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def train(self, epochs: int = 1, batch_size: int = 32) -> None:
        """Train the multi-task model by alternating between datasets."""

        if self._train_indices is None:
            self._prepare_indices()

        train_indices = self._train_indices
        assert train_indices is not None

        if self._agent.model is None:
            logger.info("No model instance found, creating a new one")
            self._agent.model = self._agent.create_model()

        logger.info("Starting training: epochs=%d, batch_size=%d", epochs, batch_size)
        steps_per_epoch = len(train_indices) // batch_size
        if steps_per_epoch == 0:
            logger.warning("Dataset too small for batch size, setting steps=1")
            steps_per_epoch = 1

        # We need to construct datasets that return TUPLES for the model inputs/outputs
        # to avoid KeyError: 0 in Keras internal handling of list-based outputs.

        # 1. Moves Dataset (trains output_moves, ignores output_valid)
        def moves_generator():
            gen = self._data_generator(train_indices, mode="moves")
            for x, y_moves in gen:
                # Target for validity is dummy (zeros), but weight is 0 so it's ignored
                y_valid = np.zeros((4032, 1), dtype=np.float32)
                yield (
                    x,
                    (y_moves, y_valid),
                    (1.0, 0.0),
                )

        moves_dataset = (
            tf.data.Dataset.from_generator(
                moves_generator,
                output_signature=(
                    tf.TensorSpec(shape=(768,), dtype=tf.float32),
                    (
                        tf.TensorSpec(shape=(4032, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(4032, 1), dtype=tf.float32),
                    ),
                    (
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                    ),
                ),
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # 2. Validity Dataset (trains output_valid, ignores output_moves)
        def validity_generator():
            gen = self._data_generator(train_indices, mode="validity")
            for x, y_valid in gen:
                # Target for moves is dummy, weight is 0
                y_moves = np.zeros((4032, 2), dtype=np.float32)
                yield (
                    x,
                    (y_moves, y_valid),
                    (0.0, 1.0),
                )

        valid_dataset = (
            tf.data.Dataset.from_generator(
                validity_generator,
                output_signature=(
                    tf.TensorSpec(shape=(768,), dtype=tf.float32),
                    (
                        tf.TensorSpec(shape=(4032, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(4032, 1), dtype=tf.float32),
                    ),
                    (
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                    ),
                ),
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .repeat()
        )

        # Training Loop: Alternate between tasks per epoch
        for epoch in range(epochs):
            # logger.info(f"Epoch {epoch + 1}/{epochs} - Task: Evaluation/Decisive")
            # self._agent.model.fit(
            #     moves_dataset,
            #     initial_epoch=epoch,
            #     epochs=epoch + 1,
            #     steps_per_epoch=steps_per_epoch,
            #     verbose=1,
            # )

            logger.info(f"Epoch {epoch + 1}/{epochs} - Task: Validity")
            self._agent.model.fit(
                valid_dataset,
                initial_epoch=epoch,
                epochs=epoch + 1,  # Note: this just continues training
                steps_per_epoch=steps_per_epoch,
                verbose=1,
            )

        # Save the trained models
        model_path = Path(self._agent.model_file_path)

        model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving trained model to %s", model_path)
        self._agent.model.save(model_path)

    def _parse_row_validity(
        self, row: list[str]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Parse row specifically for validity training (requires python-chess)."""
        if len(row) < 3:
            return None
        fen = row[0].strip()
        if not fen:
            return None

        # Encode FEN features
        feats = self._agent._encode_fen(fen)

        # Calculate valid moves mask
        valid_targets = np.zeros((4032, 1), dtype=np.float32)
        try:
            board = chess.Board(fen)
            for m in board.legal_moves:
                uci = m.uci()
                # Handle promotion (e7e8q -> e7e8)
                src = uci[:2]
                dst = uci[2:4]

                try:
                    f = self._agent._square_to_index(src)
                    t = self._agent._square_to_index(dst)
                    if f == t:
                        continue

                    loc = t if t < f else t - 1
                    mid = f * 63 + loc
                    if 0 <= mid < 4032:
                        valid_targets[mid, 0] = 1.0
                except ValueError:
                    continue
        except ValueError:
            return None

        return (feats.astype(np.float32), valid_targets)

    def evaluate(self) -> dict[str, float]:
        """Evaluate the multi-task model on the test set."""
        self._agent.load_model()

        if self._test_indices is None:
            self._prepare_indices()

        test_indices = self._test_indices
        assert test_indices is not None

        if self._agent.model is None:
            raise RuntimeError("Model is not initialized. Call train() first.")

        logger.info("Evaluating models on test set")
        steps = len(test_indices) // 32
        if steps == 0:
            steps = 1

        # Evaluate Move Task
        def moves_gen():
            gen = self._data_generator(test_indices, mode="moves")
            for x, y_moves in gen:
                y_valid = np.zeros((4032, 1), dtype=np.float32)
                yield (
                    x,
                    (y_moves, y_valid),
                    (1.0, 0.0),
                )

        test_moves_ds = tf.data.Dataset.from_generator(
            moves_gen,
            output_signature=(
                tf.TensorSpec(shape=(768,), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(4032, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(4032, 1), dtype=tf.float32),
                ),
                (
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                ),
            ),
        ).batch(32)

        # Evaluate Validity Task
        def valid_gen():
            gen = self._data_generator(test_indices, mode="validity")
            for x, y_valid in gen:
                y_moves = np.zeros((4032, 2), dtype=np.float32)
                yield (
                    x,
                    (y_moves, y_valid),
                    (0.0, 1.0),
                )

        test_valid_ds = tf.data.Dataset.from_generator(
            valid_gen,
            output_signature=(
                tf.TensorSpec(shape=(768,), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(4032, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(4032, 1), dtype=tf.float32),
                ),
                (
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                ),
            ),
        ).batch(32)

        logger.info("Evaluating Evaluation/Decisive Head...")
        moves_results = self._agent.model.evaluate(
            test_moves_ds, steps=steps, verbose=1, return_dict=True
        )
        assert isinstance(moves_results, dict)

        logger.info("Evaluating Validity Head...")
        valid_results = self._agent.model.evaluate(
            test_valid_ds, steps=steps, verbose=1, return_dict=True
        )
        assert isinstance(valid_results, dict)

        # moves_results: Valid head is weighted 0, so loss is just moves loss.
        # We access metrics by name to be robust and type-safe
        move_mae = moves_results["eval_mae"]

        # valid_results: Moves head is weighted 0.
        valid_f1 = valid_results.get("valid_f1_score", 0.0)
        valid_precision = valid_results.get("valid_precision", 0.0)
        valid_recall = valid_results.get("valid_recall", 0.0)

        logger.info("Move Model MAE: %.4f", move_mae)
        logger.info(
            "Validity Model F1: %.4f, Precision: %.4f, Recall: %.4f",
            valid_f1,
            valid_precision,
            valid_recall,
        )

        return {
            "mae": move_mae,
            "validity_f1": valid_f1,
            "validity_precision": valid_precision,
            "validity_recall": valid_recall,
        }

    def predict_starting_position(self) -> None:
        """Predict and display moves for the starting chess position."""
        self._agent.load_model()

        if self._agent.model is None:
            raise RuntimeError("Model is not initialized. Load or train a model first.")

        # Starting position FEN
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        logger.info("Predicting for starting position: %s", starting_fen)

        # Get actual legal moves using python-chess
        board = chess.Board(starting_fen)
        actual_legal_moves = set()
        for move in board.legal_moves:
            uci = move.uci()
            from_sq = uci[:2]
            to_sq = uci[2:4]
            actual_legal_moves.add((from_sq, to_sq))

        print(f"\n=== Actual Legal Moves (count: {len(actual_legal_moves)}) ===")
        for from_sq, to_sq in sorted(actual_legal_moves):
            print(f"{from_sq}{to_sq}")

        # Encode FEN
        features = self._agent._encode_fen(starting_fen)
        features_batch = np.expand_dims(features, axis=0)

        # Get predictions
        moves_pred, valid_pred = self._agent.model.predict(features_batch, verbose=0)

        # Extract predictions for the single position
        move_evals = moves_pred[0, :, 0]  # Evaluation for each move
        move_decisive = moves_pred[0, :, 1]  # Decisive score for each move
        move_valid = valid_pred[0, :, 0]  # Validity prediction for each move

        # Show model results for specific moves of interest
        for mv in ("d2d4", "e2e4"):
            from_sq = mv[:2]
            to_sq = mv[2:4]
            try:
                f_idx = self._agent._square_to_index(from_sq)
                t_idx = self._agent._square_to_index(to_sq)
                move_id = self._agent._move_to_id(f_idx, t_idx)
                m_eval = float(move_evals[move_id])
                m_dec = float(move_decisive[move_id])
                m_valid = float(move_valid[move_id])
                print(
                    f"\nModel prediction for {mv}: eval={m_eval:+.4f}, decisive={m_dec:.4f}, valid={m_valid:.4f}"
                )
            except Exception:
                print(f"\nModel prediction for {mv}: not available")

        # Print top 10 moves that the model predicts as valid (threshold 0.5),
        # sorted by the model's evaluation score.
        print("\n=== Top 10 Model-Predicted Valid Moves by Evaluation ===")
        model_valid = []
        threshold = 0.5
        for move_id in range(4032):
            from_idx = move_id // 63
            local = move_id % 63
            to_idx = local if local < from_idx else local + 1

            from_sq = self._agent._index_to_square(from_idx)
            to_sq = self._agent._index_to_square(to_idx)

            valid_score = float(move_valid[move_id])
            if valid_score <= threshold:
                continue

            eval_score = move_evals[move_id]
            decisive_score = move_decisive[move_id]
            is_legal = (from_sq, to_sq) in actual_legal_moves

            model_valid.append(
                (eval_score, from_sq, to_sq, decisive_score, valid_score, is_legal)
            )

        model_valid.sort(key=lambda x: -x[0])
        top_model_valid = model_valid[:10]

        if not top_model_valid:
            print("No moves predicted valid by the model at threshold 0.5.")
        else:
            for i, (
                eval_score,
                from_sq,
                to_sq,
                decisive_score,
                valid_score,
                is_legal,
            ) in enumerate(top_model_valid):
                print(
                    f"{i + 1:2d}. {from_sq}{to_sq}: eval={eval_score:+.4f}, decisive={decisive_score:.4f}, valid={valid_score:.4f} {'✓' if is_legal else '✗'}"
                )

        # Calculate validity accuracy
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        predicted_valid = []
        for move_id in range(4032):
            from_idx = move_id // 63
            local = move_id % 63
            to_idx = local if local < from_idx else local + 1

            from_sq = self._agent._index_to_square(from_idx)
            to_sq = self._agent._index_to_square(to_idx)

            is_predicted_valid = move_valid[move_id] > 0.5
            is_actually_legal = (from_sq, to_sq) in actual_legal_moves

            if is_predicted_valid and is_actually_legal:
                true_positives += 1
                predicted_valid.append((from_sq, to_sq, move_valid[move_id]))
            elif is_predicted_valid and not is_actually_legal:
                false_positives += 1
                predicted_valid.append((from_sq, to_sq, move_valid[move_id]))
            elif not is_predicted_valid and not is_actually_legal:
                true_negatives += 1
            elif not is_predicted_valid and is_actually_legal:
                false_negatives += 1

        # Derive a threshold that gives 100% precision (no illegal moves predicted).
        scored = []
        for move_id in range(4032):
            from_idx = move_id // 63
            local = move_id % 63
            to_idx = local if local < from_idx else local + 1

            from_sq = self._agent._index_to_square(from_idx)
            to_sq = self._agent._index_to_square(to_idx)
            score = float(move_valid[move_id])
            is_legal = (from_sq, to_sq) in actual_legal_moves
            scored.append((score, from_sq, to_sq, is_legal))

        scored.sort(key=lambda x: -x[0])

        prefix = []
        for score, from_sq, to_sq, is_legal in scored:
            if not is_legal:
                break
            prefix.append((from_sq, to_sq, score))

        if prefix:
            # Lowest score inside the precision=1.0 prefix
            threshold_p1 = prefix[-1][2]
        else:
            threshold_p1 = 1.0

        recall_p1 = len(prefix) / len(actual_legal_moves) if actual_legal_moves else 0.0

        total = true_positives + false_positives + true_negatives + false_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        print("\n=== Validity Prediction Accuracy ===")
        print(f"True Positives:  {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"True Negatives:  {true_negatives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\n=== 100% Precision Mode ===")
        print(f"Predicted Valid (precision=1.0): {len(prefix)}")
        print(f"Recall at precision=1.0:        {recall_p1:.4f}")
        print(f"Validity threshold (min score): {threshold_p1:.4f}")

        if prefix:
            print("\nMoves kept under precision=1.0:")
            for from_sq, to_sq, score in prefix:
                print(f"{from_sq}{to_sq}: {score:.4f} ✓")
        else:
            print("\nNo moves achieve precision=1.0 under current scoring.")

        print(
            f"\n=== Valid Moves According to Model (count: {len(predicted_valid)}) ==="
        )
        for from_sq, to_sq, score in sorted(predicted_valid, key=lambda x: -x[2]):
            is_legal = (from_sq, to_sq) in actual_legal_moves
            print(f"{from_sq}{to_sq}: {score:.4f} {'✓' if is_legal else '✗'}")


def main() -> int:
    """Train or evaluate a NeuralNetworkAgent from the command line."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate a NeuralNetworkAgent"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--username", type=str, required=True, help="Agent username"
    )
    train_parser.add_argument(
        "--read-limit",
        type=int,
        default=None,
        help="Maximum raw CSV rows to read when scanning/loading (stop early)",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an existing model")
    eval_parser.add_argument(
        "--username", type=str, required=True, help="Agent username"
    )
    eval_parser.add_argument(
        "--read-limit",
        type=int,
        default=None,
        help="Maximum raw CSV rows to read when scanning/loading (stop early)",
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict", help="Predict on starting position"
    )
    predict_parser.add_argument(
        "--username", type=str, required=True, help="Agent username"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create agent instance
    agent = NeuralNetworkAgent(
        username=args.username,
        password="",
        email="",
        classpath="",
        web_client=None,
    )

    tester = NeuralNetworkPretrainer(agent=agent)

    try:
        if args.command == "train":
            logger.info("Starting training...")
            tester.train()
            logger.info("Training complete")
            return 0

        elif args.command == "evaluate":
            logger.info("Starting evaluation...")
            results = tester.evaluate()
            logger.info("Evaluation complete: %s", results)
            return 0

        elif args.command == "predict":
            logger.info("Predicting starting position...")
            tester.predict_starting_position()
            return 0

    except Exception as e:
        logger.error("Error during %s: %s", args.command, e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
