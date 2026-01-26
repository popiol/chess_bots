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
        data_path: str = "data/tactic_evals.csv",
        test_split: float = 0.2,
        shuffle: bool = True,
        random_seed: int = 42,
        max_samples: int = 100_000,
    ) -> None:
        self._agent = agent
        self._data_path = Path(data_path)
        self._test_split = float(test_split)
        self._shuffle = shuffle
        self._random_seed = random_seed
        self._max_samples = max_samples
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

            for row in reader:
                if self._parse_row(row) is not None:
                    valid_count += 1
                    if valid_count >= max_samples:
                        break

        if valid_count == 0:
            raise ValueError("No valid data rows found in CSV")

        self._total_samples = valid_count
        indices = np.arange(valid_count)

        if self._shuffle:
            rng = np.random.default_rng(self._random_seed)
            rng.shuffle(indices)

        split_idx = int(valid_count * (1.0 - self._test_split))
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
        max_samples = self._max_samples

        with self._data_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)

            for row in reader:
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
                    if current_idx >= max_samples:
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
        )

        # Training Loop: Alternate between tasks per epoch
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs} - Task: Validity")
            self._agent.model.fit(
                valid_dataset,
                initial_epoch=epoch,
                epochs=epoch + 1,  # Note: this just continues training
                steps_per_epoch=steps_per_epoch,
                verbose=0,
            )

            logger.info(f"Epoch {epoch + 1}/{epochs} - Task: Evaluation/Decisive")
            self._agent.model.fit(
                moves_dataset,
                initial_epoch=epoch,
                epochs=epoch + 1,
                steps_per_epoch=steps_per_epoch,
                verbose=0,
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
        valid_acc = valid_results["valid_acc"]

        logger.info("Move Model MAE: %.4f", move_mae)
        logger.info("Validity Model Acc: %.4f", valid_acc)

        return {"mae": move_mae, "validity_acc": valid_acc}


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
        "--data-path",
        type=str,
        default="data/tactic_evals.csv",
        help="Path to training data CSV (default: data/tactic_evals.csv)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    train_parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    train_parser.add_argument(
        "--no-shuffle", action="store_true", help="Disable shuffling of training data"
    )
    train_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an existing model")
    eval_parser.add_argument(
        "--username", type=str, required=True, help="Agent username"
    )
    eval_parser.add_argument(
        "--data-path",
        type=str,
        default="data/tactic_evals.csv",
        help="Path to test data CSV (default: data/tactic_evals.csv)",
    )
    eval_parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    eval_parser.add_argument(
        "--no-shuffle", action="store_true", help="Disable shuffling of data"
    )
    eval_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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

    # Create tester
    tester = NeuralNetworkPretrainer(
        agent=agent,
        data_path=args.data_path,
        test_split=args.test_split,
        shuffle=not args.no_shuffle,
        random_seed=args.random_seed,
    )

    try:
        if args.command == "train":
            logger.info("Starting training...")
            tester.train(epochs=args.epochs, batch_size=args.batch_size)
            logger.info("Training complete")
            return 0

        elif args.command == "evaluate":
            logger.info("Starting evaluation...")
            results = tester.evaluate()
            logger.info("Evaluation complete: %s", results)
            return 0

    except Exception as e:
        logger.error("Error during %s: %s", args.command, e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
