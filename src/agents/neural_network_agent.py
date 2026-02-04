from __future__ import annotations

import logging
import random
from pathlib import Path

import chess
import numpy as np
import tensorflow as tf

from src.agents.heuristic_evaluator import HeuristicEvaluator
from src.agents.trainable_agent import (
    PIECE_TYPE_TO_INDEX,
    PredictionResult,
    TrainableAgent,
)

logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable()
def weighted_mse(y_true, y_pred):
    """Standard MSE with a heavy penalty for errors on non-zero targets."""
    # Calculate squared difference
    sq_diff = tf.square(y_true - y_pred)

    # Create weights:
    # If y_true is effectively non-zero (indicating a labeled move), weight it heavily.
    # Otherwise (the 4031 "padding" zeroes), weight it normally (1.0).
    # We use a threshold of 1e-7 to detect non-zero targets.
    weights = tf.cast(tf.abs(y_true) > 1e-7, tf.float32) * 500.0 + 1.0

    # Reduce over both the 2 channels and the 4032 moves to get a per-sample scalar
    return tf.reduce_mean(sq_diff * weights, axis=[1, 2])


@tf.keras.utils.register_keras_serializable()
def validity_loss(y_true, y_pred):
    """Dice Loss with explicit false-positive penalty.

    Dice (soft F1) helps with class imbalance, but by itself it can still allow
    too many false positives. Adding a penalty on predicted probability mass on
    negative labels pushes precision up.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate intersection and sums over moves and channels
    # Shape: (batch, 4032, 1) -> (batch,)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    sum_true = tf.reduce_sum(y_true, axis=[1, 2])
    sum_pred = tf.reduce_sum(y_pred, axis=[1, 2])

    epsilon = tf.keras.backend.epsilon()
    dice = (2.0 * intersection + epsilon) / (sum_true + sum_pred + epsilon)

    dice_loss = 1.0 - dice  # (batch,)

    # False-positive penalty: average predicted probability on negatives.
    # This is 0 when all negatives are predicted 0.
    neg_mass = tf.reduce_sum((1.0 - y_true) * y_pred, axis=[1, 2])
    neg_count = tf.reduce_sum(1.0 - y_true, axis=[1, 2])
    fp_penalty = neg_mass / (neg_count + epsilon)  # (batch,)

    # Stronger penalty to force precision to 100%
    fp_weight = 5.0

    return tf.reduce_mean(dice_loss + fp_weight * fp_penalty)


@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    """F1 score metric for binary classification."""
    # Threshold predictions at 0.5
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)

    # Calculate true positives, false positives, false negatives
    tp = tf.reduce_sum(y_true * y_pred_binary)
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    fn = tf.reduce_sum(y_true * (1 - y_pred_binary))

    # Calculate precision and recall
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1


class NeuralNetworkAgent(TrainableAgent):
    """Agent that uses a neural network for move prediction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_file_path = f"models/{self.username}.keras"
        self.model = None
        self.heuristic_evaluator = HeuristicEvaluator()

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.load_model()

    def load_model(self) -> None:
        """Load an existing model from disk or create a new one."""

        model_path = Path(self.model_file_path)

        # Load main model
        if model_path.exists():
            logger.info(
                "Loading model from %s", model_path, extra={"username": self.username}
            )
            self.model = tf.keras.models.load_model(model_path)
        else:
            base_choices = [
                "models/base_s.keras",
                "models/base_m.keras",
                "models/base_l.keras",
            ]
            base_path = Path(random.choice(base_choices))
            logger.info(
                "Loading base model from %s",
                base_path,
                extra={"username": self.username},
            )
            self.model = tf.keras.models.load_model(base_path)

    def create_model(self):
        """Create and return a new TensorFlow model.

        The model takes the 768-dimensional feature vector produced by
        TrainableAgent._encode_fen as input.
        """

        inputs = tf.keras.Input(shape=(768,))

        x = tf.keras.layers.Reshape((8, 8, -1))(inputs)
        state = x
        for _ in range(10):
            x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
            x = tf.keras.layers.Conv2D(10, 3, activation="relu")(x)
            state = tf.keras.layers.Concatenate()([state, x])
            state = tf.keras.layers.BatchNormalization()(state)
            x = state
        x = tf.keras.layers.Conv2D(10, 3)(x)
        x = tf.keras.layers.Flatten()(x)

        # Head 1: Moves (Evaluation & Decisive)
        # Output: (4032, 2)
        logits_moves = tf.keras.layers.Dense(4032 * 2)(x)
        reshaped_moves = tf.keras.layers.Reshape((4032, 2))(logits_moves)

        eval_channel = tf.keras.layers.Activation("tanh")(reshaped_moves[..., 0:1])
        dec_channel = tf.keras.layers.Activation("sigmoid")(reshaped_moves[..., 1:2])
        out_moves = tf.keras.layers.Concatenate(name="eval")(
            [eval_channel, dec_channel]
        )

        # Head 2: Validity
        # Output: (4032, 1)
        logits_valid = tf.keras.layers.Dense(4032)(x)
        out_valid = tf.keras.layers.Activation("sigmoid")(logits_valid)
        out_valid = tf.keras.layers.Reshape((4032, 1), name="valid")(out_valid)

        model = tf.keras.Model(inputs=inputs, outputs=[out_moves, out_valid])

        model.compile(
            optimizer="adam",
            loss=[weighted_mse, validity_loss],
            metrics={
                "eval": tf.keras.metrics.MeanAbsoluteError(name="mae"),
                "valid": [
                    f1_score,
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                ],
            },
        )

        return model

    def _move_to_id(self, from_idx: int, to_idx: int) -> int:
        if to_idx < from_idx:
            local = to_idx
        else:
            local = to_idx - 1
        return from_idx * 63 + local

    def _id_to_move(self, move_id: int) -> tuple[int, int]:
        from_idx = move_id // 63
        local = move_id % 63
        if local < from_idx:
            to_idx = local
        else:
            to_idx = local + 1
        return from_idx, to_idx

    # Files and ranks for square name conversion
    FILES = "abcdefgh"
    RANKS = "12345678"

    def _index_to_square(self, index: int) -> str:
        """Convert square index (0-63) to square name (a1-h8)."""
        file_idx = index % 8
        rank_idx = index // 8
        return f"{self.FILES[file_idx]}{self.RANKS[rank_idx]}"

    def _square_to_index(self, square: str) -> int:
        """Convert square name (a1-h8) to square index (0-63)."""
        file = square[0]
        rank = square[1]
        file_idx = self.FILES.index(file)
        rank_idx = self.RANKS.index(rank)
        return rank_idx * 8 + file_idx

    def _predict(self, fen: str, our_squares: list[str]) -> list[PredictionResult]:
        if self.model is None:
            raise RuntimeError(
                "Model is not initialized. Load or create the model first."
            )

        assert our_squares, "our_squares cannot be empty"

        # Encode fen and run the model to get move scores
        features = self._encode_fen(fen)
        inputs = np.asarray(features, dtype=np.float32).reshape(1, -1)
        # Model returns [moves_pred, valid_pred]
        predictions_raw = self.model.predict(inputs, verbose=0)
        moves_pred = predictions_raw[0][0]
        valid_pred = predictions_raw[1][0]

        # Create chess board for heuristic evaluation
        board = chess.Board(fen)
        is_white = board.turn

        # Iterate over actual legal moves from the board
        candidates: list[tuple[float, PredictionResult]] = []

        for move in board.legal_moves:
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)

            # Skip moves we've already attempted in this exact position
            if (from_sq, to_sq) in self._made_decisions:
                continue

            # Convert move to move_id to look up neural network predictions
            from_idx = self._square_to_index(from_sq)
            to_idx = self._square_to_index(to_sq)
            move_id = self._move_to_id(from_idx, to_idx)

            # Get neural network predictions for this move
            nn_eval_val = float(moves_pred[move_id][0])
            nn_dec_val = float(moves_pred[move_id][1])
            validity_score = float(valid_pred[move_id][0])

            # Get heuristic evaluation for this move
            board_after = board.copy()
            board_after.push(move)
            heuristic_eval, heuristic_dec = self.heuristic_evaluator.evaluate_position(
                board_after, is_white
            )

            logger.info(
                "Move %s->%s: NN eval=%.3f, NN dec=%.3f, Validity=%.3f, Heuristic eval=%.3f, Heuristic dec=%.3f",
                from_sq,
                to_sq,
                nn_eval_val,
                nn_dec_val,
                validity_score,
                heuristic_eval,
                heuristic_dec,
                extra={"username": self.username},
            )

            # Average the neural network and heuristic evaluations
            avg_eval = 0.05 * nn_eval_val + 0.95 * heuristic_eval
            avg_dec = 0.05 * nn_dec_val + 0.95 * heuristic_dec

            candidates.append(
                (
                    validity_score,
                    PredictionResult(
                        from_sq=from_sq,
                        to_sq=to_sq,
                        evaluation=avg_eval,
                        decisive=avg_dec,
                    ),
                )
            )

        # Sort by evaluation descending (best moves first)
        candidates.sort(key=lambda x: x[1].evaluation, reverse=True)

        # Return top N predictions
        return [c[1] for c in candidates[: self.prediction_count]]

    def _encode_fen(self, fen: str) -> np.ndarray:
        """Encode FEN string as numerical features for the neural network.

        Returns a flattened 8x8x12 array where channels 0-5 are our pieces and
        6-11 are opponent pieces. The active color in the FEN determines which
        side is considered "our" pieces.
        """
        parts = fen.split()
        if len(parts) < 2:
            raise ValueError(
                f"Invalid FEN string provided (missing active color): {fen}"
            )

        board_part = parts[0]
        board = np.zeros((8, 8, 12), dtype=np.float32)

        active_color = parts[1]
        our_pieces_are_uppercase = active_color == "w"

        rows = board_part.split("/")
        for rank_idx, row in enumerate(rows):
            file_idx = 0
            for char in row:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    piece_type = char.upper()
                    if piece_type in PIECE_TYPE_TO_INDEX:
                        piece_type_idx = PIECE_TYPE_TO_INDEX[piece_type]

                        is_uppercase = char.isupper()
                        is_our_piece = is_uppercase == our_pieces_are_uppercase

                        if is_our_piece:
                            piece_idx = piece_type_idx
                        else:
                            piece_idx = piece_type_idx + 6

                        board[7 - rank_idx, file_idx, piece_idx] = 1.0
                    file_idx += 1

        return board.flatten()

    def _on_game_end(self, score: int | None, reason: str | None) -> None:
        """Called when the game ends. Trains the model based on the game result."""
        assert score is not None, "Game score is missing"

        if self._moves_made < 2 or (
            reason in ["timeout", "agreement"] and self._moves_made < 10
        ):
            logger.info(
                "Skipping training on game end due to insufficient moves made (%d) for reason=%s",
                self._moves_made,
                reason,
                extra={"username": self.username},
            )
            return

        assert self._decision_history, "Decision history is empty but moves were made"

        assert self.model is not None, "Model is not initialized"

        try:
            # Prepare batch from history
            inputs = []
            move_indices = []

            for decision in self._decision_history:
                inputs.append(self._encode_fen(decision.fen))
                move = decision.move  # PredictionResult

                # Calculate move_id by converting square names to indices
                from_idx = self._square_to_index(move.from_sq)
                to_idx = self._square_to_index(move.to_sq)

                move_id = self._move_to_id(from_idx, to_idx)
                move_indices.append(move_id)

            X = np.array(inputs)

            # Predict current outputs
            preds = self.model.predict(X, verbose=0)
            target_moves = np.zeros_like(preds[0])
            target_valid = preds[1]

            # Targets for this game
            eval_target = float(score)
            decisive_target = float(abs(score))

            for i, move_id in enumerate(move_indices):
                # Update Strategy Head for the chosen move
                target_moves[i, move_id, 0] = eval_target
                target_moves[i, move_id, 1] = decisive_target

                # Update Validity Head (reinforce that this move was valid)
                target_valid[i, move_id, 0] = 1.0

            # Train on the sequence of moves from this game
            for _ in range(10):
                loss = self.model.train_on_batch(X, [target_moves, target_valid])

            # Save the model
            model_path = Path(self.model_file_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path)

            logger.info(
                "Trained on game end (score=%s) and saved model to %s. Samples: %d. Loss: %s",
                score,
                model_path,
                len(X),
                loss,
                extra={"username": self.username},
            )

        except Exception as e:
            logger.error(
                "Failed to train on game end: %s",
                e,
                exc_info=True,
                extra={"username": self.username},
            )

    def _handle_move_failure(self, fen: str, from_square: str, to_square: str) -> None:
        """Handle a move execution failure by training the model that this move is invalid."""
        assert self.model is not None, "Model is not initialized"

        assert fen and from_square and to_square, (
            "Invalid parameters for move failure handling"
        )

        try:
            # 1. Encode context
            features = self._encode_fen(fen)

            # 2. Determine move ID
            from_idx = self._square_to_index(from_square)
            to_idx = self._square_to_index(to_square)

            if from_idx == to_idx:
                return

            move_id = self._move_to_id(from_idx, to_idx)

            # 3. Get current predictions to use as baseline
            # We want to keep other predictions stable, only penalizing the invalid move
            inp = features.reshape(1, 768)
            preds = self.model.predict(inp, verbose=0)

            # Use current predictions as targets for moves to avoid changing them
            target_moves = preds[0]

            # Use current validity predictions but penalize the specific invalid move
            target_valid = np.copy(preds[1])
            target_valid[0, move_id, 0] = 0.0

            # 4. Train on this single sample
            for _ in range(10):
                loss = self.model.train_on_batch(x=inp, y=[target_moves, target_valid])

            logger.info(
                "Trained validity (invalid) for move %s->%s. Loss: %s",
                from_square,
                to_square,
                loss,
                extra={"username": self.username},
            )
        except Exception as e:
            logger.error(
                "Failed to train on move failure: %s",
                e,
                exc_info=True,
                extra={"username": self.username},
            )
