from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.agents.trainable_agent import PredictionResult, TrainableAgent

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
    """Binary crossentropy computed over all moves and reduced to a scalar."""
    # Compute binary_crossentropy per element (batch, 4032, 1)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Reduce over the move dimension to get (batch,)
    return tf.reduce_mean(bce, axis=-1)


class NeuralNetworkAgent(TrainableAgent):
    """Agent that uses a neural network for move prediction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_file_path = f"models/{self.username}.keras"
        self.model = None

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
            logger.info(
                "Model file not found at %s, creating new model",
                model_path,
                extra={"username": self.username},
            )
            self.model = self.create_model()

    def create_model(self):
        """Create and return a new TensorFlow model.

        The model takes the 768-dimensional feature vector produced by
        TrainableAgent._encode_fen as input.
        """

        inputs = tf.keras.Input(shape=(768,))

        x = tf.keras.layers.Reshape((8, 8, -1))(inputs)
        state = x
        for _ in range(1):
            x = tf.keras.layers.ZeroPadding2D(padding=1)(state)
            x = tf.keras.layers.Conv2D(1, 3, activation="relu")(x)
            state = tf.keras.layers.Concatenate()([state, x])
        x = tf.keras.layers.Conv2D(1, 3)(x)
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
                "valid": tf.keras.metrics.BinaryAccuracy(name="acc"),
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

    def _predict(
        self, features: np.ndarray, our_squares: list[str]
    ) -> list[PredictionResult]:
        if self.model is None:
            raise RuntimeError(
                "Model is not initialized. Load or create the model first."
            )

        assert our_squares, "our_squares cannot be empty"

        # Run the model to get move scores
        inputs = np.asarray(features, dtype=np.float32).reshape(1, -1)
        # Model returns [moves_pred, valid_pred]
        predictions_raw = self.model.predict(inputs, verbose=0)
        moves_pred = predictions_raw[0][0]
        valid_pred = predictions_raw[1][0]

        # Flatten move indices 0..4031 into (from_idx, to_idx) using
        # a 64 x 63 mapping over all (from, to) pairs with from != to.
        our_square_indices = {self._square_to_index(square) for square in our_squares}
        candidates: list[tuple[float, PredictionResult]] = []

        for move_id, (eval_val, dec_val) in enumerate(moves_pred):
            from_idx, to_idx = self._id_to_move(move_id)

            # Filter strictly by pieces we own
            if from_idx not in our_square_indices:
                continue

            validity_score = float(valid_pred[move_id][0])
            candidates.append(
                (
                    validity_score,
                    PredictionResult(
                        from_idx=from_idx,
                        to_idx=to_idx,
                        evaluation=float(eval_val),
                        decisive=float(dec_val),
                    ),
                )
            )

        # Filter by validity threshold
        valid_candidates = [c for c in candidates if c[0] >= 0.5]

        # Fallback to all candidates if no valid moves found
        final_pool = valid_candidates if valid_candidates else candidates

        # Sort by evaluation descending (best moves first)
        final_pool.sort(key=lambda x: x[1].evaluation, reverse=True)

        # Return top N predictions
        return [c[1] for c in final_pool[: self.prediction_count]]

    def _on_game_end(self, score: int | None, reason: str | None) -> None:
        """Called when the game ends. Trains the model based on the game result."""
        assert score is not None, "Game score is missing"

        if self._moves_made < 2 or (
            reason in ["Timeout", "Agreement"] and self._moves_made < 10
        ):
            return

        assert self._decision_history, "Decision history is empty but moves were made"

        assert self.model is not None, "Model is not initialized"

        try:
            # Prepare batch from history
            inputs = []
            move_indices = []

            for decision in self._decision_history:
                inputs.append(decision.features)
                move = decision.move  # PredictionResult

                # Calculate move_id
                from_idx = move.from_idx
                to_idx = move.to_idx

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
            loss = self.model.train_on_batch(X, [target_moves, target_valid])

            logger.info(
                "Trained on game end (score=%s). Samples: %d. Loss: %s",
                score,
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

        if not fen or not from_square or not to_square:
            return

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
