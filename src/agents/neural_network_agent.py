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

    def _predict(
        self, features: np.ndarray, our_squares: list[str]
    ) -> list[PredictionResult]:
        if self.model is None:
            raise RuntimeError(
                "Model is not initialized. Load or create the model first."
            )

        if not our_squares:
            return []

        # Run the model to get move scores
        inputs = np.asarray(features, dtype=np.float32).reshape(1, -1)
        # Model returns [moves_pred, valid_pred]
        predictions_raw = self.model.predict(inputs, verbose=0)
        moves_pred = predictions_raw[0][0]
        valid_pred = predictions_raw[1][0]

        # Flatten move indices 0..4031 into (from_idx, to_idx) using
        # a 64 x 63 mapping over all (from, to) pairs with from != to.
        predictions: list[PredictionResult] = []
        for move_id, (eval_val, dec_val) in enumerate(moves_pred):
            # Check validity using the validity model
            validity_score = float(valid_pred[move_id][0])

            # Filter clearly invalid moves (e.g. < 0.5 probability)
            if validity_score < 0.5:
                continue

            from_idx = move_id // 63
            local = move_id % 63
            if local < from_idx:
                to_idx = local
            else:
                to_idx = local + 1
            predictions.append(
                PredictionResult(
                    from_idx=from_idx,
                    to_idx=to_idx,
                    evaluation=float(eval_val),
                    decisive=float(dec_val),
                )
            )

        # Filter out moves that don't start from one of our squares
        our_square_indices = {self._square_to_index(square) for square in our_squares}
        filtered = [p for p in predictions if p.from_idx in our_square_indices]

        # If filtering removed everything, fall back to all predictions
        return filtered or predictions

    def on_game_end(self, result: str, reason: str) -> None:
        pass
