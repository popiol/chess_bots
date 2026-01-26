import numpy as np

from src.agents.trainable_agent import TrainableAgent


def make_agent() -> TrainableAgent:
    # Minimal agent instance; web_client not needed for these tests
    return TrainableAgent(
        username="test", password="x", email="x@x", classpath="", web_client=None
    )


def test_encode_and_extract_basic():
    agent = make_agent()
    fen = "8/8/8/8/8/8/4P3/8 w - - 0 1"

    features = agent._encode_fen(fen)
    assert isinstance(features, np.ndarray)
    assert features.shape == (768,)

    squares = agent._extract_our_squares_from_features(features)
    assert len(squares) >= 1
    assert "e2" in squares

    board = features.reshape(8, 8, 12)
    for s in squares:
        idx = agent._square_to_index(s)
        r = idx // 8
        f = idx % 8
        assert board[r, f, 0:6].any()


def test_encode_empty_position():
    agent = make_agent()
    fen = "8/8/8/8/8/8/8/8 w - - 0 1"

    features = agent._encode_fen(fen)
    assert features.sum() == 0.0

    squares = agent._extract_our_squares_from_features(features)
    assert squares == []


def test_encode_and_extract_black_active():
    agent = make_agent()
    # Black pawn on e7 (black to move -> our pieces are lowercase)
    fen = "8/4p3/8/8/8/8/8/8 b - - 0 1"

    features = agent._encode_fen(fen)
    assert isinstance(features, np.ndarray)
    assert features.shape == (768,)

    squares = agent._extract_our_squares_from_features(features)
    # Should find at least the pawn square
    assert len(squares) >= 1
    assert "e7" in squares

    board = features.reshape(8, 8, 12)
    for s in squares:
        idx = agent._square_to_index(s)
        r = idx // 8
        f = idx % 8
        assert board[r, f, 0:6].any()
