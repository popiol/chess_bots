from __future__ import annotations

import logging
import random

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from src.agents.base import Agent

logger = logging.getLogger(__name__)


class GuestAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._time_control_weights: dict[int, int] = {}
        self._stage = "init"
        self._matchmaking_log_step = 0
        self._decision: str | None = None
        self._draw_wait_ticks = 0

    def run(self) -> None:
        try:
            if self._stage == "init":
                index = self._pick_time_control()
                logger.info(
                    "Selecting time control index=%s",
                    index,
                    extra={"username": self.username},
                )
                self._web_client.select_time_control(index)
                self._stage = "selected"
                return

            if self._stage == "selected":
                logger.info("Play as guest", extra={"username": self.username})
                self._web_client.play_as_guest()
                self._stage = "matchmaking"
                return

            if self._stage == "matchmaking":
                if not self._web_client.is_play_ready():
                    if self._matchmaking_log_step % 60 == 0:
                        logger.info(
                            "Matchmaking pending", extra={"username": self.username}
                        )
                    self._matchmaking_log_step += 1
                    return
                logger.info("Matchmaking complete", extra={"username": self.username})
                self._stage = "playing"
                self._matchmaking_log_step = 0
                return

            if self._stage == "playing":
                if self._web_client.is_postgame_visible():
                    logger.info(
                        "Postgame visible, ending session",
                        extra={"username": self.username},
                    )
                    self._stage = "done"
                    return

                if self._decision is None:
                    if self._web_client.is_accept_draw_visible():
                        self._decision = (
                            "accept_draw" if random.random() < 0.5 else "resign"
                        )
                    else:
                        self._decision = (
                            "offer_draw" if random.random() < 0.5 else "resign"
                        )
                    return

                if self._decision == "accept_draw":
                    logger.info("Accepting draw", extra={"username": self.username})
                    self._web_client.accept_draw()
                    return

                if self._decision == "offer_draw":
                    logger.info("Offering draw", extra={"username": self.username})
                    self._web_client.offer_draw()
                    self._decision = "wait_draw"
                    return

                if self._decision == "wait_draw":
                    self._draw_wait_ticks += 1
                    if self._draw_wait_ticks >= 10:
                        logger.info(
                            "Draw timeout, resigning", extra={"username": self.username}
                        )
                        self._decision = "resign"
                    return

                if self._decision == "resign":
                    logger.info("Resigning", extra={"username": self.username})
                    self._web_client.resign()
                    self._decision = "resign_confirm"
                    return

                if self._decision == "resign_confirm":
                    logger.info("Confirming resign", extra={"username": self.username})
                    self._web_client.resign_confirm()
                    return

            if self._stage == "done":
                self.session_done = True
        except PlaywrightTimeoutError:
            logger.exception(
                "Playwright timeout stage=%s decision=%s",
                self._stage,
                self._decision,
                extra={"username": self.username},
            )

    def snapshot_state(self) -> dict:
        state = super().snapshot_state()
        state["time_control_weights"] = self._time_control_weights
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._time_control_weights = state.get("time_control_weights", {})
        if not self._time_control_weights:
            indices = self._web_client.time_control_indices()
            self._time_control_weights = {
                index: random.randint(0, 5) for index in indices
            }
            if indices and sum(self._time_control_weights.values()) == 0:
                self._time_control_weights[random.choice(indices)] = 1
        self._stage = "init"
        self._decision = None
        self._draw_wait_ticks = 0

    def _pick_time_control(self) -> int:
        indices = list(self._time_control_weights.keys())
        values = list(self._time_control_weights.values())
        if sum(values) == 0:
            return random.choice(indices)
        return random.choices(indices, weights=values, k=1)[0]
