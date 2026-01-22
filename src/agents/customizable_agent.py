from __future__ import annotations

import logging
import random
import time
from abc import ABC

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from src.agents.base import Agent

logger = logging.getLogger(__name__)

GUEST_PROBABILITY = 0.5


class CustomizableAgent(Agent, ABC):
    def __init__(
        self,
        username: str,
        password: str,
        email: str,
        classpath: str,
        web_client,
    ) -> None:
        super().__init__(username, password, email, classpath, web_client)
        self._guest = False
        self._stage = "auth"
        self._decision: str | None = None
        self._last_matchmaking_log_time = 0.0
        self._draw_wait_start_time = 0.0
        self._last_stage_change_time = 0.0
        self._time_control_weights: dict[int, int] = {}
        self._consecutive_failures = 0

    def run(self) -> None:
        try:
            if self._stage == "auth":
                self._step_auth()
                return
            if self._stage == "select_time_control":
                self._step_select_time_control()
                return
            if self._stage == "queue":
                self._step_queue()
                return
            if self._stage == "matchmaking":
                self._step_matchmaking()
                return
            if self._stage == "playing":
                self._step_playing()
                return
            if self._stage == "done":
                self._step_done()
                return
            self._consecutive_failures = 0
        except PlaywrightTimeoutError:
            self._consecutive_failures += 1
            logger.exception(
                "Playwright timeout stage=%s decision=%s failures=%d",
                self._stage,
                self._decision,
                self._consecutive_failures,
                extra={"username": self.username},
            )
            if self._consecutive_failures >= 10:
                logger.error(
                    "Too many consecutive failures, ending session",
                    extra={"username": self.username},
                )
                self.session_done = True

    def snapshot_state(self) -> dict:
        state = super().snapshot_state()
        state["guest"] = self._guest
        state["time_control_weights"] = self._time_control_weights
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        if "guest" in state:
            self._guest = bool(state.get("guest"))
        else:
            self._guest = random.random() < GUEST_PROBABILITY
        self._time_control_weights = state.get("time_control_weights", {})
        if not self._time_control_weights:
            self._time_control_weights = self._random_time_control_weights()

    def _step_auth(self) -> None:
        current_time = time.time()
        if current_time - self._last_stage_change_time < 1.0:
            return
        if not self._guest:
            try:
                self.ensure_registered()
            except PlaywrightTimeoutError:
                logger.info(
                    "Registration timeout, assuming already registered",
                    extra={"username": self.username},
                )
                self._registered = True
            self.sign_in()
        self._stage = "select_time_control"
        self._last_stage_change_time = current_time

    def _step_select_time_control(self) -> None:
        current_time = time.time()
        if current_time - self._last_stage_change_time < 1.0:
            return
        index = self._pick_time_control()
        logger.info(
            "Selecting time control index=%s",
            index,
            extra={"username": self.username},
        )
        self._web_client.select_time_control(index)
        self._stage = "queue"
        self._last_stage_change_time = current_time

    def _step_queue(self) -> None:
        current_time = time.time()
        if current_time - self._last_stage_change_time < 1.0:
            return
        if self._guest:
            logger.info("Play as guest", extra={"username": self.username})
            self._web_client.play_as_guest()
        else:
            logger.info("Play now", extra={"username": self.username})
            self._web_client.queue_play_now()
        self._stage = "matchmaking"
        self._last_stage_change_time = current_time

    def _step_matchmaking(self) -> None:
        current_time = time.time()
        if current_time - self._last_stage_change_time < 1.0:
            return
        
        if not self._web_client.is_play_ready():
            # Log every 60 seconds
            if current_time - self._last_matchmaking_log_time >= 60.0:
                logger.info("Matchmaking pending", extra={"username": self.username})
                self._last_matchmaking_log_time = current_time
            return
        
        logger.info("Matchmaking complete", extra={"username": self.username})
        self._stage = "playing"
        self._last_stage_change_time = current_time

    def _step_playing(self) -> None:
        current_time = time.time()
        if current_time - self._last_stage_change_time < 1.0:
            return

        if self._web_client.is_postgame_visible():
            logger.info(
                "Postgame visible, ending session",
                extra={"username": self.username},
            )
            self._stage = "done"
            self._last_stage_change_time = current_time
            return

        if self._decision is None:
            if self._web_client.is_accept_draw_visible():
                self._decision = "accept_draw" if random.random() < 0.5 else "resign"
            else:
                self._decision = "offer_draw" if random.random() < 0.5 else "resign"
            return

        if self._decision == "accept_draw":
            logger.info("Accepting draw", extra={"username": self.username})
            self._web_client.accept_draw()
            self._stage = "done"
            self._last_stage_change_time = current_time
            return

        if self._decision == "offer_draw":
            logger.info("Offering draw", extra={"username": self.username})
            self._web_client.offer_draw()
            self._decision = "wait_draw"
            self._draw_wait_start_time = current_time
            self._last_stage_change_time = current_time
            return

        if self._decision == "wait_draw":
            if current_time - self._draw_wait_start_time >= 10.0:
                logger.info(
                    "Draw timeout, resigning", extra={"username": self.username}
                )
                self._decision = "resign"
            return

        if self._decision == "resign":
            logger.info("Resigning", extra={"username": self.username})
            self._web_client.resign()
            self._decision = "resign_confirm"
            self._last_stage_change_time = current_time
            return

        if self._decision == "resign_confirm":
            logger.info("Confirming resign", extra={"username": self.username})
            self._web_client.resign_confirm()
            self._stage = "done"
            self._last_stage_change_time = current_time
            return

    def _step_done(self) -> None:
        self.session_done = True

    def _pick_time_control(self) -> int:
        indices = list(self._time_control_weights.keys())
        values = list(self._time_control_weights.values())
        if not indices:
            raise RuntimeError("time_control_weights is empty")
        if sum(values) == 0:
            return random.choice(indices)
        return random.choices(indices, weights=values, k=1)[0]

    def _random_time_control_weights(self) -> dict[int, int]:
        indices = self._web_client.time_control_indices()
        weights = {index: random.randint(0, 5) for index in indices}
        if indices and sum(weights.values()) == 0:
            weights[random.choice(indices)] = 1
        return weights
