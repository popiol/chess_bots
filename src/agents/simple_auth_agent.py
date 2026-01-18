from __future__ import annotations

import logging

from src.agents.base import Agent

logger = logging.getLogger(__name__)


class SimpleAuthAgent(Agent):
    def run(self) -> None:
        logger.info("Running SimpleAuthAgent", extra={"username": self.username})
        self.ensure_registered()
        self.sign_in()
        self.sign_out()
        self.session_done = True
