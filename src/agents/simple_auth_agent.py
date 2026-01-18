from __future__ import annotations

from src.agents.base import Agent


class SimpleAuthAgent(Agent):
    def run(self) -> None:
        self.ensure_registered()
        self.sign_in()
        self.sign_out()
        self.session_done = True
