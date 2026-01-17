from dataclasses import dataclass


@dataclass(frozen=True)
class BrowserConfig:
    base_url: str
    headless: bool = True
    slow_mo_ms: int = 0
    navigation_timeout_ms: int = 30000
    action_timeout_ms: int = 10000
