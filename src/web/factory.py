from playwright.sync_api import sync_playwright

from src.web.client import ChessWebClient
from src.web.config import BrowserConfig
from src.web.selectors import SiteSelectors, site_selectors


def build_web_client(base_url: str) -> ChessWebClient:
    config = BrowserConfig(base_url=base_url)
    selectors = site_selectors()
    client = ChessWebClient(config, selectors)
    client.start()
    return client


class WebClientFactory:
    def __init__(self, config: BrowserConfig, selectors: SiteSelectors) -> None:
        self._config = config
        self._selectors = selectors
        self._playwright = sync_playwright().start()
        # Use conservative launch args to reduce per-process memory usage.
        launch_args = [
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-background-timer-throttling",
            "--disable-breakpad",
            "--no-first-run",
            "--safebrowsing-disable-download-protection",
        ]

        self._browser = self._playwright.firefox.launch(
            headless=config.headless,
            slow_mo=config.slow_mo_ms,
            args=launch_args,
        )

    def create_client(self) -> ChessWebClient:
        client = ChessWebClient(
            self._config,
            self._selectors,
            playwright=self._playwright,
            browser=self._browser,
        )
        client.start()
        return client

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()
