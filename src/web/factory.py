from src.web.client import ChessWebClient
from src.web.config import BrowserConfig
from src.web.selectors import site_selectors


def build_web_client(base_url: str) -> ChessWebClient:
    config = BrowserConfig(base_url=base_url)
    selectors = site_selectors()
    client = ChessWebClient(config, selectors)
    client.start()
    return client
