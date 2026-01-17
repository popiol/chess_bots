from src.web import BrowserConfig, ChessWebClient, site_selectors


def main() -> None:
    config = BrowserConfig(base_url="https://playbullet.gg")
    selectors = site_selectors()
    client = ChessWebClient(config, selectors)
    client.start()
    try:
        client.sign_in(username="Piotr", password="just4fun")
        # client.play_now()
        client.sign_out()
    finally:
        client.close()


if __name__ == "__main__":
    main()
