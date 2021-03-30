from rgws.interface import WebsocketClient
import json, logging, asyncio


logging.getLogger().setLevel(logging.DEBUG)


class SimpleClientInterface(WebsocketClient):
    def __init__(self, **kwargs):
        super(SimpleClientInterface, self).__init__(**kwargs)

    """
    This is business logic for client, basically in this example
    we just connects to server and trying to call `process_text` once
    then exits.
    """

    async def _producer(self, websocket):
        try:
            logging.debug(await self.setup_model())
        except Exception as e:
            logging.debug(f"setup_model call failed with error: {e}")
        try:
            logging.debug(await self.status())
        except Exception as e:
            logging.debug(f"status call failed with error: {e}")
        try:
            logging.debug(await self.process_text("give me a bright guitar"))
        except Exception as e:
            logging.debug(f"process_text call failed with error: {e}")


if __name__ == "__main__":
    c = SimpleClientInterface(host="localhost", port=8080)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(c.run())
    loop.run_forever()