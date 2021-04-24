from tts_websocketserver.tts_pipeline import get_pipeline
from rgws.interface import WebsocketServer
import json, asyncio


class TTSPipelineManager:
    def __init__(self):
        self.state = "Setup"
        self.pipeline = get_pipeline()
        self.pipeline.build()
        self.state = "Ready"

    async def process_text(self, text):
        self.state = "Processing"
        yield self.pipeline.predict(text)
        self.state = "Processed"

    async def status(self):
        return self.state


    def __del__(self):
        self.state = "Disposing"
        self.pipeline.dispose()
        self.state = "Disposed"


# building in global so it can also be imported from outside
# eg. from tts_websocketserver.tts_server import tts_pipeline
tts_pipeline = TTSPipelineManager()


class TTSServerInterface(WebsocketServer):
    def __init__(self, **kwargs):
        super(TTSServerInterface, self).__init__(**kwargs)
        self._register(tts_pipeline.process_text)
        self._register(self.status)
        self._register(self.setup_model)

    async def _consumer(self, ws, message):
        ret = await self.dispatch(message)
        async for gen in ret:
            await ws.send_json(gen)

    async def status(self):
        yield {"resp": tts_pipeline.state}

    async def setup_model(self):
        yield {"resp": True if tts_pipeline.state != "Setup" else False}


if __name__ == "__main__":
    s = TTSServerInterface(host="localhost", port=8787)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(s.run())
    loop.run_forever()