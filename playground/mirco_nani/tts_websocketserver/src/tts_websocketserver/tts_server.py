from tts_websocketserver.tts_pipeline import get_pipeline
from rgws.interface import WebsocketServer
import json, asyncio


class TTSPipelineManager:
    def __init__(self):
        self.pipeline = get_pipeline()
        self.state = False
        #self.setup_model()

    def setup_model(self):
        if not self.state:
            self.pipeline.build()
            self.state = True
        return self.state
    
    async def process_text(self, text):
        res  = self.pipeline.predict(text)
        resp = json.dumps(res)
        yield resp

    async def status(self):
        return self.state


    def __del__(self):
        self.pipeline.dispose()

        
# building in global so it can also be imported from outside
# eg. from tts_websocketserver.tts_server import tts_pipeline
tts_pipeline = TTSPipelineManager()


class TTSServerInterface(WebsocketServer):
    def __init__(self, **kwargs):
        super(TTSServerInterface, self).__init__(**kwargs)
        self._register(tts_pipeline.process_text)
        self._register(tts_pipeline.setup_model)
        self._register(tts_pipeline.status)

    async def _consumer(self, ws, message):
        ret = await self.dispatch(message)
        async for msg in ret:
            await ws.send_json(msg)


if __name__ == "__main__":
    s = TTSServerInterface(host="localhost", port=8080)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(s.run())
    loop.run_forever()