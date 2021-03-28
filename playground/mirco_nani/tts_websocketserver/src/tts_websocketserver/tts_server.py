from tts_websocketserver.tts_pipeline import get_pipeline
from rgws.interface import WebsocketServer
import json, asyncio


class TTSPipelineManager:
    def __init__(self):
        self.pipeline = get_pipeline()
        self.pipeline.build()
    
    async def process_text(self, req):
        text = req["text"]
        res  = self.pipeline.predict(text)
        resp = json.dumps(res)
        yield resp

    def __del__(self):
        self.tts_pipeline.dispose()

        
# building in global so it can also be imported from outside
# eg. from tts_websocketserver.tts_server import tts_pipeline
tts_pipeline = TTSPipelineManager()


class TTSServerInterface(WebsocketServer):
    def __init__(self, **kwargs):
        super(TTSServerInterface, self).__init__(**kwargs)
        self._register(tts_pipeline.process_text)

    async def _consumer(self, websocket, message):
        ret = await self.dispatch(message)
        async for gen in ret:
            await websocket.send(gen)


if __name__ == "__main__":
    s = TTSServerInterface(host="localhost", port=8080)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(s.run(compression=None))
    loop.run_forever()