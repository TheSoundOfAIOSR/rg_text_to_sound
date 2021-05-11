# TTS WebsocketServer
This is a websocket server based on [rgws](https://github.com/Redict/rg_websocket) for integration of our [inference pipeline](../tts_pipeline) with the production team  
The server exposes an RPC method called **process_text** with the following I/O:  
**Input**:
``` 
{"text": "<A SENTENCE>"}
```   
**Output**:
```
{
    'velocity': <int>,
    'pitch': <int>,
    'source': <str>,
    'qualities': [<str>, <str>, ...],
    'latent_sample': [<float>, <float>, <float>, ...]
}
```

## Setup
``` 
pip install - r requirements.txt
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
```   
if this fails for any reason, run the following commands:  
``` 
pip install rgws
pip install git+https://git@github.com/TheSoundOfAIOSR/rg_text_to_sound.git#"subdirectory=tts_pipeline"
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
``` 

## Test run
**Server**:
```
cd src
python -m tts_websocketserver.tts_server
```  

**Client**:
```
cd src
python -m tts_websocketserver.simple_client
```  