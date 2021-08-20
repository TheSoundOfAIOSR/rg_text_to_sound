import os, sys
sys.path.append( os.path.join(os.path.dirname(os.path.abspath(__file__)),'tts_websocketserver','src') )
from tts_websocketserver.tts_server import run

if __name__ == '__main__':
    run()