#!/bin/bash

current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${current_dir}/tts_websocketserver/src"
python -m tts_websocketserver.simple_client