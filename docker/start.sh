#!/bin/bash

cd /app/ChatTTS
nohup yarn preview > /dev/null 2>&1 &

# start api
python api/server.py