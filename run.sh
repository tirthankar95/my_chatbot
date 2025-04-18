#!/bin/bash
docker compose up -d 
uvicorn sse:app --host 0.0.0.0 --port 8989