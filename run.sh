#!/bin/bash
docker compose up -d 
uvicorn SSE:app --port 8989