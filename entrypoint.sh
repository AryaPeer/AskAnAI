#!/bin/sh
ollama serve > /dev/null 2>&1 &
sleep 10
ollama pull llama3.1 > /dev/null 2>&1
python main.py