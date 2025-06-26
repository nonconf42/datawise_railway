#!/bin/bash

# Find the first process using port 8000
PID=$(lsof -i :3000 -t | head -n 1)

# Check if a process was found
if [ -n "$PID" ]; then
    echo "Killing process with PID: $PID"
    kill -9 $PID
else
    echo "No process found on port 3000"
fi
