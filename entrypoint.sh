#!/bin/bash

echo "Running setup tasks"

chmod +x ./prod_setup.sh
./prod_setup.sh

echo "Starting uvicorn"
uvicorn app:app --host 0.0.0.0 --port 8080
