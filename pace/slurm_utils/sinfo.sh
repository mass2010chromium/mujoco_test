#!/bin/env bash
SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

sinfo -eO "CPUs:8,Memory:9,Gres:40,NodeAIOT:16,Partition:14,NodeList:400" | tee "$SCRIPT_DIR/sinfo.log"
