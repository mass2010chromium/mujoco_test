#!/bin/env bash

PREFIX=/home/hice1/jpeng303/.local/
SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

mkdir -p installs
cd installs
git clone https://code.videolan.org/videolan/x264.git
cd x264
srun -c 12 bash "$SCRIPT_DIR/srun_x264.sh" "$PREFIX"
make install

cd ..
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
srun -c 12 bash "$SCRIPT_DIR/srun_ffmpeg.sh" "$PREFIX"
make install
