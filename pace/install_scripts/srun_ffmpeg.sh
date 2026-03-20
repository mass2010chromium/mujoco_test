#!/bin/env bash

PREFIX=$1
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
module load nasm/2.16.03
git checkout n4.4.6 # Old version of ffmpeg, for decord
./configure --enable-shared --enable-libx264 --enable-gpl --enable-rpath --prefix="$PREFIX"
make -j12
