#!/bin/env bash

PREFIX=$1
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
module load nasm/2.16.03
./configure --enable-shared --enable-libx264 --enable-gpl --prefix="$PREFIX"
make
