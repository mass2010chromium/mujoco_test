#!/bin/env bash

PREFIX=$1
module load nasm/2.16.03
./configure --prefix="$PREFIX" --enable-pic --enable-shared
make
