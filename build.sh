#!/bin/bash
rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" ..
make
cd ..
chmod +x ./build/convolution
./build/convolution