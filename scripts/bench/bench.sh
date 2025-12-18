#!/bin/sh
# dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
# (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE

# basic benchmark helpers
# todo - flesh these out, make more generic, make possible to do specific apps, auto log benchmarks etc.
# They should test/benchmarch both windowed and headless (except Docker/containered headless only)
# Should be possible to do either eg 'bench-all' or 'bench molecular_sim' (or bench-molecular_sim) etc.

./build-linux/samples/molecular_sim/djmolecular_sim  -N 1024 --headless --maxframes 1000
./build-linux/samples/molecular_sim/djmolecular_sim  -N 10000 --headless --maxframes 1000
./build-linux/samples/molecular_sim/djmolecular_sim  -N 20000 --headless --maxframes 1000
./build-linux/samples/molecular_sim/djmolecular_sim  -N 50000 --headless --maxframes 1000
./build-linux/samples/molecular_sim/djmolecular_sim  -N 100000 --headless --maxframes 1000

./build-linux/samples/bouncing_balls/djbouncing_balls  -N 1000000 --headless --maxframes 500
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 1000000 --headless --maxframes 1000
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 1500000 --headless --maxframes 500
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 1500000 --headless --maxframes 1000
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 2000000 --headless --maxframes 500
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 2000000 --headless --maxframes 1000
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 10000000 --headless --maxframes 500
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 10000000 --headless --maxframes 1000
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 50000000 --headless --maxframes 500
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 50000000 --headless --maxframes 1000

./build-linux/samples/bouncing_balls/djbouncing_balls  -N 1000000 --maxframes 500
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 1000000 --maxframes 200
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 1500000 --maxframes 500
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 1500000 --maxframes 200
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 2000000 --maxframes 500
./build-linux/samples/bouncing_balls/djbouncing_balls  -N 2000000 --maxframes 200
