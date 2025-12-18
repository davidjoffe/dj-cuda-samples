@rem dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
@rem (c) David Joffe / DJ Software - Business Source License (BSL 1.1). See LICENSE

@rem basic benchmark helpers
@rem todo - flesh these out, make more generic, make possible to do specific apps, auto-log benchmarks etc.
@rem They should test/benchmarch both windowed and headless (except Docker/containered headless only)
@rem Should be possible to do either eg 'bench-all' or 'bench molecular_sim' (or bench-molecular_sim) etc.

".\build-windows\samples\molecular_sim\Release\djmolecular_sim.exe"  -N 1024 --headless --maxframes 1000
".\build-windows\samples\molecular_sim\Release\djmolecular_sim.exe"  -N 10000 --headless --maxframes 1000
".\build-windows\samples\molecular_sim\Release\djmolecular_sim.exe"  -N 20000 --headless --maxframes 1000
".\build-windows\samples\molecular_sim\Release\djmolecular_sim.exe"  -N 50000 --headless --maxframes 1000
".\build-windows\samples\molecular_sim\Release\djmolecular_sim.exe"  -N 100000 --headless --maxframes 1000

".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 1000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 1000000 --headless --maxframes 1000
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 1500000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 1500000 --headless --maxframes 1000
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 2000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 2000000 --headless --maxframes 1000
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 10000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 10000000 --headless --maxframes 1000
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 50000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 50000000 --headless --maxframes 1000

".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 3000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 3500000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 4000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 5000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 6000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 8000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 9000000 --headless --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 10000000 --headless --maxframes 500

".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 1000000 --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 1000000 --maxframes 200
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 1500000 --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 1500000 --maxframes 200
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 2000000 --maxframes 500
".\build-windows\samples\bouncing_balls\Release\djbouncing_balls.exe"  -N 2000000 --maxframes 200
