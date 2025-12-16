# [Molecular Simulation (CUDA)](https://github.com/davidjoffe/dj-cuda-samples)

An early-stage **GPU-accelerated molecular dynamics sandbox**, implemented in CUDA and C++, with real-time OpenGL visualization.

This sample focuses on **engineering fundamentals** of molecular simulation:
- Force computation on the GPU
- Stable numerical integration
- Scalable data layouts
- Interactive visualization
- Clean separation of simulation and rendering

It is intentionally **not** a full scientific MD package, but a foundation that mirrors real MD engine structure and behavior.

---

## Features

- **Lennard-Jones (LJ) pairwise interactions**
- **Velocity Verlet integration**
- **CUDA-accelerated force computation**
- **100,000+ particles in real time**
- **OpenGL point-based visualization**
- Configurable parameters (spacing, timestep, point size, zoom)
- Optional **Docker** container support with headless mode

Emergent behaviors such as:
- Local clustering
- Evaporation / dispersion
- Long-term diffusion
are observable depending on initialization and parameters.

---

## What This Is (and Isn’t)

### ✅ This *is*:
- An engineering-led MD sandbox
- A performance-aware CUDA implementation
- A platform for experimentation and visualization
- A stepping stone toward research-adjacent tooling

### ❌ This is *not*:
- A validated scientific MD engine
- A replacement for LAMMPS / GROMACS / OpenMM
- A chemistry-accurate water or biomolecular model

---

## Implementation Overview

### Simulation
- Particle state stored in **Structure-of-Arrays-friendly layouts**
- Pairwise LJ force evaluation on GPU
- Velocity Verlet integration on GPU
- Fixed timestep (configurable)

### Visualization
- OpenGL point rendering
- Color mapping based on spatial coordinates
- Adjustable point size and camera zoom
- Visualization intentionally decoupled from simulation logic

---

## Controls / Parameters

Current parameters are compile-time or startup configurable, including:
- Particle count
- Initial lattice spacing
- Timestep (`dt`)
- Point size
- Camera zoom

Future work includes runtime controls and interactive parameter tuning.

---

## Performance Notes

- O(N²) force computation (no neighbor lists yet)
- Intended as a correctness and structure baseline
- Neighbor lists / spatial binning planned
- Designed to transition toward headless and batch execution

---

## Planned Extensions (Roadmap)

- Periodic Boundary Conditions (PBC)
- Neighbor lists / cell lists
- Deterministic replay
- Snapshot / restart support
- Headless execution mode
- Multi-GPU / multi-backend experimentation
- Alternative force fields (bonded terms, Coulomb, etc.)

---

## Build & Run

This sample is built as part of the main `dj-cuda-samples` CMake project.

See the top-level README for build instructions.

---

## Status

**Early but functional.**  
The focus is correctness, clarity, and extensibility — not scientific claims.

---

Conceptual design layers: 'molecular-specific' -> 'visuals' -> more generic 'renderer'.

---

## Author

David Joffe  
https://djoffe.com
