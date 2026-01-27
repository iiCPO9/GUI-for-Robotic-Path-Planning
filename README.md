# GUI for Robotic Path Planning

Visual interface for exploring robotic path planning and swarm shepherding in 2D environments. Upload a map, set start/end points, optionally add sheep, spawn dynamic obstacles, and visualize algorithms in static and dynamic settings with live replanning.

## Table of Contents
- [Overview](#overview)
- [Highlights / Changes](#highlights--changes)
- [Features](#features)
  - [Path Planning Algorithms](#path-planning-algorithms)
  - [Environments & Dynamics](#environments--dynamics)
  - [Swarm Control & Shepherding](#swarm-control--shepherding)
  - [Map Handling](#map-handling)
  - [Visualization & Analysis](#visualization--analysis)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Notes & Troubleshooting](#notes--troubleshooting)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [License](#license)

## Overview
This project provides an interactive GUI for experimenting with path-planning algorithms, dynamic obstacle handling, and swarm shepherding on 2D grid maps derived from image files. It supports both static comparisons and dynamic, real-time replanning when the environment changes.

## Highlights / Changes
- Static and dynamic 2D environments with live replanning.
- Dynamic obstacle insertion and falling-wall simulations (Dynamic2D).
- Connectivity-safe random obstacle spawning that preserves a valid start→end path.
- Shepherding-based swarm control with algorithm-aware sheep ordering.
- Global interrupt mechanism to safely stop a running algorithm before starting another.
- Plotting helper (`plot_results.py`) for algorithm performance visualization (requires `matplotlib`).

## Features

### Path Planning Algorithms
- Dijkstra’s Algorithm
- A* Search Algorithm
- Depth-First Search (DFS)
- Genetic Algorithm

Each algorithm visualizes explored nodes and the final computed path in real time.

### Environments & Dynamics
- Static 2D: Fixed environment for direct algorithm comparison.
- Dynamic 2D: Real-time obstacle insertion, falling walls, and automatic path recalculation.

### Dynamic Obstacles
- Manual collision drawing.
- Random wall cluster spawning with a reachability guarantee.
- Automatic recovery if dynamic obstacles temporarily block connectivity.

### Swarm Control & Shepherding
- Optional spawning of swarm agents ("sheep").
- Shepherding-based guidance toward target destinations.
- Algorithm-aware sheep ordering and robust multi-agent following simulation.

### Map Handling
- Image-based map upload: .png, .jpg, .jpeg, .bmp, .tif, .tiff
- Automatic resolution handling:
  - Large images are downscaled for performance.
  - Small images are padded (no silent upscaling).
- Resized maps are saved with a `_resized` suffix to prevent data loss.

### Visualization & Analysis
- Real-time visualization of search and path construction.
- Optional performance plotting using `plot_results.py`.
- Supports execution-time and path comparison metrics.

## Requirements
- Python 3.8+
- Libraries:
  - tkinter
  - opencv-python (cv2)
  - numpy
  - Pillow (PIL)
  - matplotlib (optional — for `plot_results.py`)

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/iiCPO9/GUI-for-Robotic-Path-Planning.git
cd GUI-for-Robotic-Path-Planning
pip install opencv-python numpy Pillow matplotlib
```

(omit `matplotlib` if you don't need the plotting helper)

## Usage
Run the main application:

```bash
python main.py
```

The application window will open and present two modes:
- Static 2D
- Dynamic 2D

Typical workflow:
1. Upload a map image.
2. Set start and end points.
3. Select a path-planning algorithm.
4. (Optional) Enable shepherding and spawn sheep.
5. (Dynamic 2D) Add obstacles or spawn wall clusters.
6. Run and visualize the algorithm in real time.

To visualize algorithm performance results:

```bash
python plot_results.py
```

## File Structure
- `main.py` – Application entry point and main menu.
- `Static2D.py` – Static 2D path planning interface.
- `Dynamic2D.py` – Dynamic 2D interface with live replanning.
- `algorithms.py` – Dijkstra, A*, DFS, and Genetic Algorithm implementations.
- `shep.py` – Swarm control and shepherding logic.
- `randOB.py` – Random obstacle spawner with connectivity guarantees.
- `resolution_tool.py` – Image preprocessing and resolution adjustment.
- `plot_results.py` – Performance plotting and comparison.
- `images/` – UI icons and assets.

## Notes & Troubleshooting
- If a map is unsolvable initially, the system reports “No path found.”
- If dynamic wall spawning blocks connectivity, the system attempts automatic repair.
- Genetic Algorithm performance depends on parameter tuning and map size.
- Large dynamic maps with many agents may reduce real-time responsiveness.

## Limitations
- Supports 2D environments only.
- Simulation-based (no real robot hardware integration).
- Not optimized for very large-scale swarms.
- Genetic Algorithm can be computationally expensive on dense maps.

## Future Work
- 3D path planning support.
- ROS integration.
- Additional algorithms (RRT, D*, PRM).
- Performance optimization for large dynamic environments and swarms.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
