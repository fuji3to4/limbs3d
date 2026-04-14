# limbs3d

Python package for measuring perimeter and volume from 3D-scanned limb data (currently focused on legs).

Japanese version is available in README_jp.md.

## Features (roughly)

- Point cloud preprocessing (floor alignment, floor removal, noise removal, etc.)
- Left/right leg separation
- Mesh reconstruction and hole fixing from point clouds
- Mesh volume calculation
- Helper functions for perimeter measurement workflows

## Requirements

- Python 3.11+

Main dependencies:

- open3d
- pyvista
- pymeshfix
- scikit-learn
- lsq-ellipse

## Installation

### 1) Install dependencies

Run at the repository root:

```bash
uv sync
```

or:

```bash
pip install -r requirements.txt
```

### 2) Install as a package

```bash
pip install -e .
```

## Quick Start

```python
import open3d as o3d
from limbs3d.leg import separate2legs, align2Floor, deleteFloor
from limbs3d.volume import get_surface, get_fix, get_vol

# Load point cloud
pcd = o3d.io.read_point_cloud("your_leg_pointcloud.ply")

# Preprocess
pcd = align2Floor(pcd)
pcd = deleteFloor(pcd)

# Separate into left/right legs
left_leg, right_leg = separate2legs(pcd)

# Calculate volume of left leg (example)
mesh = get_surface(left_leg, depth=10)
mesh_fixed = get_fix(mesh)
volume = get_vol(mesh_fixed)

print("volume:", volume)
```

## Examples

- Notebook examples are available in the examples folder.
- A good starting point is perimeter/volume-related notebooks.

## Notes

- Results depend on the input coordinate system and scale (mm, cm, etc.).
- Floor detection and clustering thresholds usually need tuning for each scan condition.
- The current implementation is primarily intended for leg data.

## License

See LICENSE.