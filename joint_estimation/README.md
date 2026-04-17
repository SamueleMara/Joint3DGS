# Joint3DGS Trajectory Constraint Demo

This repository contains a small Python prototype for classifying a sampled 3D point trajectory as one of:

- `circle`
- `translation`
- `screw`

The code fits a fixed axis, converts the trajectory into cylindrical coordinates around that axis, and compares simple discrete-time residual models.

## Files

- [trajectory_constraint_fit.py](/home/samuelemara/Joint3DGS/trajectory_constraint_fit.py): axis fitting, cylindrical conversion, model fitting, and classification
- [simulate_trajectories.py](/home/samuelemara/Joint3DGS/simulate_trajectories.py): synthetic circle / translation / screw generators
- [demo.py](/home/samuelemara/Joint3DGS/demo.py): random demo, plotting, and real-trajectory loading
- [environment.yml](/home/samuelemara/Joint3DGS/environment.yml): conda environment spec
- [example_real_trajectory.csv](/home/samuelemara/Joint3DGS/example_real_trajectory.csv): minimal input example
- [theory.md](/home/samuelemara/Joint3DGS/theory.md): theoretical description of the implemented method

## Dependencies

The project is intentionally small. It uses:

- Python `3.7.12`
- `numpy`
- `scipy`
- `matplotlib`

The provided conda environment pins:

- `numpy=1.21.5`
- `scipy=1.7.3`
- `matplotlib=3.5.3`

## Environment

Create the environment:

```bash
conda env create -f /home/samuelemara/Joint3DGS/environment.yml
conda activate joint-3dgs
```

The environment name is `joint-3dgs`. It matches the Python version used by the existing `gaussian_splatting` environment.

## Practical Implementation

The implementation is deliberately pragmatic rather than fully symbolic.

### 1. Axis estimation

The code first computes PCA directions from the trajectory. It then evaluates a few axis hypotheses:

- `circle`: use the smallest-variance PCA direction as the axis direction candidate
- `translation`: use the largest-variance PCA direction as the motion / axis direction
- `screw`: run a general cylinder fit with nonlinear least squares

For circle and screw, the axis point and radius are refined by minimizing radial-distance residuals to a cylinder-like model.

### 2. Cylindrical coordinates

After choosing an axis candidate, the points are expressed in an orthonormal axis frame:

- radial distance `rho`
- angle `phi`
- axial coordinate `h`

The angle is unwrapped over the sample order.

### 3. Model fitting

Each class is scored with a simple residual:

- `circle`: constant radius and approximately constant axial height
- `translation`: constant radius and approximately constant angle
- `screw`: constant radius and linear relation between `h` and `phi`

The lowest residual wins.

### 4. Output

The classifier returns:

- predicted label
- axis direction
- axis point
- fitted radius
- fitted pitch for screw
- circle center for circle
- normalized residuals
- confidence
- estimated motion extent from the fitted cylindrical coordinates

## Demo Modes

### Random synthetic demo

This is the default mode. It randomly picks one of the three motion types, generates a synthetic trajectory, fits the model, prints the result, and shows a 3D plot.

```bash
python /home/samuelemara/Joint3DGS/demo.py
python /home/samuelemara/Joint3DGS/demo.py --seed 7
```

### All synthetic cases

```bash
python /home/samuelemara/Joint3DGS/demo.py --all
```

### No-plot mode

```bash
python /home/samuelemara/Joint3DGS/demo.py --no-plot
```

## Real Trajectory Input

You can classify a real trajectory instead of generating a synthetic one.

Recommended format: CSV with shape `Nx3`.

Example:

```csv
0.0,0.0,0.0
0.1,0.0,0.0
0.2,0.0,0.0
```

Supported formats:

- `.csv`
- `.npy`
- `.json`

Run:

```bash
python /home/samuelemara/Joint3DGS/demo.py --input /path/to/trajectory.csv
```

For JSON, supported forms are:

```json
[[0, 0, 0], [1, 0, 0], [2, 0, 0]]
```

or

```json
{"points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}
```

## Interpretation Notes

- For `circle`, the `axis_point` is not the circle center. It is the canonical point on the axis closest to the origin. The reported `circle_center` is the 3D center of the fitted circle.
- For `translation`, the axis direction is identifiable from a single tracked point, but the exact parallel axis location is not unique.
- For `screw`, both angular span and axial displacement are estimated from the fitted cylindrical coordinates.

## Limitations

- Very short arcs can be ambiguous.
- Very small-radius trajectories are ill-conditioned.
- Translation axis position is non-unique for a single tracked point.
- The method is designed as a readable prototype, not a production-grade estimator.
