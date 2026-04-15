# Codex Context: Python simulation for best-fitting axis constraint from a 3D trajectory

## Goal
Build a **simple Python simulation and fitting tool** that takes as input a sampled **3D trajectory of a single point** and outputs the **best-fitting motion constraint** among:

1. **Circular motion around an axis**
2. **Translation parallel to an axis**
3. **Screw motion around an axis**

The implementation should follow a **fixed-axis cylindrical-coordinate formulation** in **discrete time**. Do **not** rely on continuous-time instantaneous kinematics.

---

## Problem statement
Given a discrete trajectory

\[
p_k \in \mathbb{R}^3, \quad k=1,\dots,N
\]

and optionally timestamps

\[
t_k
\]

estimate a candidate axis

\[
\mathcal{L} = \{c + \lambda u : \lambda \in \mathbb{R}\}, \quad \|u\| = 1
\]

where:
- \(u \in \mathbb{R}^3\) is the axis direction
- \(c \in \mathbb{R}^3\) is one canonical point on the axis

Then convert the trajectory to **cylindrical coordinates around that axis** and classify the trajectory as:
- `circle`
- `translation`
- `screw`

based on model residuals.

---

## Core formulation

### 1) Candidate axis and radius
Fit a cylinder-like model to the trajectory by estimating \((u, c, r)\), where \(r\) is the distance from the point trajectory to the axis.

Use the objective:

\[
E_{\text{cyl}}(u,c,r) = \sum_{k=1}^N \left(\|(I-uu^T)(p_k-c)\| - r\right)^2
\]

subject to:

\[
\|u\| = 1, \qquad u^T c = 0
\]

Notes:
- \(u^T c = 0\) is a gauge choice so that \(c\) is the point on the axis closest to the origin.
- A practical first implementation can use a robust approximation / optimization rather than an exact closed-form solution.

### 2) Axis frame
Choose an orthonormal basis \((e_1, e_2, u)\) with:

\[
e_1^T u = 0, \quad e_2^T u = 0, \quad e_2 = u \times e_1
\]

and define:

\[
B = [e_1\ e_2\ u] \in SO(3)
\]

### 3) Cylindrical coordinates
For each point \(p_k\), compute:

\[
x_k = B^T (p_k - c)
\]

with

\[
x_k = \begin{bmatrix} x_k^{(1)} \\ x_k^{(2)} \\ x_k^{(3)} \end{bmatrix}
\]

Then define:

\[
\rho_k = \sqrt{(x_k^{(1)})^2 + (x_k^{(2)})^2}
\]

\[
\phi_k = \operatorname{atan2}(x_k^{(2)}, x_k^{(1)})
\]

\[
h_k = x_k^{(3)}
\]

Unwrap \(\phi_k\) over time.

---

## Discrete-time model classes

### A) Circular motion around axis
Ideal behavior:

\[
\rho_k \approx r, \qquad h_k \approx h_0, \qquad \phi_k \text{ varies}
\]

Fit constants \(r, h_0\) and use residual:

\[
E_{\text{circ}} = \sum_{k=1}^N \left[(\rho_k-r)^2 + (h_k-h_0)^2\right]
\]

### B) Translation parallel to axis
Ideal behavior:

\[
\rho_k \approx r, \qquad \phi_k \approx \phi_0, \qquad h_k \text{ varies}
\]

Fit constants \(r, \phi_0\) and use residual:

\[
E_{\text{trans}} = \sum_{k=1}^N \left[(\rho_k-r)^2 + d_{\angle}(\phi_k, \phi_0)^2\right]
\]

where \(d_{\angle}\) is wrapped angular distance.

### C) Screw motion around axis
Ideal behavior:

\[
\rho_k \approx r, \qquad h_k \approx h_0 + k\phi_k
\]

Fit \(r, h_0, k\) and use residual:

\[
E_{\text{screw}} = \sum_{k=1}^N \left[(\rho_k-r)^2 + (h_k-h_0-k\phi_k)^2\right]
\]

where \(k\) is the pitch per radian.

---

## Classification logic
Compute the three model residuals and choose the lowest one.

Recommended output:
- predicted label: `circle`, `translation`, or `screw`
- fitted axis direction \(u\)
- fitted axis point \(c\)
- fitted radius \(r\)
- fitted screw pitch \(k\) when applicable
- normalized residuals for all models
- confidence score or margin between best and second-best residual

Optional normalization:
- divide each residual by \(N\)
- optionally scale by trajectory size or variance for comparability

---

## Recommended implementation scope
Keep the code simple and readable. Prefer a **single-file Python prototype** or a **small module with 2–3 files**.

Suggested files:

- `trajectory_constraint_fit.py`
  - main fitting/classification logic
- `simulate_trajectories.py`
  - generate synthetic circle / translation / screw trajectories with optional noise
- `demo.py`
  - run simulation, fit, print results, optionally visualize

A single-file implementation is also acceptable if it is simpler.

---

## Recommended Python stack
Use:
- `numpy`
- `scipy` (`scipy.optimize.least_squares` is acceptable)
- `matplotlib` for optional visualization

Avoid unnecessary dependencies.

---

## Suggested algorithmic approach

### Stage 1: initialization
Use a practical initialization for \((u,c,r)\).

Possible initialization strategy:
1. Compute trajectory centroid
   \[
   \bar p = \frac{1}{N}\sum_k p_k
   \]
2. Run PCA/SVD on centered points
3. Use principal directions to initialize a candidate axis direction
4. Initialize \(c\) near the centroid, projected to satisfy \(u^T c = 0\)
5. Initialize \(r\) as the mean distance to the candidate axis

Because circle, translation, and screw all live approximately on a cylinder, a cylinder fit is the right common initialization.

### Stage 2: cylinder refinement
Refine \((u,c,r)\) by minimizing:

\[
E_{\text{cyl}}(u,c,r)
\]

A practical parametrization is:
- represent axis direction \(u\) with two angles, or optimize a 3-vector and renormalize inside the objective
- represent \(c\) as a 3-vector and enforce \(u^T c = 0\) after each update or as a penalty
- optimize \(r > 0\)

### Stage 3: cylindrical coordinate conversion
Build \(e_1, e_2\), transform all points to the axis frame, compute \((\rho_k, \phi_k, h_k)\), and unwrap \(\phi_k\).

### Stage 4: model fitting
Fit the three models:
- circle
- translation
- screw

Implementation details:
- circle: fit \(r = \text{mean}(\rho_k)\), \(h_0 = \text{mean}(h_k)\)
- translation: fit \(r = \text{mean}(\rho_k)\), \(\phi_0\) using a circular mean
- screw: fit \(r = \text{mean}(\rho_k)\), then linear least squares for \(h_k \approx h_0 + k\phi_k\)

### Stage 5: classification
Choose the model with the smallest normalized residual.

### Stage 6: optional refinement by model
Optional but desirable:
- once a model is selected, refine \((u,c,r)\) jointly with model-specific parameters
- this is a bonus, not required for the first version

---

## Utility functions to implement
Please implement small, testable functions such as:

- `normalize(v)`
- `build_axis_frame(u)`
- `project_to_axis_frame(points, u, c)`
- `unwrap_angles(phi)`
- `wrapped_angle_distance(a, b)`
- `fit_cylinder_axis(points)`
- `cylindrical_coordinates(points, u, c)`
- `fit_circle_model(rho, h)`
- `fit_translation_model(rho, phi)`
- `fit_screw_model(rho, phi, h)`
- `classify_trajectory(points, times=None)`

Recommended main API:

```python
result = classify_trajectory(points, times=None)
```

with output as a dictionary or dataclass containing:

```python
{
    "label": "circle" | "translation" | "screw",
    "axis_direction": np.ndarray,   # shape (3,)
    "axis_point": np.ndarray,       # shape (3,)
    "radius": float,
    "pitch": float | None,
    "residuals": {
        "circle": float,
        "translation": float,
        "screw": float,
    },
    "confidence": float,
    "rho": np.ndarray,
    "phi": np.ndarray,
    "h": np.ndarray,
}
```

---

## Simulation requirements
Create synthetic trajectories for all three classes.

### 1) Circle generator
Generate points from:

\[
p_k = c + r\cos\phi_k\,e_1 + r\sin\phi_k\,e_2 + h_0 u
\]

### 2) Translation generator
Generate points from:

\[
p_k = c + r\cos\phi_0\,e_1 + r\sin\phi_0\,e_2 + h_k u
\]

### 3) Screw generator
Generate points from:

\[
p_k = c + r\cos\phi_k\,e_1 + r\sin\phi_k\,e_2 + (h_0 + k\phi_k)u
\]

Add optional Gaussian noise to the 3D positions.

The demo should:
- simulate one trajectory of each class
- run the fitter/classifier
- print fitted parameters and residuals
- optionally show a 3D plot and a cylindrical-coordinate plot

---

## Visualization requirements (optional but useful)
If plotting is included, create:

1. **3D plot**
   - original trajectory points
   - fitted axis line
   - optionally fitted circular / helical / line reconstruction

2. **Cylindrical plots**
   - \(\rho_k\) vs sample index
   - \(\phi_k\) vs sample index
   - \(h_k\) vs sample index
   - optionally \(h_k\) vs \(\phi_k\)

These plots make the classification interpretable.

---

## Edge cases and caveats
Handle or at least detect the following:

1. **Very small radius**
   - if \(r \approx 0\), axis estimation is ill-conditioned
   - classification between translation and screw may become unreliable

2. **Very short arc**
   - a short circular arc can resemble a line or helix segment
   - confidence should be reduced

3. **Nearly constant trajectory**
   - motion may be too small to classify

4. **Angle wrapping**
   - always unwrap \(\phi_k\) before screw fitting

5. **Noisy data**
   - prefer robust global fitting over finite-difference-based derivatives

6. **Translation along the axis with arbitrary radial offset**
   - the point trajectory is a line parallel to the axis at constant radius
   - do not assume the line passes through the axis

---

## Acceptance criteria
The resulting code should:

1. Accept an `Nx3` NumPy array of trajectory samples
2. Estimate a best-fitting axis and radius
3. Convert the trajectory to cylindrical coordinates around that axis
4. Compute residuals for circle / translation / screw
5. Return the best-fitting class and fitted parameters
6. Correctly classify synthetic trajectories with small noise in the demo
7. Be simple, readable, and easy to extend

---

## Preferred coding style
- Clear functions with docstrings
- Minimal but useful comments
- Deterministic synthetic examples with a random seed
- No overengineering
- Prioritize correctness and clarity over maximum mathematical sophistication

---

## Nice-to-have improvements
If time permits, include:
- confidence score based on residual gap
- optional nonlinear refinement after initial classification
- support for irregular timestamps
- support for JSON/CSV trajectory input
- CLI arguments for demo settings

---

## Minimal deliverable
At minimum, deliver a working Python script that:

1. generates synthetic trajectories for circle / translation / screw,
2. estimates a candidate axis,
3. computes cylindrical coordinates,
4. evaluates the three residual models,
5. prints the predicted class and fitted parameters.

---

## Short instruction to Codex
Implement a simple Python prototype for classifying a single 3D point trajectory as `circle`, `translation`, or `screw` around a best-fitting axis. Use a common cylinder/axis fit, transform points to cylindrical coordinates around that axis, then compare the three discrete-time residual models described above. Include a small simulation/demo and keep the code readable.
