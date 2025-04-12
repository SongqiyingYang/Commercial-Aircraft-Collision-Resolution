# Commercial-Aircraft-Collision-Resolution

This repository contains the source code and experimental results for a Mixed-Integer Linear Programming (MILP)-based collision avoidance model designed for commercial aircraft formations. The proposed model ensures safe navigation while minimizing formation synergy loss, maneuver complexity, and avoidance time.

This work supports the experiments and methodology presented in the paper:  
**"Optimized Collision Avoidance Strategies for Commercial Aircraft Formations"**

---

## 🔧 Environment

- Python 3.11.0  
- Gurobi Optimizer 11.0.0 (academic license recommended)  

## 🛫 How to Run

The main entry point is:

```bash
python main.py
```

Before running, you may modify parameters inside `main.py` to customize the scenario. Key configurable settings include:

```python
A = 5  # Number of formation aircraft
intruder_degree = 0  # Intruder approaching angle in degrees
```

### Formation Shape Initialization (Excerpt)

```python
offset = np.zeros(A)
sideCount = 0
for ac in range(A):
    offset[ac] = sideCount
    if ac % 2 == 0:
        sideCount += 1

for ac in range(A):
    if ac == 0:
        sign = 0
    elif ac % 2 == 0:
        sign = 1
    else:
        sign = -1
    x0[0, ac] = startDist * (A - ac)
    x0[1, ac] = sign * (offset[ac] * 0.9 * wingspan)
    if D > 2:
        x0[2, ac] = 0.0
```

---

## 📁 Folder Structure

Each experiment result corresponds to a dedicated folder. Naming conventions are provided below for easier reference.

### `exp1/` – Intruder Approach Angles
- Investigates the effect of intruder approach angle.
- Example:  
  `e1d20f3.pdf` → Experiment 1, 20° intruder, 3 formation aircrafts.

### `exp2/` – Cost Function Weights
- Explores different maneuver/synergy weight ratios.
- Example:  
  `e2m2.pdf` → Maneuver weight : Synergy weight = 2.

### `exp3/` – Formation Shapes
- Compares V, Inverted-V, and Echelon formations.
- Example:  
  `e3d40_invertedV.pdf` → 40° intruder, Inverted-V formation.

### `exp4/` – Leader Position Flexibility
- Evaluates performance of formations with fixed vs. free leading aircraft.
- Example:  
  `e4f2d45f.pdf` → 2 aircraft, 45° intruder, **f** = free leader, **s** = solid (fixed) leader.

---

## 📦 Output and Visualization

Simulation results can be visualized dynamically using the provided `plot.py` script.

To generate trajectory animations or multi-view plots, follow these steps:

1. **Prepare solution files**: Get the following six `.csv` files:
    - `xMatrix0.csv`, `xMatrix1.csv`, `xMatrix2.csv` *(if 3D)*
    - `vMatrix0.csv`, `vMatrix1.csv`, `vMatrix2.csv` *(if 3D)*

2.  **Open and modify `plot.py`** to match your simulation settings:

```python
T = 50        # total number of time steps
delT = 0.8    #seconds per iteration 
A = 5         # number of aircraft in the formation
intruder_angle = 0  # intruder approach angle in degrees
D = 3         # dimensionality (2 for planar, 3 for spatial)
jetwashTrack = 1
```

3. **Load solution matrices** in `plot.py` (already provided in code):

```python
xMatrix0 = pd.read_csv('different_angles/0degree/5S50T500126xMatrix0.csv', index_col=0).values
xMatrix1 = pd.read_csv('different_angles/0degree/5S50T500126xMatrix1.csv', index_col=0).values
# Optional if 3D
xMatrix2 = pd.read_csv('different_angles/0degree/5S50T500126xMatrix2.csv', index_col=0).values

vMatrix0 = pd.read_csv('different_angles/0degree/5S50T500126vMatrix0.csv', index_col=0).values
vMatrix1 = pd.read_csv('different_angles/0degree/5S50T500126vMatrix1.csv', index_col=0).values
# Optional if 3D
vMatrix2 = pd.read_csv('different_angles/0degree/5S50T500126vMatrix2.csv', index_col=0).values
```

4. **Run the script**:

```bash
python plot.py
```

The script will produce multi-view trajectory plots (3D, top-down, and side views), showing:

---

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

---

## 📬 Contact

For questions or collaboration opportunities, feel free to reach out:

**Songqiying Yang**  
*King Abdullah University of Science and Technology (KAUST)*  
✉️ [songqiying.yang@kaust.edu.sa]  
