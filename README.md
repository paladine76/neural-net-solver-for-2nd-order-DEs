# nn-de-solver

A Physics-Informed Neural Network (PINN) framework for solving 2nd-order differential equations using TensorFlow 2. The network learns to satisfy a differential equation by minimising a collocation residual loss, with no labelled data required — only the equation itself and its initial/boundary conditions.

---

## Project Structure

```
nn_de_solver/
├── __init__.py
├── model.py              # MLP architecture (tf.Module)
├── trainer.py            # Collocation training loop
└── problems/
    ├── base.py           # Abstract DEProblem interface
    ├── ode2.py           # 2nd-order linear ODE
    └── pde2.py           # 1D heat equation (parabolic PDE)

notebooks/
├── ode2_demo.ipynb       # ODE solver demo and results
└── pde2_demo.ipynb       # PDE solver demo and results
```

---

## Problems Implemented

### 2nd-Order ODE (`ode2.py`)

$$y\'\' + ay\' + by = f(x), \quad y(x_0) = y_0,\ y\'(x_0) = y_0\'$$

**Default demo:** critically-damped free oscillator

$$y\'\' + 6y\' + 9y = 0, \quad y(0) = 2,\ y\'(0) = 0$$

**Analytic solution:** $y(x) = (2 + 6x)\,e^{-3x}$

**Training:** converges in ~12,400 steps (~10 seconds on CPU). Final loss ≈ 0.025.

---

### 1D Heat Equation (`pde2.py`)

$$u_t = D\, u_{xx}, \quad x \in [x_{\text{lb}},\, x_{\text{ub}}],\ t \in [0,\, t_{\text{ub}}]$$

**Initial condition:** $u(x, 0) = \cos(\pi x) + \sin(\pi x)$

**Analytic solution:** $u(x, t) = (\cos \pi x + \sin \pi x)\, e^{-D\pi^2 t}$

**Constraints enforced:**
- IC sampled at 100 uniform points across $x$
- Spatial boundary values $u(x_{\text{lb}}, t)$ and $u(x_{\text{ub}}, t)$ sampled at 50 time points
- Point constraints: $u(0, 1)$, $u(-1, 1)$, $\partial u/\partial t\,(0, 1)$

**Default demo config:**

```python
HeatEquationProblem(x_lb=-1.0, x_ub=1.0, t_ub=3.0, diffusivity=1/np.pi**2)
```

This recovers the convention $u(x,t) = (\cos\pi x + \sin\pi x)\,e^{-t}$.

**Training:** 40,000 steps (~340 seconds on CPU). Final mean loss ≈ 0.002.

---

## Architecture

**`MLP`** (`model.py`) — a configurable `tf.Module` multilayer perceptron:

| Parameter | ODE default | PDE default |
|-----------|-------------|-------------|
| `input_dim` | 1 | 2 |
| `hidden` | `(128, 64)` | `(256, 256, 256, 256)` |
| `activation` | `sigmoid` | `tanh` |
| Weight init | Glorot uniform | Glorot uniform |
| Bias init | `random.normal` | `random.normal` |

Derivatives are computed via nested `tf.GradientTape`, enabling automatic differentiation through the network for residual computation.

---

## Training

**`Trainer`** (`trainer.py`) minimises:

$$\mathcal{L} = \sqrt{\frac{1}{2}\left(\text{MSE}_{\text{residual}} + \text{MSE}_{\text{constraints}}\right)}$$

Collocation points are sampled uniformly from the problem domain at each step. Optimiser: Adam with optional exponential LR decay (`decay_steps=5000`, `decay_rate=0.5`).

```python
trainer = Trainer(net, problem, lr=1e-3, use_lr_decay=True)
history = trainer.train(steps=40_000, batch_size=200, tol=5e-4, log_every=2_000)
```

**`@tf.function`** is applied to `_step` for graph-mode execution. Spatial/temporal input coordinates requiring gradient tracking inside `constraints()` are declared as persistent `tf.Variable` attributes on the problem class to remain compatible with graph tracing.

---

## Quick Start

```python
import numpy as np
import tensorflow as tf
from nn_de_solver.model import MLP
from nn_de_solver.trainer import Trainer

# --- ODE ---
from nn_de_solver.problems.ode2 import ODE2Problem
problem = ODE2Problem()
net = MLP(input_dim=1)
trainer = Trainer(net, problem)
history = trainer.train(steps=15_000, batch_size=50, tol=0.025)

# --- PDE ---
from nn_de_solver.problems.pde2 import HeatEquationProblem
problem = HeatEquationProblem(x_lb=-1.0, x_ub=1.0, t_ub=3.0, diffusivity=1/np.pi**2)
net = MLP(input_dim=2, hidden=(256, 256, 256, 256), activation='tanh')
trainer = Trainer(net, problem, lr=1e-3, use_lr_decay=True)
history = trainer.train(steps=40_000, batch_size=200, tol=5e-4)
```

---

## Extending to New Problems

Subclass `DEProblem` and implement three methods:

```python
from nn_de_solver.problems.base import DEProblem

class MyProblem(DEProblem):
    def residual(self, net, x):
        # Return DE residual tensor evaluated at collocation points x
        ...

    def constraints(self, net):
        # Return (pred, target) tensors for all ICs/BCs
        ...

    def domain(self):
        # ODE: return {'lb': float, 'ub': float}
        # PDE: return {'x_lb': ..., 'x_ub': ..., 't_lb': ..., 't_ub': ...}
        ...
```

The `Trainer` and `MLP` require no modification for new problems.

---

## Requirements

- Python 3.11+
- TensorFlow 2.15
- NumPy 1.24+
- Matplotlib

---

## Notes on PINN Training Difficulty

Standard PINNs are sensitive to problem scale. Known failure modes encountered and resolved in this project:

- **Trivial constant solutions**: the PDE residual $u_t - Du_{xx} = 0$ is satisfied by any constant. Avoided by strongly enforcing the IC over the full spatial domain, not just a few points.
- **Exploding gradients at initialisation**: caused by `tf.random.normal` weight init with deep networks. Fixed by switching to Glorot (Xavier) uniform initialisation.
- **Boundary divergence**: the network is unconstrained at spatial boundaries without explicit BC enforcement. Fixed by sampling BC residuals across the full time horizon.
- **Spectral bias**: networks with smooth activations struggle with high-frequency spatial inputs. Mitigated by using `tanh` and restricting the spatial domain to a single oscillation period during initial validation.
