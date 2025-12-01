# Julia implementation of Chris Sims' `csminwel' optimization routine

The `csminwel` algorithm solves the unconstrained optimization problem:\

$$
\min_\{x \in \mathbb\{R\}^n\} f(x)
$$

where $f$ is a scalar-valued objective function. Unlike standard Quasi-Newton implementations, `csminwel` includes heuristics to recover from "bad gradients" (where derivatives are undefined or unreliable) and "cliffs" (steep drop-offs in the objective surface).\
\
---\

## 2. Mathematical Foundations\

### 2.1 Quasi-Newton Method\
The core of the algorithm is a Quasi-Newton iteration. At step $k$, we seek a search direction $d_k$ such that moving along $d_k$ reduces $f(x)$.\

The search direction is determined by:\

$$
d_k = -H_k \nabla f(x_k)
$$

where:
* $\nabla f(x_k)$ is the gradient vector at the current point.
* $H_k$ is an approximation of the *inverse Hessian* matrix, $[\nabla^2 f(x)]^\{-1\}$.

### 2.2 BFGS Update
The matrix $H_k$ is updated iteratively using the Broyden\'96Fletcher\'96Goldfarb\'96Shanno (BFGS) formula. This allows the algorithm to learn the curvature of the function without explicitly computing second derivatives.

Let:
* $s_k = x_\{k+1\} - x_k$ (Change in parameters)
* $y_k = \nabla f(x_\{k+1\}) - \nabla f(x_k)$ (Change in gradients)

The Inverse Hessian update is given by:\

$$
H_\{k+1\} = \left(I - \rho_k s_k y_k^T \right) H_k \left(I - \rho_k y_k s_k^T \right) + \rho_k s_k s_k^T
$$\
\
where $\rho_k = \frac\{1\}\{y_k^T s_k\}$.\
\
This formula ensures that $H_\{k+1\}$ satisfies the *secant equation* $H_\{k+1\} y_k = s_k$ and maintains symmetry and positive definiteness.\
\
### 2.3 Numerical Gradient Approximation\
If an analytical gradient is not provided, the algorithm approximates $\nabla f(x)$ using a central difference method. For each dimension $i$:\
\
$$\
[\nabla f(x)]_i \approx \frac\{f(x + \epsilon e_i) - f(x - \epsilon e_i)\}\{2\epsilon\}\
$$\
\
where $\epsilon$ is a small perturbation (default $10^\{-8\}$) and $e_i$ is the unit vector in the $i$-th direction.\
\
---\
\
## 3. Algorithmic Components\
\
### 3.1 The Main Driver: `csminwel`\
The main function with the **Cliff Detection Logic**.\
\
#### Cliff Detection Logic\
Standard optimizers often fail when the line search returns a "bad gradient" (undefined function value) or gets stuck. `csminwel` attempts to rescue the search:\
\
1.  **Detection:** If the line search returns a failure code (indicating a wall or cliff).\
2.  **Perturbation:** The Hessian $H$ is perturbed by adding random noise to its diagonal:\
    $$H_\{\text\{cliff\}\} = H + \text\{diag\}(\text\{rand\}(n)) \cdot \text\{diag\}(H)$$\
    This alters the search direction $d_k = -H_\{\text\{cliff\}\} \nabla f(x)$, potentially steering the optimizer away from the singularity.\
3.  **Reset:** If perturbation fails, the algorithm may reset the Hessian to the Identity matrix ($H=I$), reverting temporarily to Steepest Descent.\
\
### 3.2 Robust Line Search: `csminit`\
The function `csminit` performs a line search along the direction $d_k$. It seeks a scalar step size $\lambda$ to minimize:\
\
$$\
\phi(\lambda) = f(x_k + \lambda d_k)\
$$\
\
**Key features of this specific line search:**\
* It dynamically expands (grows) or shrinks the step size $\lambda$ based on performance.\
* It checks for "Bad Gradients" ($\nabla f$ is undefined or $f$ returns error).\
* It enforces strict descent conditions roughly equivalent to the Wolfe conditions, ensuring sufficient decrease in $f$.\
\
---\
\
## 4. Function Reference\
\
**`csminwel(fcn, x0, H0, ...)`**\
The main wrapper. It initializes variables, manages the iteration loop, handles the "cliff" logic, and calls the updater. It returns the optimal $x$, the final function value $f$, and the approximate Hessian $H$.\
\
**`csminit(fcn, x0, f0, g0, ...)`**\
The line search subroutine. It takes the current point $x_0$ and direction, searches for a better point $x_\{new\}$, and returns the new point along with a *return code* indicating success or the specific type of failure.\
\
**`bfgsi(H, delta_g, delta_x)`**\
Implements the matrix update formula described in Section 2.2. It takes the current $H$ and the changes in $x$ and $g$, returning the updated $H_\{new\}$.\
\
**`numerical_gradient(fcn, x)`**\
Computes the finite difference approximation of the gradient vector if an analytical gradient function is not supplied.\
\
---\
\
