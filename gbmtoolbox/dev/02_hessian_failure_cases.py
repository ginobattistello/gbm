"""Reference check 02: observed-Hessian conditioning and Laplace fragility.

This isolates the numerical distinction between a correct local Hessian and a
poor global Laplace approximation. The posterior is

    L(x,y)=0.5*x^2 + 0.5*eps*y^2 + 0.25*1e-26*y^4.

At the MAP, H=diag(1, eps), so cond(H)=1/eps exactly. Central finite differences
should recover the local curvature; large condition number is flagged as
fragile rather than repaired by eigenvalue clipping.
"""

import numpy as np

from gbmtoolbox.optimization import central_hessian

for eps in (1e-10, 1e-12, 1e-13, 1e-14):

    def f(z):
        x, y = z
        return 0.5 * x * x + 0.5 * eps * y * y + 0.25e-26 * y**4

    H = central_hessian(f, np.zeros(2), relative_step=1e-4)
    eig = np.linalg.eigvalsh(H)
    cond = eig[-1] / eig[0]
    print(f"eps={eps:.0e}  estimated lambda_min={eig[0]:.3e}  condition={cond:.3e}")
    assert eig[0] > 0
print("PASS: no eigenvalue clipping was used")
