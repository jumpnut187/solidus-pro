"""Implicit P1 finite-element heat conduction on a tetrahedral mesh.

Contract C2 of ``docs/tet_migration/DESIGN.md``. Mesh-agnostic and SI only:
node coordinates in metres, conductivity in W/(m K), volumetric heat capacity
in J/(m^3 K), HTC in W/(m^2 K), temperatures in degC, time in seconds. The
caller converts inch geometry with :mod:`physics.units` before construction.

Each step solves the backward-Euler system with a lumped (diagonal) capacity
matrix and lumped Robin boundary terms::

    (C/dt + K + R(t+dt)) T_new = (C/dt) T_old + R(t+dt) T_inf

``C_i = sum_e rho_cp(T_e) V_e / 4`` and ``K`` is the P1 stiffness matrix with
per-element ``k(T_e)``, where ``T_e`` is the mean of the element's four nodal
temperatures at the start of the step (properties lagged one step). Each
boundary group adds ``h(t) * A_face / 3`` to the diagonal of its three face
nodes, with ``h(t) = htc * time_function(t)`` as in
:class:`physics.heat_transfer.BoundaryCondition`.

The sparsity pattern is built once, together with two gather matrices: the
stiffness gather ``S`` (nnz x T, ``K.data = S @ k_e``) and the node incidence
``Q`` (N x T, entries ``V_e / 4``, ``C = Q @ rho_cp_e``). A step refills the
CSR ``data`` array with one sparse product and solves with Jacobi-
preconditioned CG, falling back to a direct solve if CG fails. With numba the
element properties and the gathers run in parallel kernels and the CG loop in
one compiled kernel (parallel from ``PARALLEL_CG_MIN_NODES``). Without numba
the same products run through scipy and ``scipy.sparse.linalg.cg``. The CG
starts from ``T_old`` plus the last step's change scaled by ``dt``.

Usage::

    model = TetHeatTransferModel(nodes_m, tets, tet_material, materials, faces)
    model.add_boundary_group(cast_faces, htc=500.0, t_inf=25.0)
    model.set_initial_temperature(700.0)
    model.step(0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, cg, spsolve

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba ships with the app
    NUMBA_AVAILABLE = False

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: CG relative residual target, ``||b - A x|| <= CG_RTOL * ||b||``.
CG_RTOL = 1e-10
#: CG iteration cap before falling back to ``spsolve``.
CG_MAX_ITER = 5000
#: Node count from which the numba CG runs its parallel build. Measured on
#: thick_thin (18 cores): serial 25 ms vs parallel 33 ms at 35k nodes,
#: serial 315 ms vs parallel 123 ms at 215k nodes.
PARALLEL_CG_MIN_NODES = 100_000
#: Tets with ``|V| <= DEGENERATE_VOLUME_REL * mean(|V|)`` are rejected.
DEGENERATE_VOLUME_REL = 1e-12

# Local (a, b) index pairs of the 4x4 element matrix, row-major.
_LOCAL_A = np.repeat(np.arange(4), 4)
_LOCAL_B = np.tile(np.arange(4), 4)


if NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _element_properties_kernel(
        temperatures, tets, tet_material, temp_points, conductivity, rho_cp, n_points, k_out, rho_cp_out
    ):  # pragma: no cover - compiled
        """k and rho_cp per tet at the mean of its 4 nodal temperatures (np.interp clamping)."""
        for t in prange(tets.shape[0]):
            te = 0.25 * (
                temperatures[tets[t, 0]]
                + temperatures[tets[t, 1]]
                + temperatures[tets[t, 2]]
                + temperatures[tets[t, 3]]
            )
            m = tet_material[t]
            last = n_points[m] - 1
            if last == 0 or te <= temp_points[m, 0]:
                k_out[t] = conductivity[m, 0]
                rho_cp_out[t] = rho_cp[m, 0]
            elif te >= temp_points[m, last]:
                k_out[t] = conductivity[m, last]
                rho_cp_out[t] = rho_cp[m, last]
            else:
                j = 0
                while temp_points[m, j + 1] < te:
                    j += 1
                w = (te - temp_points[m, j]) / (temp_points[m, j + 1] - temp_points[m, j])
                k_out[t] = conductivity[m, j] + w * (conductivity[m, j + 1] - conductivity[m, j])
                rho_cp_out[t] = rho_cp[m, j] + w * (rho_cp[m, j + 1] - rho_cp[m, j])

    @njit(parallel=True, cache=True)
    def _csr_matvec_kernel(indptr, indices, data, x, out):  # pragma: no cover - compiled
        """out = A @ x for CSR (indptr, indices, data), parallel over rows."""
        for i in prange(indptr.shape[0] - 1):
            s = 0.0
            for jj in range(indptr[i], indptr[i + 1]):
                s += data[jj] * x[indices[jj]]
            out[i] = s

    @njit(parallel=True, cache=True)
    def _nodal_gradient_kernel(
        nodes, incidence_indptr, incidence_tets, incidence_weight, tets, grads, temperatures, tet_mask, out
    ):  # pragma: no cover - compiled
        """Weighted mean of P1 tet gradients at ``nodes``; see ``nodal_gradient``.

        ``tet_mask`` all True is the plain volume-weighted mean. A node with no
        masked incident tet falls back to all incident tets.
        """
        for row in prange(nodes.shape[0]):
            node = nodes[row]
            masked = np.zeros(3)
            masked_weight = 0.0
            full = np.zeros(3)
            full_weight = 0.0
            for jj in range(incidence_indptr[node], incidence_indptr[node + 1]):
                t = incidence_tets[jj]
                w = incidence_weight[jj]
                for axis in range(3):
                    g = 0.0
                    for a in range(4):
                        g += temperatures[tets[t, a]] * grads[t, a, axis]
                    full[axis] += w * g
                    if tet_mask[t]:
                        masked[axis] += w * g
                full_weight += w
                if tet_mask[t]:
                    masked_weight += w
            for axis in range(3):
                if masked_weight > 0.0:
                    out[row, axis] = masked[axis] / masked_weight
                else:
                    out[row, axis] = full[axis] / full_weight

    def _pcg_jacobi(indptr, indices, data, inv_diag, b, x, rtol, maxiter):  # pragma: no cover - compiled
        """Jacobi-preconditioned CG on CSR A, in place on x.

        Stops when ``||b - A x|| <= rtol * ||b||`` (scipy ``cg`` criterion with
        atol 0). Returns (iterations, converged).
        """
        n = b.shape[0]
        r = np.empty(n)
        p = np.empty(n)
        ap = np.empty(n)
        b_norm2 = 0.0
        for i in prange(n):
            b_norm2 += b[i] * b[i]
        tol2 = rtol * rtol * b_norm2
        rz = 0.0
        rr = 0.0
        for i in prange(n):
            s = 0.0
            for jj in range(indptr[i], indptr[i + 1]):
                s += data[jj] * x[indices[jj]]
            ri = b[i] - s
            r[i] = ri
            zi = inv_diag[i] * ri
            p[i] = zi
            rz += ri * zi
            rr += ri * ri
        if rr <= tol2:
            return 0, True
        for iteration in range(1, maxiter + 1):
            pap = 0.0
            for i in prange(n):
                s = 0.0
                for jj in range(indptr[i], indptr[i + 1]):
                    s += data[jj] * p[indices[jj]]
                ap[i] = s
                pap += p[i] * s
            if not pap > 0.0:
                return iteration, False
            alpha = rz / pap
            rr = 0.0
            rz_new = 0.0
            for i in prange(n):
                x[i] += alpha * p[i]
                ri = r[i] - alpha * ap[i]
                r[i] = ri
                rr += ri * ri
                rz_new += ri * inv_diag[i] * ri
            if rr <= tol2:
                return iteration, True
            beta = rz_new / rz
            rz = rz_new
            for i in prange(n):
                p[i] = inv_diag[i] * r[i] + beta * p[i]
        return maxiter, False

    #: Serial build (cached). Below ~100k nodes it beats the parallel build,
    #: whose per-region launch and reduction overhead dominates.
    _pcg_jacobi_kernel = njit(cache=True)(_pcg_jacobi)
    #: Parallel build of the same body, for PARALLEL_CG_MIN_NODES and up.
    #: Not cached: numba's cache key ignores the parallel flag, so a cached
    #: copy would collide with the serial build.
    _pcg_jacobi_parallel_kernel = njit(parallel=True)(_pcg_jacobi)


def _as_float_vector(name: str, values: Any) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_index_array(name: str, values: Any, shape_tail: tuple[int, ...]) -> IntArray:
    array = np.asarray(values)
    if array.size == 0 and array.ndim >= 1:
        array = array.reshape((0,) + shape_tail)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype, got {array.dtype}")
    if array.ndim != 1 + len(shape_tail) or array.shape[1:] != shape_tail:
        expected = "(n," + ",".join(str(s) for s in shape_tail) + ")" if shape_tail else "(n,)"
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    return array.astype(np.int64, copy=False)


def _check_range(name: str, values: IntArray, upper: int, what: str) -> None:
    if values.size and (values.min() < 0 or values.max() >= upper):
        raise ValueError(
            f"{name} indices must lie in [0, {upper}) ({what}), "
            f"got range [{values.min()}, {values.max()}]"
        )


@dataclass(frozen=True)
class TetMaterial:
    """Temperature-dependent thermal properties of one material.

    Attributes:
        temp_points: (P,) degC, strictly ascending. One point means constant
            properties.
        conductivity: (P,) W/(m K), positive.
        rho_cp: (P,) J/(m^3 K), density times specific heat, positive.
        latent_heat_vol: J/m^3, 0 for non-solidifying materials.

    Values outside ``temp_points`` are clamped to the end points (np.interp).
    """

    temp_points: np.ndarray
    conductivity: np.ndarray
    rho_cp: np.ndarray
    latent_heat_vol: float = 0.0

    def __post_init__(self) -> None:
        temp = _as_float_vector("temp_points", self.temp_points)
        k = _as_float_vector("conductivity", self.conductivity)
        rho_cp = _as_float_vector("rho_cp", self.rho_cp)
        if not (temp.size == k.size == rho_cp.size):
            raise ValueError(
                "temp_points, conductivity and rho_cp must have equal length, got "
                f"{temp.size}, {k.size}, {rho_cp.size}"
            )
        if temp.size > 1 and np.any(np.diff(temp) <= 0.0):
            raise ValueError("temp_points must be strictly ascending")
        if np.any(k <= 0.0):
            raise ValueError("conductivity must be positive")
        if np.any(rho_cp <= 0.0):
            raise ValueError("rho_cp must be positive")
        latent = float(self.latent_heat_vol)
        if not np.isfinite(latent) or latent < 0.0:
            raise ValueError(f"latent_heat_vol must be finite and >= 0, got {latent}")
        object.__setattr__(self, "temp_points", temp)
        object.__setattr__(self, "conductivity", k)
        object.__setattr__(self, "rho_cp", rho_cp)
        object.__setattr__(self, "latent_heat_vol", latent)


@dataclass(frozen=True)
class _BoundaryGroup:
    """Lumped Robin group: per-node area (m^2) on unique nodes."""

    nodes: IntArray
    node_area: FloatArray
    htc: float
    t_inf: float
    time_function: Optional[Callable[[float], float]]

    def htc_at(self, time: float) -> float:
        """Return h(time) = htc * time_function(time) in W/(m^2 K)."""
        scale = 1.0 if self.time_function is None else float(self.time_function(time))
        h = self.htc * scale
        if not np.isfinite(h) or h < 0.0:
            raise ValueError(f"boundary group HTC at t={time} s must be finite and >= 0, got {h}")
        return h


class TetHeatTransferModel:
    """Backward-Euler P1 FEM heat conduction with lumped capacity (SI units).

    Args:
        nodes_m: (N,3) node coordinates in metres.
        tets: (T,4) node indices. Either orientation is accepted.
        tet_material: (T,) index into ``materials`` per tet.
        materials: material property curves.
        boundary_faces: (F,3) node indices of boundary triangles. Robin groups
            reference rows of this array.

    Attributes:
        temperatures: (N,) float64 degC. Writes to the returned array are seen
            by the next step. Assignment copies into the same buffer.
        time: simulated time in s.
        last_iterations: CG iterations of the last step.
        last_residual: ``||b - A x|| / ||b||`` of the last step.
        last_used_direct_solver: True when the last step fell back to spsolve.
    """

    backend_description = (
        "numba CG (Jacobi) with spsolve fallback"
        if NUMBA_AVAILABLE
        else "scipy sparse CG (Jacobi) with spsolve fallback"
    )

    def __init__(
        self,
        nodes_m: Any,
        tets: Any,
        tet_material: Any,
        materials: Sequence[TetMaterial],
        boundary_faces: Any,
    ) -> None:
        nodes = np.asarray(nodes_m, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 4:
            raise ValueError(f"nodes_m must have shape (N>=4, 3), got {nodes.shape}")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("nodes_m contains non-finite coordinates")
        n_nodes = nodes.shape[0]

        tet_idx = _as_index_array("tets", tets, (4,))
        if tet_idx.shape[0] == 0:
            raise ValueError("tets is empty")
        _check_range("tets", tet_idx, n_nodes, "node count")
        n_tets = tet_idx.shape[0]

        materials = list(materials)
        if not materials:
            raise ValueError("materials is empty")
        for i, material in enumerate(materials):
            if not isinstance(material, TetMaterial):
                raise ValueError(f"materials[{i}] is not a TetMaterial")
        mat_idx = _as_index_array("tet_material", tet_material, ())
        if mat_idx.shape[0] != n_tets:
            raise ValueError(
                f"tet_material length {mat_idx.shape[0]} does not match tet count {n_tets}"
            )
        _check_range("tet_material", mat_idx, len(materials), "material count")

        faces = _as_index_array("boundary_faces", boundary_faces, (3,))
        _check_range("boundary_faces", faces, n_nodes, "node count")

        referenced = np.bincount(tet_idx.ravel(), minlength=n_nodes)
        if np.any(referenced == 0):
            raise ValueError(
                f"{int(np.count_nonzero(referenced == 0))} nodes are not referenced by any tet"
            )

        self.n_nodes = n_nodes
        self.n_tets = n_tets
        self._nodes = nodes
        self._tets = tet_idx
        self._materials = materials
        self._tet_material = mat_idx
        self._material_tets = [np.flatnonzero(mat_idx == m) for m in range(len(materials))]
        self._faces = faces

        # Geometry: edge matrix J (rows = x1-x0, x2-x0, x3-x0), |det J| / 6 = volume.
        x = nodes[tet_idx]  # (T,4,3)
        jac = x[:, 1:, :] - x[:, :1, :]
        volumes = np.abs(np.linalg.det(jac)) / 6.0
        mean_volume = float(volumes.mean())
        degenerate = volumes <= DEGENERATE_VOLUME_REL * mean_volume
        if np.any(degenerate):
            raise ValueError(f"{int(np.count_nonzero(degenerate))} tets have zero volume")
        self.tet_volumes: FloatArray = volumes

        # P1 gradients: grad(xi_i) is row i of inv(J)^T, grad(N0) = -sum of the rest.
        inv_t = np.transpose(np.linalg.inv(jac), (0, 2, 1))
        grads = np.empty((n_tets, 4, 3), dtype=np.float64)
        grads[:, 1:, :] = inv_t
        grads[:, 0, :] = -inv_t.sum(axis=1)
        self._grads = grads  # (T,4,3) 1/m
        # V_e * grad(N_a) . grad(N_b), flattened row-major to 16 entries (m).
        local_stiffness = (
            np.einsum("tai,tbi->tab", grads, grads) * volumes[:, None, None]
        ).reshape(n_tets, 16)

        # CSR pattern and the scatter map (tet, a, b) -> data index.
        rows = tet_idx[:, _LOCAL_A].ravel()
        cols = tet_idx[:, _LOCAL_B].ravel()
        keys, scatter = np.unique(rows * n_nodes + cols, return_inverse=True)
        scatter = scatter.astype(np.int64, copy=False).ravel()
        nnz = keys.size
        index_dtype = np.int32 if max(nnz, n_nodes, 16 * n_tets) < np.iinfo(np.int32).max else np.int64
        # Stiffness gather S (nnz x T): K.data = S @ k_e. Row i lists the
        # (tet, V * grad.grad) contributions to data entry i.
        order = np.argsort(scatter, kind="stable")
        gather_indptr = np.zeros(nnz + 1, dtype=index_dtype)
        np.cumsum(np.bincount(scatter, minlength=nnz), out=gather_indptr[1:])
        self._stiffness_gather = csr_array(
            (
                local_stiffness.ravel()[order],
                (order // 16).astype(index_dtype),
                gather_indptr,
            ),
            shape=(nnz, n_tets),
        )
        del local_stiffness, scatter, order
        # Node incidence Q (N x T) with entries V_e / 4: C = Q @ rho_cp_e.
        flat_nodes = tet_idx.ravel()
        order = np.argsort(flat_nodes, kind="stable")
        incidence_indptr = np.zeros(n_nodes + 1, dtype=index_dtype)
        np.cumsum(np.bincount(flat_nodes, minlength=n_nodes), out=incidence_indptr[1:])
        incidence_tets = (order // 4).astype(index_dtype)
        self._node_incidence = csr_array(
            (volumes[incidence_tets] / 4.0, incidence_tets, incidence_indptr),
            shape=(n_nodes, n_tets),
        )
        key_rows = keys // n_nodes
        indptr = np.zeros(n_nodes + 1, dtype=index_dtype)
        np.cumsum(np.bincount(key_rows, minlength=n_nodes), out=indptr[1:])
        indices = (keys % n_nodes).astype(index_dtype)
        self._diag_index = np.searchsorted(keys, np.arange(n_nodes, dtype=np.int64) * (n_nodes + 1))
        self._nnz = nnz
        self._indptr = indptr
        self._indices = indices
        self._matrix = csr_array(
            (np.zeros(nnz, dtype=np.float64), indices, indptr), shape=(n_nodes, n_nodes)
        )

        # Boundary face areas (m^2).
        if faces.shape[0]:
            p = nodes[faces]
            self.boundary_face_areas: FloatArray = 0.5 * np.linalg.norm(
                np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
            )
        else:
            self.boundary_face_areas = np.zeros(0, dtype=np.float64)

        latent_e = np.array([m.latent_heat_vol for m in materials], dtype=np.float64)[mat_idx]
        self._node_latent = self._node_incidence @ latent_e

        # Material curves packed for the numba property kernel (rows padded
        # with the last value; n_points gives the true length).
        max_points = max(m.temp_points.size for m in materials)
        self._curve_points = np.array([m.temp_points.size for m in materials], dtype=np.int64)
        self._curve_temp = np.empty((len(materials), max_points), dtype=np.float64)
        self._curve_k = np.empty_like(self._curve_temp)
        self._curve_rho_cp = np.empty_like(self._curve_temp)
        for m, material in enumerate(materials):
            for target, curve in (
                (self._curve_temp, material.temp_points),
                (self._curve_k, material.conductivity),
                (self._curve_rho_cp, material.rho_cp),
            ):
                target[m, : curve.size] = curve
                target[m, curve.size :] = curve[-1]
        self._k_e = np.empty(n_tets, dtype=np.float64)
        self._rho_cp_e = np.empty(n_tets, dtype=np.float64)
        self._node_buffer = np.empty(n_nodes, dtype=np.float64)

        self._groups: List[_BoundaryGroup] = []
        self._temperatures = np.zeros(n_nodes, dtype=np.float64)
        self.time = 0.0
        self.last_iterations = 0
        self.last_residual = 0.0
        self.last_used_direct_solver = False
        # Heat-step change and dt of the last solve, for the CG initial guess.
        self._last_change: Optional[FloatArray] = None
        self._last_dt = 0.0

    # ------------------------------------------------------------------ state

    @property
    def temperatures(self) -> FloatArray:
        """(N,) nodal temperatures in degC (the live buffer)."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, values: Any) -> None:
        self._temperatures[:] = self._validated_field(values)

    def _validated_field(self, values: Union[float, Any]) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(self.n_nodes, float(array))
        if array.shape != (self.n_nodes,):
            raise ValueError(f"temperature field must have shape ({self.n_nodes},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("temperature field contains non-finite values")
        return array

    def set_initial_temperature(self, value: Union[float, Any]) -> None:
        """Set nodal temperatures (degC) from a scalar or an (N,) array."""
        self._temperatures[:] = self._validated_field(value)
        self._last_change = None

    def add_boundary_group(
        self,
        face_indices: Any,
        htc: float,
        t_inf: float,
        time_function: Optional[Callable[[float], float]] = None,
    ) -> int:
        """Register a lumped Robin group on rows of ``boundary_faces``.

        Args:
            face_indices: unique indices into ``boundary_faces``.
            htc: base heat transfer coefficient, W/(m^2 K), >= 0.
            t_inf: ambient temperature, degC.
            time_function: optional multiplier; h(t) = htc * time_function(t).

        Returns:
            Group id (0-based, registration order).
        """
        idx = _as_index_array("face_indices", face_indices, ())
        _check_range("face_indices", idx, self._faces.shape[0], "boundary face count")
        if np.unique(idx).size != idx.size:
            raise ValueError("face_indices contains duplicates")
        htc = float(htc)
        t_inf = float(t_inf)
        if not np.isfinite(htc) or htc < 0.0:
            raise ValueError(f"htc must be finite and >= 0, got {htc}")
        if not np.isfinite(t_inf):
            raise ValueError(f"t_inf must be finite, got {t_inf}")
        if time_function is not None and not callable(time_function):
            raise ValueError("time_function must be callable or None")
        face_nodes = self._faces[idx].ravel()
        node_area = np.bincount(
            face_nodes,
            weights=np.repeat(self.boundary_face_areas[idx] / 3.0, 3),
            minlength=self.n_nodes,
        )
        nodes = np.unique(face_nodes)
        self._groups.append(
            _BoundaryGroup(nodes, node_area[nodes], htc, t_inf, time_function)
        )
        return len(self._groups) - 1

    # ------------------------------------------------------------ properties

    def _element_properties(self, temperatures: FloatArray) -> tuple[FloatArray, FloatArray]:
        """(k_e, rho_cp_e) per tet at the mean of its nodal temperatures.

        Returns internal buffers that the next call overwrites.
        """
        if NUMBA_AVAILABLE:
            _element_properties_kernel(
                temperatures, self._tets, self._tet_material, self._curve_temp,
                self._curve_k, self._curve_rho_cp, self._curve_points, self._k_e, self._rho_cp_e,
            )
            return self._k_e, self._rho_cp_e
        t_elem = temperatures[self._tets].mean(axis=1)
        for material, tet_ids in zip(self._materials, self._material_tets):
            if tet_ids.size == 0:
                continue
            for out, curve in ((self._k_e, material.conductivity), (self._rho_cp_e, material.rho_cp)):
                if curve.size == 1:
                    out[tet_ids] = curve[0]
                else:
                    out[tet_ids] = np.interp(t_elem[tet_ids], material.temp_points, curve)
        return self._k_e, self._rho_cp_e

    @staticmethod
    def _matvec(matrix: csr_array, x: FloatArray, out: FloatArray) -> FloatArray:
        """out = matrix @ x (parallel numba kernel when available)."""
        if NUMBA_AVAILABLE:
            _csr_matvec_kernel(matrix.indptr, matrix.indices, matrix.data, x, out)
        else:
            out[:] = matrix @ x
        return out

    def stiffness_matrix(self) -> csr_array:
        """Return the assembled conduction matrix K (W/K) at current temperatures."""
        k_e, _ = self._element_properties(self._temperatures)
        data = self._matvec(self._stiffness_gather, k_e, np.empty(self._nnz, dtype=np.float64))
        return csr_array(
            (data, self._indices.copy(), self._indptr.copy()),
            shape=(self.n_nodes, self.n_nodes),
        )

    def node_heat_capacity(self) -> FloatArray:
        """(N,) lumped heat capacity C_i in J/K at the current temperatures."""
        _, rho_cp_e = self._element_properties(self._temperatures)
        return self._matvec(self._node_incidence, rho_cp_e, np.empty(self.n_nodes, dtype=np.float64))

    def node_latent_capacity(self) -> FloatArray:
        """(N,) lumped latent heat in J: sum of latent_heat_vol_e * V_e / 4."""
        return self._node_latent.copy()

    def total_energy(self) -> float:
        """Return sum_i C_i T_i in J (degC reference) at the current temperatures."""
        return float(np.dot(self.node_heat_capacity(), self._temperatures))

    # ------------------------------------------------------------- gradients

    def nodal_gradient(self, tet_mask: Any = None, nodes: Any = None) -> FloatArray:
        """(N,3) temperature gradient in K/m: volume-weighted mean of P1 tet gradients.

        Args:
            tet_mask: optional (T,) bool. Only masked tets enter the average.
                A node with no incident masked tet gets the all-tet average.
                Use it to keep one material's gradient at an interface with a
                material of different conductivity, where the field has a kink.
            nodes: optional node indices. Returns (len(nodes), 3) for those
                nodes only, at a cost proportional to their incident tets.
        """
        if tet_mask is None:
            mask = np.ones(self.n_tets, dtype=bool)
        else:
            mask = np.asarray(tet_mask, dtype=bool)
            if mask.shape != (self.n_tets,):
                raise ValueError(f"tet_mask must have shape ({self.n_tets},), got {mask.shape}")
        if nodes is None:
            rows = np.arange(self.n_nodes, dtype=np.int64)
        else:
            rows = _as_index_array("nodes", nodes, ())
            _check_range("nodes", rows, self.n_nodes, "node count")
        incidence = self._node_incidence
        if NUMBA_AVAILABLE:
            out = np.empty((rows.size, 3), dtype=np.float64)
            _nodal_gradient_kernel(
                rows, incidence.indptr, incidence.indices, incidence.data,
                self._tets, self._grads, self._temperatures, mask, out,
            )
            return out
        grad_e = np.einsum("ta,tai->ti", self._temperatures[self._tets], self._grads)
        sub = incidence[rows]
        out = np.asarray(sub @ grad_e, dtype=np.float64)
        weight = np.asarray(sub @ np.ones(self.n_tets), dtype=np.float64)
        masked = np.asarray(sub @ (grad_e * mask[:, None]), dtype=np.float64)
        masked_weight = np.asarray(sub @ mask.astype(np.float64), dtype=np.float64)
        covered = masked_weight > 0.0
        out[covered] = masked[covered]
        weight[covered] = masked_weight[covered]
        out /= weight[:, None]
        return out

    def gradient_magnitude(self, tet_mask: Any = None, nodes: Any = None) -> FloatArray:
        """|grad T| in K/m, (N,) or (len(nodes),). Arguments as in :meth:`nodal_gradient`."""
        return np.linalg.norm(self.nodal_gradient(tet_mask, nodes), axis=1)

    def get_cooling_rate(self, prev: Any, dt: float) -> FloatArray:
        """(N,) cooling rate in degC/s, positive when cooling."""
        return (np.asarray(prev, dtype=np.float64) - self._temperatures) / float(dt)

    # ------------------------------------------------------------------ step

    def step(self, dt: float) -> None:
        """Advance one backward-Euler step of ``dt`` seconds."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        t_old = self._temperatures.copy()
        t_new_time = self.time + dt

        k_e, rho_cp_e = self._element_properties(t_old)
        data = self._matrix.data
        self._matvec(self._stiffness_gather, k_e, data)
        diag_extra = self._matvec(self._node_incidence, rho_cp_e, self._node_buffer)
        diag_extra /= dt  # C / dt
        rhs = diag_extra * t_old
        for group in self._groups:
            h = group.htc_at(t_new_time)
            if h == 0.0:
                continue
            robin = h * group.node_area
            diag_extra[group.nodes] += robin
            rhs[group.nodes] += robin * group.t_inf
        data[self._diag_index] += diag_extra
        inv_diag = 1.0 / data[self._diag_index]

        # Initial guess: T_old plus the last heat-step change scaled to this dt.
        guess = t_old.copy()
        if self._last_change is not None:
            guess += (dt / self._last_dt) * self._last_change
        if NUMBA_AVAILABLE:
            solution = guess
            kernel = (
                _pcg_jacobi_parallel_kernel
                if self.n_nodes >= PARALLEL_CG_MIN_NODES
                else _pcg_jacobi_kernel
            )
            iterations, converged = kernel(
                self._indptr, self._indices, data, inv_diag, rhs, solution, CG_RTOL, CG_MAX_ITER
            )
            info = 0 if converged else 1
        else:
            preconditioner = LinearOperator(
                (self.n_nodes, self.n_nodes), matvec=lambda r: inv_diag * r.ravel(), dtype=np.float64
            )
            iterations = 0

            def _count(_: FloatArray) -> None:
                nonlocal iterations
                iterations += 1

            solution, info = cg(
                self._matrix,
                rhs,
                x0=guess,
                rtol=CG_RTOL,
                maxiter=CG_MAX_ITER,
                M=preconditioner,
                callback=_count,
            )
        used_direct = info != 0 or not np.all(np.isfinite(solution))
        if used_direct:
            solution = spsolve(self._matrix.tocsc(), rhs)
            if not np.all(np.isfinite(solution)):
                raise FloatingPointError("tet heat solve produced non-finite temperatures")

        rhs_norm = float(np.linalg.norm(rhs))
        residual = float(np.linalg.norm(rhs - self._matrix @ solution))
        self.last_iterations = iterations
        self.last_residual = residual / rhs_norm if rhs_norm > 0.0 else residual
        self.last_used_direct_solver = bool(used_direct)
        self._last_change = solution - t_old
        self._last_dt = dt
        self._temperatures[:] = solution
        self.time = t_new_time
