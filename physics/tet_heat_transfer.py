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

The sparsity pattern and the (tet, a, b) -> CSR scatter map are built once.
A step only refills the CSR ``data`` array with ``np.bincount`` and solves with
Jacobi-preconditioned CG, falling back to a direct solve if CG fails.

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

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: CG relative residual target, ``||b - A x|| <= CG_RTOL * ||b||``.
CG_RTOL = 1e-10
#: CG iteration cap before falling back to ``spsolve``.
CG_MAX_ITER = 5000
#: Tets with ``|V| <= DEGENERATE_VOLUME_REL * mean(|V|)`` are rejected.
DEGENERATE_VOLUME_REL = 1e-12

# Local (a, b) index pairs of the 4x4 element matrix, row-major.
_LOCAL_A = np.repeat(np.arange(4), 4)
_LOCAL_B = np.tile(np.arange(4), 4)


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

    backend_description = "scipy sparse CG (Jacobi) with spsolve fallback"

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
        self._local_stiffness = (
            np.einsum("tai,tbi->tab", grads, grads) * volumes[:, None, None]
        ).reshape(n_tets, 16)

        # CSR pattern and scatter map (tet, a, b) -> data index.
        rows = tet_idx[:, _LOCAL_A].ravel()
        cols = tet_idx[:, _LOCAL_B].ravel()
        keys, scatter = np.unique(rows * n_nodes + cols, return_inverse=True)
        self._scatter = scatter.astype(np.int64, copy=False).ravel()
        nnz = keys.size
        index_dtype = np.int32 if max(nnz, n_nodes) < np.iinfo(np.int32).max else np.int64
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

        # Incident tet volume per node, for the volume-weighted gradient average.
        self._node_incident_volume = np.bincount(
            tet_idx.ravel(), weights=np.repeat(volumes, 4), minlength=n_nodes
        )
        latent_e = np.array([m.latent_heat_vol for m in materials], dtype=np.float64)[mat_idx]
        self._node_latent = np.bincount(
            tet_idx.ravel(), weights=np.repeat(latent_e * volumes / 4.0, 4), minlength=n_nodes
        )

        self._groups: List[_BoundaryGroup] = []
        self._temperatures = np.zeros(n_nodes, dtype=np.float64)
        self.time = 0.0
        self.last_iterations = 0
        self.last_residual = 0.0
        self.last_used_direct_solver = False

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

    def _element_temperatures(self, temperatures: FloatArray) -> FloatArray:
        return temperatures[self._tets].mean(axis=1)

    def _element_property(self, t_elem: FloatArray, name: str) -> FloatArray:
        """Per-tet property ``name`` ('conductivity' or 'rho_cp') at t_elem degC."""
        out = np.empty(self.n_tets, dtype=np.float64)
        for material, tet_ids in zip(self._materials, self._material_tets):
            if tet_ids.size == 0:
                continue
            curve = getattr(material, name)
            if curve.size == 1:
                out[tet_ids] = curve[0]
            else:
                out[tet_ids] = np.interp(t_elem[tet_ids], material.temp_points, curve)
        return out

    def _lumped_capacity(self, rho_cp_e: FloatArray) -> FloatArray:
        return np.bincount(
            self._tets.ravel(),
            weights=np.repeat(rho_cp_e * self.tet_volumes / 4.0, 4),
            minlength=self.n_nodes,
        )

    def _stiffness_data(self, k_e: FloatArray) -> FloatArray:
        return np.bincount(
            self._scatter,
            weights=(self._local_stiffness * k_e[:, None]).ravel(),
            minlength=self._nnz,
        )

    def stiffness_matrix(self) -> csr_array:
        """Return the assembled conduction matrix K (W/K) at current temperatures."""
        k_e = self._element_property(self._element_temperatures(self._temperatures), "conductivity")
        return csr_array(
            (self._stiffness_data(k_e), self._indices.copy(), self._indptr.copy()),
            shape=(self.n_nodes, self.n_nodes),
        )

    def node_heat_capacity(self) -> FloatArray:
        """(N,) lumped heat capacity C_i in J/K at the current temperatures."""
        rho_cp_e = self._element_property(self._element_temperatures(self._temperatures), "rho_cp")
        return self._lumped_capacity(rho_cp_e)

    def node_latent_capacity(self) -> FloatArray:
        """(N,) lumped latent heat in J: sum of latent_heat_vol_e * V_e / 4."""
        return self._node_latent.copy()

    def total_energy(self) -> float:
        """Return sum_i C_i T_i in J (degC reference) at the current temperatures."""
        return float(np.dot(self.node_heat_capacity(), self._temperatures))

    # ------------------------------------------------------------- gradients

    def nodal_gradient(self) -> FloatArray:
        """(N,3) temperature gradient in K/m: volume-weighted mean of P1 tet gradients."""
        grad_e = np.einsum("ta,tai->ti", self._temperatures[self._tets], self._grads)
        weighted = np.repeat(grad_e * self.tet_volumes[:, None], 4, axis=0)
        flat_nodes = self._tets.ravel()
        out = np.empty((self.n_nodes, 3), dtype=np.float64)
        for axis in range(3):
            out[:, axis] = np.bincount(flat_nodes, weights=weighted[:, axis], minlength=self.n_nodes)
        out /= self._node_incident_volume[:, None]
        return out

    def gradient_magnitude(self) -> FloatArray:
        """(N,) |grad T| in K/m."""
        return np.linalg.norm(self.nodal_gradient(), axis=1)

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

        t_elem = self._element_temperatures(t_old)
        k_e = self._element_property(t_elem, "conductivity")
        rho_cp_e = self._element_property(t_elem, "rho_cp")
        cap_over_dt = self._lumped_capacity(rho_cp_e) / dt

        diag_extra = cap_over_dt.copy()
        rhs = cap_over_dt * t_old
        for group in self._groups:
            h = group.htc_at(t_new_time)
            if h == 0.0:
                continue
            robin = h * group.node_area
            diag_extra[group.nodes] += robin
            rhs[group.nodes] += robin * group.t_inf

        data = self._matrix.data
        data[:] = self._stiffness_data(k_e)
        data[self._diag_index] += diag_extra
        inv_diag = 1.0 / data[self._diag_index]
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
            x0=t_old,
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
        self._temperatures[:] = solution
        self.time = t_new_time
