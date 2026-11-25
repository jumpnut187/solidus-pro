import numpy as np
from typing import Optional, Tuple, Callable, Union
from enum import Enum
import platform
import warnings
import io
import contextlib
import multiprocessing
from config.hardware_config import hardware

def _restore_default_fp_environment() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        default_env = ctypes.c_char.in_dll(libc, "_FE_DFL_ENV")
        return int(libc.fesetenv(ctypes.byref(default_env))) == 0
    except (OSError, ValueError, AttributeError):
        return False

def _restore_fp_environment_after_taichi_init() -> None:
    if not _restore_default_fp_environment() and platform.system() == "Darwin":
        warnings.warn(
            "Could not restore the default floating-point environment after Taichi init; "
            "flush-to-zero may still be set and scipy cKDTree can crash (SIGSEGV). "
            "Check np.float64(5e-324) > 0.",
            RuntimeWarning,
            stacklevel=2,
        )

class GPUBackend(Enum):
    NONE = "none"
    METAL = "metal"
    CUDA = "cuda"
    OPENCL = "opencl"

class GPUComputeManager:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._detection_attempted = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.backend = GPUBackend.NONE
        self.device_name = "CPU"
        self.device_memory_gb = 0.0

        self._cuda_available = False
        self._metal_available = False
        self._opencl_available = False

        self._cupy = None
        self._opencl_context = None
        self._opencl_queue = None
        self._metal_device = None

        self._mlx = None
        self._taichi_initialized = False

        self._kernel_times = {}

        self._initialized = True

    def _ensure_initialized(self):
        if not self._detection_attempted:
            self._detection_attempted = True
            self._detect_backends()

    def _detect_backends(self):
        system = platform.system()
        is_main_process = multiprocessing.current_process().name == "MainProcess"
        devnull = contextlib.nullcontext() if is_main_process else contextlib.redirect_stdout(io.StringIO())

        if is_main_process:
            print("\n" + "="*60)
            print("GPU Compute Manager: Detecting backends...")
            print("="*60)

        with devnull:
            cuda_ready = self._try_init_cuda()
        if cuda_ready:
            self.backend = GPUBackend.CUDA
            if is_main_process:
                print(f"✓ Selected backend: CUDA ({self.device_name})")
                print(f"  GPU Memory: {self.device_memory_gb:.1f} GB")
                print("="*60 + "\n")
            return

        if system == 'Darwin' and platform.machine() == 'arm64':
            with devnull:
                metal_ready = self._try_init_metal()
            if metal_ready:
                self.backend = GPUBackend.METAL
                if is_main_process:
                    print(f"✓ Selected backend: Metal ({self.device_name})")
                    print("=" * 60 + "\n")
                return
            else:
                if is_main_process:
                    print("  Apple Silicon (arm64) - MLX/Taichi Metal bridge not available")

        with devnull:
            opencl_ready = self._try_init_opencl()
        if opencl_ready:
            self.backend = GPUBackend.OPENCL
            if is_main_process:
                print(f"✓ Selected backend: OpenCL ({self.device_name})")
                print(f"  GPU Memory: {self.device_memory_gb:.1f} GB")
                print("="*60 + "\n")
            return

        self.backend = GPUBackend.NONE
        self.device_name = "CPU (Numba JIT)"

        from config.hardware_config import hardware as hw_config
        numba_threads = hw_config.get_optimal_thread_count('compute')

        if is_main_process:
            print("✓ Selected backend: CPU (Numba JIT)")
            print(f"  Using {numba_threads} parallel threads")

        system = platform.system()
        if is_main_process:
            if system == 'Darwin':
                print("  Note: macOS deprecated OpenCL - Metal requires native implementation")
            else:
                print("  For GPU acceleration, install:")
                print("    - NVIDIA: pip install cupy-cuda12x")
                print("    - Other: pip install pyopencl")
            print("=" * 60 + "\n")

    def _try_init_cuda(self) -> bool:
        try:
            import cupy as cp

            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                print("  CUDA: No NVIDIA GPU found")
                return False

            device = cp.cuda.Device(0)
            self.device_name = device.pci_bus_id

            try:
                props = cp.cuda.runtime.getDeviceProperties(0)
                self.device_name = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
            except:
                self.device_name = f"CUDA Device {device.id}"

            mem_info = device.mem_info
            self.device_memory_gb = mem_info[1] / (1024**3)

            self._cupy = cp
            self._cuda_available = True
            print(f"  CUDA: Found {self.device_name}")
            return True

        except ImportError:
            print("  CUDA: CuPy not installed")
            return False
        except Exception as e:
            print(f"  CUDA: Initialization failed - {e}")
            return False

    def _try_init_metal(self) -> bool:
        try:
            import mlx.core as mx

            _ = mx.array([1.0, 2.0, 3.0])
            _ = mx.sum(_)

            self._mlx = mx
            self._metal_available = True
            self.device_name = "Apple Silicon GPU (Metal via MLX + Taichi)"
            print(f"  Metal: MLX initialized ({self.device_name})")

        except ImportError:
            print("  Metal: MLX not installed (pip install mlx)")
            return False
        except Exception as e:
            print(f"  Metal: MLX init failed - {e}")
            return False

        try:
            import taichi as ti

            if not self._taichi_initialized:
                ti.init(arch=ti.metal, log_level=ti.WARN)
                _restore_fp_environment_after_taichi_init()
                self._taichi_initialized = True
                print(f"  Metal: Taichi Metal backend initialized")

            return True

        except ImportError:
            print("  Metal: Taichi not installed (pip install taichi)")
            return False
        except Exception as e:
            print(f"  Metal: Taichi init failed - {e}")
            return False

    def _try_init_opencl(self) -> bool:
        try:
            import pyopencl as cl

            platforms = cl.get_platforms()
            if not platforms:
                print("  OpenCL: No platforms found")
                return False

            for plat in platforms:
                try:
                    devices = plat.get_devices(device_type=cl.device_type.GPU)
                    if devices:
                        self._opencl_context = cl.Context(devices=[devices[0]])
                        self._opencl_queue = cl.CommandQueue(self._opencl_context)
                        self.device_name = devices[0].name
                        self.device_memory_gb = devices[0].global_mem_size / (1024**3)
                        self._opencl_available = True
                        print(f"  OpenCL: Found {self.device_name}")
                        return True
                except:
                    continue

            print("  OpenCL: No GPU devices found")
            return False

        except ImportError:
            print("  OpenCL: PyOpenCL not installed")
            return False
        except Exception as e:
            print(f"  OpenCL: Initialization failed - {e}")
            return False

    @property
    def is_gpu_available(self) -> bool:
        self._ensure_initialized()
        return self.backend != GPUBackend.NONE

    def get_array_module(self):
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return self._cupy
        return np

    def to_gpu(self, arr: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return self._cupy.asarray(arr)
        return arr

    def to_cpu(self, arr) -> np.ndarray:
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            if hasattr(arr, 'get'):
                return arr.get()
        return np.asarray(arr)

    def compute_distances_gpu(self,
                              points: np.ndarray,
                              reference_points: np.ndarray) -> np.ndarray:
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA:
            return self._compute_distances_cuda(points, reference_points)
        elif self.backend == GPUBackend.METAL:
            return self._compute_distances_metal(points, reference_points)
        elif self.backend == GPUBackend.OPENCL:
            return self._compute_distances_opencl(points, reference_points)
        else:
            from core.acceleration import compute_min_distances_parallel
            return compute_min_distances_parallel(
                np.ascontiguousarray(points, dtype=np.float32),
                np.ascontiguousarray(reference_points, dtype=np.float32)
            )

    def _get_distance_chunk_size(self) -> int:
        try:
            from config.settings import get_settings
            chunk_size = int(get_settings().performance.gpu_distance_chunk_size)
        except Exception:
            chunk_size = 10000
        return max(1, chunk_size)

    def _compute_distances_cuda(self, points: np.ndarray,
                                reference_points: np.ndarray) -> np.ndarray:
        cp = self._cupy

        points_gpu = cp.asarray(points, dtype=cp.float32)
        refs_gpu = cp.asarray(reference_points, dtype=cp.float32)

        n_points = points_gpu.shape[0]
        n_refs = refs_gpu.shape[0]

        chunk_size = min(self._get_distance_chunk_size(), n_points)
        min_distances = cp.empty(n_points, dtype=cp.float32)

        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            chunk = points_gpu[start:end]

            diff = chunk[:, cp.newaxis, :] - refs_gpu[cp.newaxis, :, :]
            distances = cp.sqrt(cp.sum(diff ** 2, axis=2))

            min_distances[start:end] = cp.min(distances, axis=1)

        return cp.asnumpy(min_distances)

    def _compute_distances_opencl(self, points: np.ndarray,
                                  reference_points: np.ndarray) -> np.ndarray:
        import pyopencl as cl
        import pyopencl.array as cl_array

        kernel_source = """
        __kernel void min_distance(
            __global const float* points,
            __global const float* refs,
            __global float* min_dists,
            const int n_refs
        ) {
            int gid = get_global_id(0);
            
            float px = points[gid * 3];
            float py = points[gid * 3 + 1];
            float pz = points[gid * 3 + 2];
            
            float min_dist = 1e30f;
            
            for (int j = 0; j < n_refs; j++) {
                float dx = px - refs[j * 3];
                float dy = py - refs[j * 3 + 1];
                float dz = pz - refs[j * 3 + 2];
                float dist = sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            
            min_dists[gid] = min_dist;
        }
        """

        program = cl.Program(self._opencl_context, kernel_source).build()

        points_flat = np.ascontiguousarray(points.flatten(), dtype=np.float32)
        refs_flat = np.ascontiguousarray(reference_points.flatten(), dtype=np.float32)
        min_dists = np.empty(len(points), dtype=np.float32)

        mf = cl.mem_flags
        points_buf = cl.Buffer(self._opencl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=points_flat)
        refs_buf = cl.Buffer(self._opencl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=refs_flat)
        result_buf = cl.Buffer(self._opencl_context, mf.WRITE_ONLY, min_dists.nbytes)

        program.min_distance(
            self._opencl_queue,
            (len(points),),
            None,
            points_buf,
            refs_buf,
            result_buf,
            np.int32(len(reference_points))
        )

        cl.enqueue_copy(self._opencl_queue, min_dists, result_buf)

        return min_dists

    def _compute_distances_metal(self, points: np.ndarray,
                              reference_points: np.ndarray) -> np.ndarray:
        mx = self._mlx

        pts = np.ascontiguousarray(points, dtype=np.float32)
        refs = np.ascontiguousarray(reference_points, dtype=np.float32)

        n_points = len(pts)
        chunk_size = min(self._get_distance_chunk_size(), n_points)
        min_distances = np.empty(n_points, dtype=np.float32)

        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            chunk = mx.array(pts[start:end])
            refs_mx = mx.array(refs)

            diff = chunk[:, None, :] - refs_mx[None, :, :]
            distances = mx.min(mx.sqrt(mx.sum(diff ** 2, axis=2)), axis=1)
            min_distances[start:end] = np.array(distances)

        return min_distances

    def _update_solidification_metal(self,
                                  voxel_states: np.ndarray,
                                  neighbors: np.ndarray,
                                  moduli: np.ndarray,
                                  heights: np.ndarray,
                                  chill_boost: np.ndarray,
                                  riser_slowdown: np.ndarray,
                                  part_types: np.ndarray,
                                  time_step: float,
                                  material_factor: float,
                                  feeding_zone_vertical_bias: float = 0.5) -> np.ndarray:
        import taichi as ti

        n_voxels = len(voxel_states)
        n_neighbors = neighbors.shape[1] if len(neighbors.shape) > 1 else 26

        states = np.ascontiguousarray(voxel_states, dtype=np.float32)
        neighs = np.ascontiguousarray(neighbors, dtype=np.int32)
        mods = np.ascontiguousarray(moduli, dtype=np.float32)
        hts = np.ascontiguousarray(heights, dtype=np.float32)
        chills = np.ascontiguousarray(chill_boost, dtype=np.float32)
        risers = np.ascontiguousarray(riser_slowdown, dtype=np.float32)
        parts = np.ascontiguousarray(part_types, dtype=np.int32)

        new_states = states.copy()

        @ti.kernel
        def _kernel(
            states: ti.types.ndarray(dtype=ti.f32, ndim=1),
            neighs: ti.types.ndarray(dtype=ti.i32, ndim=2),
            mods: ti.types.ndarray(dtype=ti.f32, ndim=1),
            hts: ti.types.ndarray(dtype=ti.f32, ndim=1),
            chills: ti.types.ndarray(dtype=ti.f32, ndim=1),
            risers: ti.types.ndarray(dtype=ti.f32, ndim=1),
            parts: ti.types.ndarray(dtype=ti.i32, ndim=1),
            new_states: ti.types.ndarray(dtype=ti.f32, ndim=1),
            dt: ti.f32,
            mat_factor: ti.f32,
            fz_bias: ti.f32,
        ):
            feeding_zone_threshold = 0.6
            CAST = 0
            RISER = 1
            CHILL = 2

            for i in range(n_voxels):
                if states[i] >= 1.0:
                    new_states[i] = states[i]
                    continue

                is_riser_voxel = parts[i] == RISER
                has_liquid_riser_neighbor = False
                has_liquid_feeding_neighbor = False
                has_solid_neighbor = False

                for j in range(n_neighbors):
                    n_idx = neighs[i, j]
                    if n_idx >= 0:
                        neighbor_state = states[n_idx]
                        neighbor_part_type = parts[n_idx]
                        neighbor_slowdown = risers[n_idx]

                        if neighbor_state >= 1.0:
                            has_solid_neighbor = True
                        else:
                            if neighbor_part_type == RISER:
                                has_liquid_riser_neighbor = True
                            elif neighbor_slowdown < feeding_zone_threshold:
                                has_liquid_feeding_neighbor = True

                in_feeding_zone = (
                    is_riser_voxel or
                    has_liquid_riser_neighbor or
                    (risers[i] < feeding_zone_threshold and has_liquid_feeding_neighbor)
                )

                effective_slowdown = risers[i]
                if is_riser_voxel and in_feeding_zone:
                    effective_slowdown = risers[i]
                elif has_liquid_riser_neighbor and in_feeding_zone:
                    effective_slowdown = risers[i] * 0.4
                elif in_feeding_zone:
                    effective_slowdown = risers[i] * 0.6

                if has_solid_neighbor:
                    base_rate = 0.02
                    solidification_rate = base_rate / (1.0 + mods[i] * mods[i] * 2.0)

                    vertical_factor = 1.0 + (1.0 - hts[i]) * 0.3
                    if in_feeding_zone:
                        vertical_factor = 1.0 + (1.0 - hts[i]) * fz_bias

                    thermal_factor = mat_factor
                    part_type_factor = chills[i] * effective_slowdown

                    delta = (solidification_rate * vertical_factor * thermal_factor *
                             part_type_factor * dt)
                    new_states[i] = min(states[i] + delta, 1.0)
                else:
                    new_states[i] = states[i]

        _kernel(states, neighs, mods, hts, chills, risers, parts, new_states,
                float(time_step), float(material_factor), float(feeding_zone_vertical_bias))

        return np.array(new_states)

    def compute_voxel_containment_gpu(self,
                                      voxel_centers: np.ndarray,
                                      mesh_vertices: np.ndarray,
                                      mesh_faces: np.ndarray) -> np.ndarray:
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA:
            return self._voxel_containment_cuda(voxel_centers, mesh_vertices, mesh_faces)
        elif self.backend == GPUBackend.METAL:
            return self._voxel_containment_metal(voxel_centers, mesh_vertices, mesh_faces)
        else:
            return None

    def _voxel_containment_cuda(self, voxel_centers: np.ndarray,
                                mesh_vertices: np.ndarray,
                                mesh_faces: np.ndarray) -> np.ndarray:
        cp = self._cupy

        centers_gpu = cp.asarray(voxel_centers, dtype=cp.float32)
        vertices_gpu = cp.asarray(mesh_vertices, dtype=cp.float32)

        bbox_min = cp.min(vertices_gpu, axis=0)
        bbox_max = cp.max(vertices_gpu, axis=0)

        in_bbox = cp.all(
            (centers_gpu >= bbox_min) & (centers_gpu <= bbox_max),
            axis=1
        )

        return cp.asnumpy(in_bbox)

    def _voxel_containment_metal(self,
                              voxel_centers: np.ndarray,
                              mesh_vertices: np.ndarray,
                              mesh_faces: np.ndarray) -> np.ndarray:
        import taichi as ti
        import taichi.math as tm

        n_voxels = len(voxel_centers)
        n_faces = len(mesh_faces)

        verts = np.ascontiguousarray(mesh_vertices, dtype=np.float32)
        faces = np.ascontiguousarray(mesh_faces, dtype=np.int32)
        centers = np.ascontiguousarray(voxel_centers, dtype=np.float32)

        inside = np.zeros(n_voxels, dtype=np.int32)

        @ti.kernel
        def _kernel(
            centers: ti.types.ndarray(dtype=ti.f32, ndim=2),
            verts: ti.types.ndarray(dtype=ti.f32, ndim=2),
            faces: ti.types.ndarray(dtype=ti.i32, ndim=2),
            inside: ti.types.ndarray(dtype=ti.i32, ndim=1),
        ):
            EPSILON = 1e-7
            for i in range(n_voxels):
                origin = ti.Vector([centers[i, 0], centers[i, 1], centers[i, 2]])
                ray_dir = ti.Vector([1.0, 0.0, 0.0])

                crossings = 0

                for f in range(n_faces):
                    v0_idx = faces[f, 0]
                    v1_idx = faces[f, 1]
                    v2_idx = faces[f, 2]
                    v0 = ti.Vector([verts[v0_idx, 0], verts[v0_idx, 1], verts[v0_idx, 2]])
                    v1 = ti.Vector([verts[v1_idx, 0], verts[v1_idx, 1], verts[v1_idx, 2]])
                    v2 = ti.Vector([verts[v2_idx, 0], verts[v2_idx, 1], verts[v2_idx, 2]])

                    e1 = v1 - v0
                    e2 = v2 - v0

                    pvec = tm.cross(ray_dir, e2)
                    det = tm.dot(e1, pvec)

                    if det > -EPSILON and det < EPSILON:
                        continue

                    inv_det = 1.0 / det
                    tvec = origin - v0
                    u = tm.dot(tvec, pvec) * inv_det

                    if u < 0.0 or u > 1.0:
                        continue

                    qvec = tm.cross(tvec, e1)
                    v = tm.dot(ray_dir, qvec) * inv_det

                    if v < 0.0 or (u + v) > 1.0:
                        continue

                    t = tm.dot(e2, qvec) * inv_det

                    if t > 0.0:
                        crossings += 1

                inside[i] = crossings % 2

        _kernel(centers, verts, faces, inside)

        min_dist_check = np.min(np.linalg.norm(centers[:, None, :] - verts[None, :, :], axis=2), axis=1)
        if np.any(min_dist_check < 1e-4):
            return None

        return inside.astype(bool)

    def update_solidification_gpu(self,
                                  voxel_states: np.ndarray,
                                  neighbors: np.ndarray,
                                  moduli: np.ndarray,
                                  heights: np.ndarray,
                                  chill_boost: np.ndarray,
                                  riser_slowdown: np.ndarray,
                                  part_types: np.ndarray,
                                  time_step: float,
                                  material_factor: float,
                                  feeding_zone_vertical_bias: float = 0.5) -> np.ndarray:
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA:
            return self._update_solidification_cuda(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types, time_step, material_factor,
                feeding_zone_vertical_bias,
            )
        elif self.backend == GPUBackend.METAL:
            return self._update_solidification_metal(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types, time_step, material_factor,
                feeding_zone_vertical_bias,
            )
        elif self.backend == GPUBackend.OPENCL:
            return self._update_solidification_opencl(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types, time_step, material_factor,
                feeding_zone_vertical_bias,
            )
        else:
            from core.acceleration import update_solidification_with_part_types_parallel
            return update_solidification_with_part_types_parallel(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, time_step, material_factor
            )

    def _update_solidification_opencl(self,
                                      voxel_states: np.ndarray,
                                      neighbors: np.ndarray,
                                      moduli: np.ndarray,
                                      heights: np.ndarray,
                                      chill_boost: np.ndarray,
                                      riser_slowdown: np.ndarray,
                                      part_types: np.ndarray,
                                      time_step: float,
                                      material_factor: float,
                                      feeding_zone_vertical_bias: float = 0.5) -> np.ndarray:
        import pyopencl as cl

        fz_bias_lit = f"{float(feeding_zone_vertical_bias):.6f}f"

        kernel_source = """
        #define CAST 0
        #define RISER 1
        #define CHILL 2

        __kernel void solidification_update(
            __global const float* states,
            __global const int* neighbors,
            __global const float* moduli,
            __global const float* heights,
            __global const float* chill_boost,
            __global const float* riser_slowdown,
            __global const int* part_types,
            __global float* new_states,
            const int n_neighbors,
            const float time_step,
            const float material_factor
        ) {
            int i = get_global_id(0);

            float state = states[i];
            if (state >= 1.0f) {
                new_states[i] = state;
                return;
            }

            float feeding_zone_threshold = 0.6f;

            int is_riser_voxel = part_types[i] == RISER;
            int has_liquid_riser_neighbor = 0;
            int has_liquid_feeding_neighbor = 0;
            int has_solid_neighbor = 0;

            for (int j = 0; j < n_neighbors; j++) {
                int neighbor_idx = neighbors[i * n_neighbors + j];
                if (neighbor_idx >= 0) {
                    float neighbor_state = states[neighbor_idx];
                    int neighbor_part_type = part_types[neighbor_idx];
                    float neighbor_slowdown = riser_slowdown[neighbor_idx];

                    if (neighbor_state >= 1.0f) {
                        has_solid_neighbor = 1;
                    } else {
                        if (neighbor_part_type == RISER) {
                            has_liquid_riser_neighbor = 1;
                        } else if (neighbor_slowdown < feeding_zone_threshold) {
                            has_liquid_feeding_neighbor = 1;
                        }
                    }
                }
            }

            int in_feeding_zone = is_riser_voxel || has_liquid_riser_neighbor ||
                (riser_slowdown[i] < feeding_zone_threshold && has_liquid_feeding_neighbor);

            float effective_slowdown;
            if (in_feeding_zone) {
                if (is_riser_voxel) {
                    effective_slowdown = riser_slowdown[i];
                } else if (has_liquid_riser_neighbor) {
                    effective_slowdown = riser_slowdown[i] * 0.4f;
                } else {
                    effective_slowdown = riser_slowdown[i] * 0.6f;
                }
            } else {
                effective_slowdown = riser_slowdown[i];
            }

            if (has_solid_neighbor) {
                float base_rate = 0.02f;
                float solidification_rate = base_rate / (1.0f + moduli[i] * moduli[i] * 2.0f);

                float vertical_factor;
                if (in_feeding_zone) {
                    vertical_factor = 1.0f + (1.0f - heights[i]) * __FZ_BIAS__;
                } else {
                    vertical_factor = 1.0f + (1.0f - heights[i]) * 0.3f;
                }

                float thermal_factor = material_factor;
                float part_type_factor = chill_boost[i] * effective_slowdown;

                float delta = solidification_rate * vertical_factor * thermal_factor *
                             part_type_factor * time_step;
                new_states[i] = fmin(state + delta, 1.0f);
            } else {
                new_states[i] = state;
            }
        }
        """.replace("__FZ_BIAS__", fz_bias_lit)

        states = np.ascontiguousarray(voxel_states, dtype=np.float32)
        neighs = np.ascontiguousarray(neighbors, dtype=np.int32)
        mods = np.ascontiguousarray(moduli, dtype=np.float32)
        hts = np.ascontiguousarray(heights, dtype=np.float32)
        chill = np.ascontiguousarray(chill_boost, dtype=np.float32)
        riser = np.ascontiguousarray(riser_slowdown, dtype=np.float32)
        parts = np.ascontiguousarray(part_types, dtype=np.int32)

        n_voxels = len(states)
        n_neighbors = neighs.shape[1] if len(neighs.shape) > 1 else 26

        program = cl.Program(self._opencl_context, kernel_source).build()

        states_buf = cl.Buffer(self._opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=states)
        neighbors_buf = cl.Buffer(self._opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=neighs)
        moduli_buf = cl.Buffer(self._opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=mods)
        heights_buf = cl.Buffer(self._opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hts)
        chill_buf = cl.Buffer(self._opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=chill)
        riser_buf = cl.Buffer(self._opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=riser)
        parts_buf = cl.Buffer(self._opencl_context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=parts)

        new_states = np.empty_like(states)
        new_states_buf = cl.Buffer(self._opencl_context, cl.mem_flags.WRITE_ONLY, new_states.nbytes)

        program.solidification_update(
            self._opencl_queue,
            (n_voxels,),
            None,
            states_buf, neighbors_buf, moduli_buf, heights_buf,
            chill_buf, riser_buf, parts_buf, new_states_buf,
            np.int32(n_neighbors),
            np.float32(time_step),
            np.float32(material_factor)
        )

        cl.enqueue_copy(self._opencl_queue, new_states, new_states_buf)

        return new_states

    def _update_solidification_cuda(self,
                                    voxel_states: np.ndarray,
                                    neighbors: np.ndarray,
                                    moduli: np.ndarray,
                                    heights: np.ndarray,
                                    chill_boost: np.ndarray,
                                    riser_slowdown: np.ndarray,
                                    part_types: np.ndarray,
                                    time_step: float,
                                    material_factor: float,
                                    feeding_zone_vertical_bias: float = 0.5) -> np.ndarray:
        cp = self._cupy

        states_gpu = cp.asarray(voxel_states, dtype=cp.float32)
        neighbors_gpu = cp.asarray(neighbors, dtype=cp.int32)
        moduli_gpu = cp.asarray(moduli, dtype=cp.float32)
        heights_gpu = cp.asarray(heights, dtype=cp.float32)
        chill_gpu = cp.asarray(chill_boost, dtype=cp.float32)
        riser_gpu = cp.asarray(riser_slowdown, dtype=cp.float32)
        parts_gpu = cp.asarray(part_types, dtype=cp.int32)

        new_states = states_gpu.copy()
        n_voxels = len(states_gpu)

        fz_bias_lit = f"{float(feeding_zone_vertical_bias):.6f}f"

        solidification_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void solidification_update(
            const float* states,
            const int* neighbors,
            const float* moduli,
            const float* heights,
            const float* chill_boost,
            const float* riser_slowdown,
            const int* part_types,
            float* new_states,
            int n_voxels,
            int n_neighbors,
            float time_step,
            float material_factor
        ) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n_voxels) return;

            #define CAST 0
            #define RISER 1
            #define CHILL 2

            if (states[i] >= 1.0f) {
                new_states[i] = states[i];
                return;
            }

            float feeding_zone_threshold = 0.6f;

            int is_riser_voxel = part_types[i] == RISER;
            int has_liquid_riser_neighbor = 0;
            int has_liquid_feeding_neighbor = 0;
            int has_solid_neighbor = 0;

            for (int j = 0; j < n_neighbors; j++) {
                int neighbor_idx = neighbors[i * n_neighbors + j];
                if (neighbor_idx >= 0) {
                    float neighbor_state = states[neighbor_idx];
                    int neighbor_part_type = part_types[neighbor_idx];
                    float neighbor_slowdown = riser_slowdown[neighbor_idx];

                    if (neighbor_state >= 1.0f) {
                        has_solid_neighbor = 1;
                    } else {
                        if (neighbor_part_type == RISER) {
                            has_liquid_riser_neighbor = 1;
                        } else if (neighbor_slowdown < feeding_zone_threshold) {
                            has_liquid_feeding_neighbor = 1;
                        }
                    }
                }
            }

            int in_feeding_zone = is_riser_voxel || has_liquid_riser_neighbor ||
                (riser_slowdown[i] < feeding_zone_threshold && has_liquid_feeding_neighbor);

            float effective_slowdown;
            if (in_feeding_zone) {
                if (is_riser_voxel) {
                    effective_slowdown = riser_slowdown[i];
                } else if (has_liquid_riser_neighbor) {
                    effective_slowdown = riser_slowdown[i] * 0.4f;
                } else {
                    effective_slowdown = riser_slowdown[i] * 0.6f;
                }
            } else {
                effective_slowdown = riser_slowdown[i];
            }

            if (has_solid_neighbor) {
                float base_rate = 0.02f;
                float solidification_rate = base_rate / (1.0f + moduli[i] * moduli[i] * 2.0f);

                float vertical_factor;
                if (in_feeding_zone) {
                    vertical_factor = 1.0f + (1.0f - heights[i]) * __FZ_BIAS__;
                } else {
                    vertical_factor = 1.0f + (1.0f - heights[i]) * 0.3f;
                }

                float thermal_factor = material_factor;
                float part_type_factor = chill_boost[i] * effective_slowdown;

                float delta = solidification_rate * vertical_factor * thermal_factor *
                             part_type_factor * time_step;
                new_states[i] = fminf(states[i] + delta, 1.0f);
            } else {
                new_states[i] = states[i];
            }
        }
        '''.replace("__FZ_BIAS__", fz_bias_lit), 'solidification_update')

        block_size = 256
        grid_size = (n_voxels + block_size - 1) // block_size

        solidification_kernel(
            (grid_size,), (block_size,),
            (states_gpu, neighbors_gpu, moduli_gpu, heights_gpu,
             chill_gpu, riser_gpu, parts_gpu, new_states,
             n_voxels, neighbors.shape[1],
             np.float32(time_step), np.float32(material_factor))
        )

        return cp.asnumpy(new_states)

    def parallel_reduce_sum(self, arr: np.ndarray) -> float:
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return float(self._cupy.sum(self._cupy.asarray(arr)))
        return float(np.sum(arr))

    def parallel_reduce_max(self, arr: np.ndarray) -> float:
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return float(self._cupy.max(self._cupy.asarray(arr)))
        return float(np.max(arr))

    def parallel_reduce_min(self, arr: np.ndarray) -> float:
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return float(self._cupy.min(self._cupy.asarray(arr)))
        return float(np.min(arr))

    def cleanup(self):
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            mempool = self._cupy.get_default_memory_pool()
            mempool.free_all_blocks()

        if self._opencl_queue is not None:
            self._opencl_queue.finish()

    def get_stats(self) -> dict:
        self._ensure_initialized()
        stats = {
            'backend': self.backend.value,
            'device_name': self.device_name,
            'device_memory_gb': self.device_memory_gb,
            'gpu_available': self.is_gpu_available,
        }

        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            mempool = self._cupy.get_default_memory_pool()
            stats['gpu_memory_used_mb'] = mempool.used_bytes() / (1024**2)
            stats['gpu_memory_total_mb'] = mempool.total_bytes() / (1024**2)

        return stats

    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "="*60)
        print("GPU Compute Statistics")
        print("="*60)
        print(f"Backend: {stats['backend'].upper()}")
        print(f"Device: {stats['device_name']}")
        if stats['device_memory_gb'] > 0:
            print(f"GPU Memory: {stats['device_memory_gb']:.1f} GB")
        if 'gpu_memory_used_mb' in stats:
            print(f"Memory Used: {stats['gpu_memory_used_mb']:.1f} MB")
        print("="*60 + "\n")

gpu_compute = GPUComputeManager()

def get_gpu_compute() -> GPUComputeManager:
    return gpu_compute

def compute_distances_accelerated(points: np.ndarray,
                                  reference_points: np.ndarray) -> np.ndarray:
    return gpu_compute.compute_distances_gpu(points, reference_points)

def update_solidification_accelerated(
    voxel_states: np.ndarray,
    neighbors: np.ndarray,
    moduli: np.ndarray,
    heights: np.ndarray,
    chill_boost: np.ndarray,
    riser_slowdown: np.ndarray,
    part_types: np.ndarray,
    time_step: float,
    material_factor: float,
    feeding_zone_vertical_bias: float = 0.5,
) -> np.ndarray:
    return gpu_compute.update_solidification_gpu(
        voxel_states, neighbors, moduli, heights,
        chill_boost, riser_slowdown, part_types, time_step, material_factor,
        feeding_zone_vertical_bias,
    )
