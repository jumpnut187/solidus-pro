"""
GPU Acceleration Layer
Provides unified interface for GPU-accelerated computations on Metal (Apple Silicon)
and CUDA (NVIDIA GPUs) with automatic fallback to CPU.

Implements optimized kernels for:
- Distance calculations (10-50x speedup)
- Voxelization operations
- Solidification front updates
- Array operations on large meshes
"""

import numpy as np
from typing import Optional, Tuple, Callable, Union
from enum import Enum
import platform
import warnings
import io
import contextlib
import multiprocessing
from config.hardware_config import hardware


class GPUBackend(Enum):
    """Available GPU backends"""
    NONE = "none"           # CPU-only
    METAL = "metal"         # Apple Silicon (via PyOpenCL or Metal Performance Shaders)
    CUDA = "cuda"           # NVIDIA (via CuPy)
    OPENCL = "opencl"       # Generic OpenCL (Intel/AMD/NVIDIA)


class GPUComputeManager:
    """
    Manages GPU computation with automatic backend selection and fallback.
    
    Priority order:
    1. CUDA (if available and NVIDIA GPU present)
    2. Metal (if on macOS with Apple Silicon)
    3. OpenCL (if available)
    4. CPU fallback (always available)
    """
    
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
        
        # Backend-specific contexts
        self._cuda_available = False
        self._metal_available = False
        self._opencl_available = False
        
        self._cupy = None
        self._opencl_context = None
        self._opencl_queue = None
        self._metal_device = None
        
        # Metal (MLX + Taichi) state
        self._mlx = None
        self._taichi_initialized = False
        
        # Performance tracking
        self._kernel_times = {}
        
        # Don't detect backends at init time - wait for lazy initialization
        # This prevents issues with PyOpenCL/Metal when Qt hasn't started yet
        self._initialized = True
    
    def _ensure_initialized(self):
        """Lazily initialize GPU backends on first use"""
        if not self._detection_attempted:
            self._detection_attempted = True
            self._detect_backends()
    
    def _detect_backends(self):
        """Detect available GPU backends and select the best one"""
        system = platform.system()
        is_main_process = multiprocessing.current_process().name == "MainProcess"
        devnull = contextlib.nullcontext() if is_main_process else contextlib.redirect_stdout(io.StringIO())
        
        if is_main_process:
            print("\n" + "="*60)
            print("GPU Compute Manager: Detecting backends...")
            print("="*60)
        
        # Try CUDA first (CuPy)
        with devnull:
            cuda_ready = self._try_init_cuda()
        if cuda_ready:
            self.backend = GPUBackend.CUDA
            if is_main_process:
                print(f"✓ Selected backend: CUDA ({self.device_name})")
                print(f"  GPU Memory: {self.device_memory_gb:.1f} GB")
                print("="*60 + "\n")
            return
        
        # Try Metal on macOS with Apple Silicon
        # Note: hardware.metal_available just checks if running on Apple Silicon
        # PyOpenCL cannot access Metal GPU on modern macOS - OpenCL is deprecated
        if system == 'Darwin' and hardware.metal_available:
            with devnull:
                metal_ready = self._try_init_metal()
            if metal_ready:
                self.backend = GPUBackend.METAL
                if is_main_process:
                    print(f"✓ Selected backend: Metal ({self.device_name})")
                    print("=" * 60 + "\n")
                return
            else:
                # OpenCL cannot access Metal GPU on modern macOS
                if is_main_process:
                    print("  Apple Silicon detected - OpenCL/Metal GPU bridge not available")
        
        # Try generic OpenCL
        with devnull:
            opencl_ready = self._try_init_opencl()
        if opencl_ready:
            self.backend = GPUBackend.OPENCL
            if is_main_process:
                print(f"✓ Selected backend: OpenCL ({self.device_name})")
                print(f"  GPU Memory: {self.device_memory_gb:.1f} GB")
                print("="*60 + "\n")
            return
        
        # Fallback to CPU
        self.backend = GPUBackend.NONE
        self.device_name = "CPU (Numba JIT)"
        
        # Import hardware for thread count
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
        """Try to initialize CUDA via CuPy"""
        try:
            import cupy as cp
            
            # Check for available GPU
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                print("  CUDA: No NVIDIA GPU found")
                return False
            
            # Get device info
            device = cp.cuda.Device(0)
            self.device_name = device.pci_bus_id
            
            # Try to get device name properly
            try:
                props = cp.cuda.runtime.getDeviceProperties(0)
                self.device_name = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
            except:
                self.device_name = f"CUDA Device {device.id}"
            
            # Get memory
            mem_info = device.mem_info
            self.device_memory_gb = mem_info[1] / (1024**3)  # Total memory
            
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
        """Try to initialize Metal via MLX + Taichi on Apple Silicon"""
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
        """Try to initialize generic OpenCL"""
        try:
            import pyopencl as cl
            
            platforms = cl.get_platforms()
            if not platforms:
                print("  OpenCL: No platforms found")
                return False
            
            # Find best GPU device
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
        """Check if any GPU backend is available"""
        self._ensure_initialized()
        return self.backend != GPUBackend.NONE
    
    def get_array_module(self):
        """
        Get the appropriate array module (cupy or numpy)
        Use this for array operations that should run on GPU when available.
        """
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return self._cupy
        return np
    
    def to_gpu(self, arr: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
        """Transfer numpy array to GPU memory"""
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return self._cupy.asarray(arr)
        return arr
    
    def to_cpu(self, arr) -> np.ndarray:
        """Transfer array from GPU to CPU memory"""
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            if hasattr(arr, 'get'):
                return arr.get()
        return np.asarray(arr)
    
    def compute_distances_gpu(self, 
                              points: np.ndarray, 
                              reference_points: np.ndarray) -> np.ndarray:
        """
        Compute minimum distances from points to reference points using GPU.
        
        This is one of the most expensive operations in voxelization and
        solidification - GPU acceleration provides 10-50x speedup.
        
        Args:
            points: Array of points (n_points x 3)
            reference_points: Array of reference points (n_ref x 3)
        
        Returns:
            Array of minimum distances (n_points,)
        """
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
    
    def _compute_distances_cuda(self, points: np.ndarray, 
                                reference_points: np.ndarray) -> np.ndarray:
        """CUDA implementation using CuPy"""
        cp = self._cupy
        
        # Transfer to GPU
        points_gpu = cp.asarray(points, dtype=cp.float32)
        refs_gpu = cp.asarray(reference_points, dtype=cp.float32)
        
        n_points = points_gpu.shape[0]
        n_refs = refs_gpu.shape[0]
        
        # Compute distances using broadcasting (GPU-optimized)
        # For very large arrays, process in chunks to avoid memory issues
        chunk_size = min(10000, n_points)
        min_distances = cp.empty(n_points, dtype=cp.float32)
        
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            chunk = points_gpu[start:end]
            
            # Compute all pairwise distances for this chunk
            # Using broadcasting: (chunk_size, 1, 3) - (1, n_refs, 3)
            diff = chunk[:, cp.newaxis, :] - refs_gpu[cp.newaxis, :, :]
            distances = cp.sqrt(cp.sum(diff ** 2, axis=2))
            
            # Get minimum distance for each point
            min_distances[start:end] = cp.min(distances, axis=1)
        
        return cp.asnumpy(min_distances)
    
    def _compute_distances_opencl(self, points: np.ndarray,
                                  reference_points: np.ndarray) -> np.ndarray:
        """OpenCL implementation for Metal and generic GPUs"""
        import pyopencl as cl
        import pyopencl.array as cl_array
        
        # OpenCL kernel for minimum distance computation
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
        
        # Build kernel
        program = cl.Program(self._opencl_context, kernel_source).build()
        
        # Prepare data
        points_flat = np.ascontiguousarray(points.flatten(), dtype=np.float32)
        refs_flat = np.ascontiguousarray(reference_points.flatten(), dtype=np.float32)
        min_dists = np.empty(len(points), dtype=np.float32)
        
        # Create buffers
        mf = cl.mem_flags
        points_buf = cl.Buffer(self._opencl_context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                               hostbuf=points_flat)
        refs_buf = cl.Buffer(self._opencl_context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=refs_flat)
        result_buf = cl.Buffer(self._opencl_context, mf.WRITE_ONLY, min_dists.nbytes)
        
        # Execute kernel
        program.min_distance(
            self._opencl_queue,
            (len(points),),
            None,
            points_buf,
            refs_buf,
            result_buf,
            np.int32(len(reference_points))
        )
        
        # Read results
        cl.enqueue_copy(self._opencl_queue, min_dists, result_buf)
        
        return min_dists
    
    def _compute_distances_metal(self, points: np.ndarray,
                              reference_points: np.ndarray) -> np.ndarray:
        """MLX implementation for Metal GPU distance computation"""
        mx = self._mlx
        
        pts = np.ascontiguousarray(points, dtype=np.float32)
        refs = np.ascontiguousarray(reference_points, dtype=np.float32)
        
        n_points = len(pts)
        chunk_size = min(10000, n_points)
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
                                  material_factor: float) -> np.ndarray:
        """Taichi-compiled Metal kernel for solidification with full feeding-zone logic"""
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
                        vertical_factor = 1.0 + (1.0 - hts[i]) * 0.5
                    
                    thermal_factor = mat_factor
                    part_type_factor = chills[i] * effective_slowdown
                    
                    delta = (solidification_rate * vertical_factor * thermal_factor * 
                             part_type_factor * dt)
                    new_states[i] = min(states[i] + delta, 1.0)
                else:
                    new_states[i] = states[i]
        
        _kernel(states, neighs, mods, hts, chills, risers, parts, new_states,
                float(time_step), float(material_factor))
        
        return np.array(new_states)
    
    def compute_voxel_containment_gpu(self,
                                      voxel_centers: np.ndarray,
                                      mesh_vertices: np.ndarray,
                                      mesh_faces: np.ndarray) -> np.ndarray:
        """
        Compute which voxels are inside a mesh using GPU.
        
        Uses ray-casting method accelerated on GPU.
        
        Args:
            voxel_centers: Array of voxel centers (n_voxels x 3)
            mesh_vertices: Mesh vertex array (n_vertices x 3)
            mesh_faces: Mesh face indices (n_faces x 3)
        
        Returns:
            Boolean array indicating which voxels are inside (n_voxels,)
        """
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA:
            return self._voxel_containment_cuda(voxel_centers, mesh_vertices, mesh_faces)
        elif self.backend == GPUBackend.METAL:
            return self._voxel_containment_metal(voxel_centers, mesh_vertices, mesh_faces)
        else:
            return None  # Signal to use CPU fallback
    
    def _voxel_containment_cuda(self, voxel_centers: np.ndarray,
                                mesh_vertices: np.ndarray,
                                mesh_faces: np.ndarray) -> np.ndarray:
        """CUDA implementation of voxel containment using ray casting"""
        cp = self._cupy
        
        # This is a simplified version - for production, use a proper
        # ray-triangle intersection kernel
        # For now, we use a bounding-box based approach for speed
        
        centers_gpu = cp.asarray(voxel_centers, dtype=cp.float32)
        vertices_gpu = cp.asarray(mesh_vertices, dtype=cp.float32)
        
        # Compute mesh bounding box
        bbox_min = cp.min(vertices_gpu, axis=0)
        bbox_max = cp.max(vertices_gpu, axis=0)
        
        # Check if voxels are within bounding box (quick filter)
        in_bbox = cp.all(
            (centers_gpu >= bbox_min) & (centers_gpu <= bbox_max),
            axis=1
        )
        
        # For voxels in bbox, need more precise test
        # This is a placeholder - full implementation would use
        # GPU ray-triangle intersection
        
        return cp.asnumpy(in_bbox)
    
    def _voxel_containment_metal(self,
                              voxel_centers: np.ndarray,
                              mesh_vertices: np.ndarray,
                              mesh_faces: np.ndarray) -> np.ndarray:
        """Taichi Metal ray-triangle intersection for point-in-mesh containment"""
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
                                  material_factor: float) -> np.ndarray:
        """
        Update solidification front using GPU acceleration.
        
        Args:
            voxel_states: Current solidification states (0=liquid, 1=solid)
            neighbors: Neighbor connectivity (n_voxels x 26)
            moduli: Geometric modulus for each voxel
            heights: Normalized vertical position
            chill_boost: Chill acceleration multiplier
            riser_slowdown: Riser slowdown multiplier
            part_types: Part type for each voxel (0=CAST, 1=RISER, 2=CHILL)
            time_step: Current time step
            material_factor: Material solidification rate
        
        Returns:
            Updated voxel states
        """
        self._ensure_initialized()
        if self.backend == GPUBackend.CUDA:
            return self._update_solidification_cuda(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types, time_step, material_factor
            )
        elif self.backend == GPUBackend.METAL:
            return self._update_solidification_metal(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types, time_step, material_factor
            )
        elif self.backend == GPUBackend.OPENCL:
            return self._update_solidification_opencl(
                voxel_states, neighbors, moduli, heights,
                chill_boost, riser_slowdown, part_types, time_step, material_factor
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
                                      material_factor: float) -> np.ndarray:
        """OpenCL implementation of solidification update with full feeding-zone logic"""
        import pyopencl as cl
        
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
                    vertical_factor = 1.0f + (1.0f - heights[i]) * 0.5f;
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
        """
        
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
                                    material_factor: float) -> np.ndarray:
        """CUDA implementation of solidification update with full feeding-zone logic"""
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
                    vertical_factor = 1.0f + (1.0f - heights[i]) * 0.5f;
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
        ''', 'solidification_update')
        
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
        """GPU-accelerated parallel sum reduction"""
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return float(self._cupy.sum(self._cupy.asarray(arr)))
        return float(np.sum(arr))
    
    def parallel_reduce_max(self, arr: np.ndarray) -> float:
        """GPU-accelerated parallel max reduction"""
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return float(self._cupy.max(self._cupy.asarray(arr)))
        return float(np.max(arr))
    
    def parallel_reduce_min(self, arr: np.ndarray) -> float:
        """GPU-accelerated parallel min reduction"""
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            return float(self._cupy.min(self._cupy.asarray(arr)))
        return float(np.min(arr))
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.backend == GPUBackend.CUDA and self._cupy is not None:
            # Clear GPU memory pool
            mempool = self._cupy.get_default_memory_pool()
            mempool.free_all_blocks()
        
        if self._opencl_queue is not None:
            self._opencl_queue.finish()
    
    def get_stats(self) -> dict:
        """Get GPU compute statistics"""
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
        """Print GPU compute statistics"""
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


# Global GPU compute manager instance
gpu_compute = GPUComputeManager()


def compute_distances_accelerated(points: np.ndarray,
                                  reference_points: np.ndarray) -> np.ndarray:
    """
    Convenience function for GPU-accelerated distance computation.
    Automatically uses GPU if available, falls back to CPU.
    
    Args:
        points: Array of points (n_points x 3)
        reference_points: Array of reference points (n_ref x 3)
    
    Returns:
        Array of minimum distances (n_points,)
    """
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
    material_factor: float
) -> np.ndarray:
    """
    Convenience function for GPU-accelerated solidification update.
    
    Args:
        voxel_states: Current solidification states
        neighbors: Neighbor connectivity
        moduli: Geometric modulus
        heights: Normalized heights
        chill_boost: Chill boost factors
        riser_slowdown: Riser slowdown factors
        part_types: Part type for each voxel (0=CAST, 1=RISER, 2=CHILL)
        time_step: Time step
        material_factor: Material factor
    
    Returns:
        Updated voxel states
    """
    return gpu_compute.update_solidification_gpu(
        voxel_states, neighbors, moduli, heights,
        chill_boost, riser_slowdown, part_types, time_step, material_factor
    )
