/**
 * Casting Solidification Simulator - Web Application
 * Three.js based 3D visualization with Flask backend
 */

// ============================================
// Global State
// ============================================

const state = {
    // Simulation
    simulationId: null,
    parts: [],
    selectedPartId: null,
    
    // Materials
    materials: [],
    
    // Three.js
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    
    // Meshes
    partMeshes: {},
    resultMeshes: {},
    gridHelper: null,
    axesHelper: null,
    
    // Settings
    showWireframe: false,
    showAxes: true,
    showGrid: true,
    
    // Results
    hasResults: false,
    resultData: null,
    
    // Simulation state
    isSimulating: false,
    
    // Measurement
    measureMode: false,
    measurePoints: [],
    measureLine: null,
    
    // Cross Section
    crossSectionEnabled: false,
    clippingPlane: null,
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    loadMaterials();
    updateMaterialInfo();
    updateSandRestraintLabel();
    toggleAutoMesh();  // Initialize auto mesh state
    initDragAndDrop();
    initCleanupHandlers();
    animate();
});

// Cleanup running simulations when browser closes or refreshes
function initCleanupHandlers() {
    // Use sendBeacon for reliable cleanup on page unload
    window.addEventListener('beforeunload', (event) => {
        if (state.simulationId && state.isSimulating) {
            // Use sendBeacon for reliable delivery during unload
            navigator.sendBeacon('/api/simulations/cleanup', JSON.stringify({}));
        }
    });
    
    // Also handle visibility change (tab switch, minimize)
    document.addEventListener('visibilitychange', () => {
        // Don't cancel on visibility change - only on actual close
    });
}

async function cancelSimulation() {
    if (!state.simulationId) return;
    
    try {
        await fetch(`/api/simulations/${state.simulationId}/cancel`, {
            method: 'POST'
        });
        state.isSimulating = false;
        setStatus('Simulation cancelled');
    } catch (error) {
        console.error('Failed to cancel simulation:', error);
    }
}

function initDragAndDrop() {
    const viewport = document.getElementById('viewport3d');
    
    viewport.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        viewport.classList.add('drag-over');
    });
    
    viewport.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        viewport.classList.remove('drag-over');
    });
    
    viewport.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        viewport.classList.remove('drag-over');
        
        const files = Array.from(e.dataTransfer.files).filter(f => 
            f.name.toLowerCase().endsWith('.stl')
        );
        
        if (files.length === 0) {
            setStatus('Please drop STL files');
            return;
        }
        
        // Create a fake event object for the handler
        await handleDroppedFiles(files);
    });
}

async function handleDroppedFiles(files) {
    setStatus(`Uploading ${files.length} file(s)...`);
    
    let successCount = 0;
    let errorCount = 0;
    
    for (const file of files) {
        try {
            await uploadSingleFile(file);
            successCount++;
            setStatus(`Uploaded ${successCount}/${files.length}: ${file.name}`);
        } catch (error) {
            console.error('Upload error for', file.name, error);
            errorCount++;
        }
    }
    
    // Update UI after all uploads
    updatePartsList();
    if (state.parts.length > 0) {
        selectPart(state.parts[state.parts.length - 1].id);
        fitCameraToModel();
    }
    
    // Get auto mesh recommendation if enabled
    if (document.getElementById('autoMesh').checked && state.parts.length > 0) {
        getAutoMeshRecommendation();
    }
    
    updateModelInfo();
    
    if (errorCount > 0) {
        setStatus(`Uploaded ${successCount} files, ${errorCount} failed`);
    } else {
        setStatus(`Loaded ${successCount} file(s)`);
    }
}

function initThreeJS() {
    const container = document.getElementById('viewport3d');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x0a0a12);
    
    // Camera - Z is up
    state.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
    state.camera.position.set(50, -50, 50);
    state.camera.up.set(0, 0, 1);  // Z-axis up
    
    // Renderer
    state.renderer = new THREE.WebGLRenderer({ antialias: true });
    state.renderer.setSize(width, height);
    state.renderer.setPixelRatio(window.devicePixelRatio);
    state.renderer.localClippingEnabled = true;
    container.appendChild(state.renderer.domElement);
    
    // Controls (OrbitControls) - Z is up
    state.controls = new THREE.OrbitControls(state.camera, state.renderer.domElement);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;
    state.controls.screenSpacePanning = true;
    state.controls.minDistance = 1;
    state.controls.maxDistance = 1000;
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    state.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(50, -50, 100);
    state.scene.add(directionalLight);
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-50, 50, -100);
    state.scene.add(directionalLight2);
    
    // Grid - XY plane (Z up)
    state.gridHelper = new THREE.GridHelper(100, 100, 0x444444, 0x222222);
    state.gridHelper.rotation.x = Math.PI / 2;  // Rotate grid to XY plane
    state.scene.add(state.gridHelper);
    
    // Axes - Z is up (red=X, green=Y, blue=Z)
    state.axesHelper = new THREE.AxesHelper(20);
    state.scene.add(state.axesHelper);
    
    // Clipping plane for cross-section
    state.clippingPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    
    // Resize handler
    window.addEventListener('resize', onWindowResize);
    
    // Click handler for measurement
    state.renderer.domElement.addEventListener('click', onViewportClick);
}

function animate() {
    requestAnimationFrame(animate);
    state.controls.update();
    state.renderer.render(state.scene, state.camera);
}

function onWindowResize() {
    const container = document.getElementById('viewport3d');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    state.camera.aspect = width / height;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(width, height);
}

// ============================================
// API Functions
// ============================================

async function loadMaterials() {
    try {
        const response = await fetch('/api/materials');
        state.materials = await response.json();
    } catch (error) {
        console.error('Failed to load materials:', error);
    }
}

// ============================================
// File Upload
// ============================================

function uploadParts() {
    document.getElementById('fileInput').click();
}

// Legacy single file upload
function uploadPart() {
    uploadParts();
}

async function handleMultiFileUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    
    setStatus(`Uploading ${files.length} file(s)...`);
    
    let successCount = 0;
    let errorCount = 0;
    
    for (const file of files) {
        try {
            await uploadSingleFile(file);
            successCount++;
            setStatus(`Uploaded ${successCount}/${files.length}: ${file.name}`);
        } catch (error) {
            console.error('Upload error for', file.name, error);
            errorCount++;
        }
    }
    
    // Reset file input
    event.target.value = '';
    
    // Update UI after all uploads
    updatePartsList();
    if (state.parts.length > 0) {
        selectPart(state.parts[state.parts.length - 1].id);
        fitCameraToModel();
    }
    
    // Get auto mesh recommendation if enabled
    if (document.getElementById('autoMesh').checked && state.parts.length > 0) {
        getAutoMeshRecommendation();
    }
    
    updateModelInfo();
    
    if (errorCount > 0) {
        setStatus(`Uploaded ${successCount} files, ${errorCount} failed`);
    } else {
        setStatus(`Loaded ${successCount} file(s)`);
    }
}

async function uploadSingleFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    
    if (data.error) {
        throw new Error(data.error);
    }
    
    // Auto-detect part type from filename
    const filename = file.name.toLowerCase();
    let partType = 'cast';
    
    // Check for more specific patterns first
    if (filename.includes('chill') || filename.includes('cool')) {
        partType = 'chill';
    } else if (filename.includes('riser_post') || filename.includes('riser post') || filename.includes('post') || filename.includes('neck')) {
        partType = 'riser_post';
    } else if (filename.includes('riser') || filename.includes('feeder') || filename.includes('feed') || filename.includes('pad') || filename.includes('bridge')) {
        partType = 'riser';
    }
    
    // Add part to state
    const part = {
        id: data.file_id,
        name: data.filename.replace('.stl', '').replace(/_/g, ' '),
        type: partType,
        filepath: data.filepath,
        meshData: data.mesh_data,
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        scale: 1
    };
    
    state.parts.push(part);
    
    // Add mesh to scene
    addPartMesh(part);
    
    return part;
}

// Legacy single file handler (redirects to multi)
async function handleFileUpload(event) {
    await handleMultiFileUpload(event);
}

function clearAll() {
    // Clear all parts from scene
    for (const mesh of Object.values(state.partMeshes)) {
        state.scene.remove(mesh);
    }
    state.partMeshes = {};
    state.parts = [];
    state.selectedPartId = null;
    
    // Clear results
    clearResults();
    
    // Reset simulation ID
    state.simulationId = null;
    
    // Update UI
    updatePartsList();
    document.getElementById('partPropertiesPanel').style.display = 'none';
    document.getElementById('resultsPanel').style.display = 'none';
    updateModelInfo();
    
    setStatus('Cleared - ready for new simulation');
}

function clearResults() {
    // Clear result meshes from scene and dispose GPU resources
    for (const mesh of Object.values(state.resultMeshes)) {
        state.scene.remove(mesh);
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
            if (Array.isArray(mesh.material)) {
                mesh.material.forEach(m => m.dispose());
            } else {
                mesh.material.dispose();
            }
        }
    }
    state.resultMeshes = {};
    state.resultData = null;
    state.hasResults = false;
    
    // Hide results panel
    document.getElementById('resultsPanel').style.display = 'none';
    document.getElementById('fractionLiquidControls').style.display = 'none';
}

function addPartMesh(part) {
    const meshData = part.meshData;
    
    // Create geometry from vertices and faces
    const geometry = new THREE.BufferGeometry();
    
    // Vertices
    const vertices = new Float32Array(meshData.vertices);
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    
    // Faces (indices)
    const indices = new Uint32Array(meshData.faces);
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    
    // Compute normals
    geometry.computeVertexNormals();
    
    // Get color based on part type
    const colors = {
        cast: 0xb0bec5,   // Grey-blue
        riser: 0xf59e0b,  // Orange
        chill: 0x3b82f6   // Blue
    };
    
    // Material - transparent to see defects inside
    const material = new THREE.MeshPhongMaterial({
        color: colors[part.type] || colors.cast,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
        shininess: 50,
        depthWrite: false  // Don't write to depth buffer so defects show through
    });
    
    // Create mesh
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.partId = part.id;
    mesh.renderOrder = 0;  // Render model first (but with depthWrite false)
    
    // Apply transforms
    mesh.position.set(part.position.x, part.position.y, part.position.z);
    mesh.scale.setScalar(part.scale);
    
    // Add to scene
    state.scene.add(mesh);
    state.partMeshes[part.id] = mesh;
    
    // Create wireframe
    const wireframe = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.1 })
    );
    mesh.add(wireframe);
    mesh.userData.wireframe = wireframe;
    wireframe.visible = state.showWireframe;
}

// ============================================
// Parts Management
// ============================================

function updatePartsList() {
    const container = document.getElementById('partsList');
    
    if (state.parts.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-cube"></i>
                <p>No parts loaded</p>
                <small>Upload STL files or drag & drop</small>
            </div>
        `;
        document.getElementById('partPropertiesPanel').style.display = 'none';
        return;
    }
    
    // Count parts by type
    const castCount = state.parts.filter(p => p.type === 'cast').length;
    const riserCount = state.parts.filter(p => p.type === 'riser').length;
    const chillCount = state.parts.filter(p => p.type === 'chill').length;
    
    let html = `
        <div class="parts-summary">
            <span class="part-count cast">${castCount} Cast</span>
            <span class="part-count riser">${riserCount} Riser</span>
            <span class="part-count chill">${chillCount} Chill</span>
        </div>
    `;
    
    for (const part of state.parts) {
        const selected = part.id === state.selectedPartId ? 'selected' : '';
        const icon = part.type === 'cast' ? 'cube' : part.type === 'riser' ? 'fire' : 'snowflake';
        html += `
            <div class="part-item ${selected}" onclick="selectPart('${part.id}')">
                <div class="part-item-icon ${part.type}">
                    <i class="fas fa-${icon}"></i>
                </div>
                <div class="part-item-info">
                    <div class="part-item-name">${part.name}</div>
                    <div class="part-item-type">${part.type.charAt(0).toUpperCase() + part.type.slice(1)}</div>
                </div>
            </div>
        `;
    }
    container.innerHTML = html;
}

function selectPart(partId) {
    state.selectedPartId = partId;
    const part = state.parts.find(p => p.id === partId);
    
    if (!part) return;
    
    // Update UI
    updatePartsList();
    document.getElementById('partPropertiesPanel').style.display = 'block';
    
    // Fill properties form
    document.getElementById('partName').value = part.name;
    document.getElementById('partType').value = part.type;
    document.getElementById('posX').value = part.position.x;
    document.getElementById('posY').value = part.position.y;
    document.getElementById('posZ').value = part.position.z;
    document.getElementById('partScale').value = part.scale;
    document.getElementById('scaleValue').textContent = part.scale.toFixed(1) + 'x';
    
    // Highlight mesh
    for (const [id, mesh] of Object.entries(state.partMeshes)) {
        if (id === partId) {
            mesh.material.emissive = new THREE.Color(0x333333);
        } else {
            mesh.material.emissive = new THREE.Color(0x000000);
        }
    }
}

function updateSelectedPart() {
    if (!state.selectedPartId) return;
    
    const part = state.parts.find(p => p.id === state.selectedPartId);
    if (!part) return;
    
    // Update part data
    part.name = document.getElementById('partName').value;
    part.type = document.getElementById('partType').value;
    part.position.x = parseFloat(document.getElementById('posX').value) || 0;
    part.position.y = parseFloat(document.getElementById('posY').value) || 0;
    part.position.z = parseFloat(document.getElementById('posZ').value) || 0;
    part.scale = parseFloat(document.getElementById('partScale').value) || 1;
    
    document.getElementById('scaleValue').textContent = part.scale.toFixed(1) + 'x';
    
    // Update mesh
    const mesh = state.partMeshes[part.id];
    if (mesh) {
        mesh.position.set(part.position.x, part.position.y, part.position.z);
        mesh.scale.setScalar(part.scale);
        
        // Update color based on type
        const colors = { cast: 0xb0bec5, riser: 0xf59e0b, chill: 0x3b82f6 };
        mesh.material.color.setHex(colors[part.type]);
    }
    
    updatePartsList();
}

function deleteSelectedPart() {
    if (!state.selectedPartId) return;
    
    const partId = state.selectedPartId;
    
    // Remove mesh from scene
    const mesh = state.partMeshes[partId];
    if (mesh) {
        state.scene.remove(mesh);
        delete state.partMeshes[partId];
    }
    
    // Remove from parts array
    state.parts = state.parts.filter(p => p.id !== partId);
    
    // Clear selection
    state.selectedPartId = null;
    
    updatePartsList();
    updateModelInfo();
}

// ============================================
// Material Info
// ============================================

function getEquivalentLinearShrinkage(volumetric) {
    const total = Number(volumetric);
    if (!Number.isFinite(total) || total <= 0) {
        return '--';
    }
    const retained = Math.max(0, 1 - (total / 100));
    const linearApprox = (1 - Math.pow(retained, 1 / 3)) * 100;
    return `${linearApprox.toFixed(1)}%`;
}

function updateMaterialInfo() {
    const alloy = document.getElementById('alloySelect').value;
    const material = state.materials.find(m => m.name === alloy);
    
    if (!material) {
        // Use fallback data
        const fallbackData = {
            WCB: { shrinkage: 8.3, feeding: 6.0, niyama: 1.0 },
            CF8: { shrinkage: 7.5, feeding: 5.5, niyama: 0.7 },
            C958: { shrinkage: 7.1, feeding: 8.0, niyama: 1.2 },
            C964: { shrinkage: 5.4, feeding: 7.8, niyama: 1.1 },
            CA6NM: { shrinkage: 6.3, feeding: 5.9, niyama: 0.95 },
            MONEL400: { shrinkage: 3.0, feeding: 6.4, niyama: 0.85 },
            IN625: { shrinkage: 2.7, feeding: 5.8, niyama: 0.9 }
        };
        const data = fallbackData[alloy] || fallbackData.WCB;
        
        document.getElementById('materialInfo').innerHTML = `
            <div class="material-info-row">
                <span>Volumetric Shrinkage</span>
                <span>${data.shrinkage}%</span>
            </div>
            <div class="material-info-row">
                <span>Equivalent linear (isotropic proxy)</span>
                <span>${getEquivalentLinearShrinkage(data.shrinkage)}</span>
            </div>
            <div class="material-info-row">
                <span>Feeding Multiplier</span>
                <span>${data.feeding}x</span>
            </div>
            <div class="material-info-row">
                <span>Niyama Threshold</span>
                <span>${data.niyama}</span>
            </div>
        `;
        return;
    }
    
    document.getElementById('materialInfo').innerHTML = `
        <div class="material-info-row">
            <span>Volumetric Shrinkage</span>
            <span>${material.volumetric_shrinkage}%</span>
        </div>
        <div class="material-info-row">
            <span>Equivalent linear (isotropic proxy)</span>
            <span>${getEquivalentLinearShrinkage(material.volumetric_shrinkage)}</span>
        </div>
        <div class="material-info-row">
            <span>Feeding Multiplier</span>
            <span>${material.feeding_multiplier}x</span>
        </div>
        <div class="material-info-row">
            <span>Niyama Threshold</span>
            <span>${material.niyama_threshold}</span>
        </div>
    `;
}

// ============================================
// Mesh Preview
// ============================================

// ============================================
// Mesh Settings
// ============================================

function toggleAutoMesh() {
    const autoMesh = document.getElementById('autoMesh').checked;
    const manualSettings = document.getElementById('manualMeshSettings');
    const autoInfo = document.getElementById('autoMeshInfo');
    
    if (autoMesh) {
        manualSettings.style.opacity = '0.5';
        manualSettings.style.pointerEvents = 'none';
        autoInfo.style.display = 'block';
        // Get ML recommendation if we have parts loaded
        if (state.parts.length > 0) {
            getAutoMeshRecommendation();
        }
    } else {
        manualSettings.style.opacity = '1';
        manualSettings.style.pointerEvents = 'auto';
        autoInfo.style.display = 'none';
    }
}

async function getAutoMeshRecommendation() {
    if (state.parts.length === 0) return;
    
    const part = state.parts[0];
    if (!part.meshData) return;
    
    try {
        const response = await fetch('/api/mesh/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bounds: part.meshData.bounds,
                vertex_count: part.meshData.vertex_count,
                face_count: part.meshData.face_count,
                volume: part.meshData.volume || 0
            })
        });
        
        const recommendation = await response.json();
        
        if (recommendation.voxel_size) {
            document.getElementById('voxelSize').value = recommendation.voxel_size.toFixed(3);
            document.getElementById('refinementLevel').value = recommendation.refinement_level || 4;
            document.getElementById('mlConfidence').textContent = 
                (recommendation.confidence * 100).toFixed(0) + '%';
            
            updateMeshPreview();
        }
    } catch (error) {
        console.error('Auto mesh recommendation failed:', error);
        document.getElementById('mlConfidence').textContent = 'N/A';
    }
}

async function updateMeshPreview() {
    if (state.parts.length === 0) return;
    
    const voxelSize = parseFloat(document.getElementById('voxelSize').value) || 0.1;
    const refinementLevel = parseInt(document.getElementById('refinementLevel').value) || 3;
    
    // For now, use simple estimation
    const part = state.parts[0];
    if (!part || !part.meshData) return;
    
    const bounds = part.meshData.bounds;
    const xRange = bounds.x[1] - bounds.x[0];
    const yRange = bounds.y[1] - bounds.y[0];
    const zRange = bounds.z[1] - bounds.z[0];
    
    const nx = Math.ceil(xRange / voxelSize);
    const ny = Math.ceil(yRange / voxelSize);
    const nz = Math.ceil(zRange / voxelSize);
    
    const totalGrid = nx * ny * nz;
    const fillRatio = 0.5;  // Estimate
    const estimatedVoxels = Math.floor(totalGrid * fillRatio * (1 + (refinementLevel - 1) * 0.3));
    const estimatedMemory = (estimatedVoxels * 280 / 1024 / 1024).toFixed(0);
    
    document.getElementById('estVoxels').textContent = estimatedVoxels.toLocaleString();
    document.getElementById('estMemory').textContent = estimatedMemory + ' MB';
}

function updateSandRestraintLabel() {
    const slider = document.getElementById('sandRestraintIndex');
    const valueLabel = document.getElementById('sandRestraintValue');
    if (!slider || !valueLabel) return;
    valueLabel.textContent = (parseInt(slider.value, 10) / 100).toFixed(2);
}

// ============================================
// Simulation
// ============================================

async function runSimulation() {
    if (state.parts.length === 0) {
        alert('Please load at least one part first.');
        return;
    }
    
    // Clear previous results before running new simulation
    clearResults();
    
    // Mark as simulating
    state.isSimulating = true;
    
    // Create simulation
    const getSp = (id, fallback) => parseFloat(document.getElementById(id)?.value) || fallback;
    const simData = {
        name: 'Web Simulation ' + new Date().toLocaleString(),
        alloy: document.getElementById('alloySelect').value,
        voxel_size: parseFloat(document.getElementById('voxelSize').value) || 0.1,
        refinement_level: parseInt(document.getElementById('refinementLevel').value) || 4,
        analysis_options: {
            shrinkage: document.getElementById('optShrinkage').checked,
            hotspot: document.getElementById('optHotspot').checked,
            niyama: document.getElementById('optNiyama').checked,
            hot_tear: document.getElementById('optHotTear').checked,
            sand_burn_in: document.getElementById('optSandBurnIn').checked,
            fraction_liquid: document.getElementById('optFractionLiquid').checked,
            sand_restraint_index: (parseInt(document.getElementById('sandRestraintIndex').value, 10) || 50) / 100
        },
        solidification_params: {
            feeding: {
                base_feeding_distance: getSp('sp_base_feeding_distance', 6.0),
                min_influence_factor: getSp('sp_min_influence_factor', 0.6),
                riser_max_reduction: getSp('sp_riser_max_reduction', 0.10),
                riser_min_reduction: getSp('sp_riser_min_reduction', 1.0),
                post_contact_reduction: getSp('sp_post_contact_reduction', 0.35),
            },
            chill: {
                max_chill_boost: getSp('sp_max_chill_boost', 3.0),
                shrinkage_radius_multiplier: getSp('sp_shrinkage_radius_multiplier', 1.5),
                shrinkage_max_reduction: getSp('sp_shrinkage_max_reduction', 0.15),
            },
            exothermic: {
                ignition_delay: getSp('sp_ignition_delay', 10.0),
                burn_duration: getSp('sp_burn_duration', 120.0),
                insulation_only_factor: getSp('sp_insulation_only_factor', 0.7),
                full_exothermic_factor: getSp('sp_full_exothermic_factor', 0.5),
                decay_time_constant: getSp('sp_decay_time_constant', 60.0),
            },
            air_gap: {
                time_constant: getSp('sp_air_gap_time_constant', 60.0),
                surface_influence_depth: getSp('sp_surface_influence_depth', 0.5),
                max_slowdown: getSp('sp_max_slowdown', 0.75),
            },
            thermal_boundary: {
                mold_temperature: getSp('sp_mold_temperature', 200.0),
                chill_temperature: getSp('sp_chill_temperature', 25.0),
                htc_cast_interface: getSp('sp_htc_cast_interface', 500.0),
                htc_riser_post: getSp('sp_htc_riser_post', 350.0),
                htc_riser: getSp('sp_htc_riser', 220.0),
            },
            timing: {
                chvorinov_mold_constant: getSp('sp_chvorinov_mold_constant', 280.0),
                safety_factor_with_risers: getSp('sp_safety_factor_with_risers', 2.5),
                safety_factor_no_risers: getSp('sp_safety_factor_no_risers', 1.5),
                min_simulation_time: getSp('sp_min_simulation_time', 120.0),
                max_simulation_time: getSp('sp_max_simulation_time', 14400.0),
                sand_saturation_time: getSp('sp_sand_saturation_time', 30.0),
                max_hot_sand_slowdown: getSp('sp_max_hot_sand_slowdown', 0.6),
            },
        }
    };
    
    try {
        // Create simulation in database
        let response = await fetch('/api/simulations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(simData)
        });
        
        const simResult = await response.json();
        state.simulationId = simResult.id;
        
        // Add parts to simulation
        for (const part of state.parts) {
            await fetch(`/api/simulations/${state.simulationId}/parts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: part.name,
                    part_type: part.type,
                    filepath: part.filepath,
                    position_x: part.position.x,
                    position_y: part.position.y,
                    position_z: part.position.z,
                    scale: part.scale
                })
            });
        }
        
        // Start simulation
        response = await fetch(`/api/simulations/${state.simulationId}/run`, {
            method: 'POST'
        });
        
        const runResult = await response.json();
        
        // Check for rate limiting or queue full
        if (runResult.error) {
            alert(runResult.error);
            setStatus(runResult.error);
            document.getElementById('runSimBtn').disabled = false;
            state.isSimulating = false;
            return;
        }
        
        // Show progress
        document.getElementById('simProgress').style.display = 'block';
        document.getElementById('runSimBtn').disabled = true;
        
        // Show queue status if queued
        if (runResult.status === 'queued') {
            document.getElementById('progressText').textContent = 
                `Queued (position ${runResult.queue_position} of ${runResult.queue_size})`;
            setStatus(`Simulation queued - position ${runResult.queue_position}`);
        }
        
        // Poll for status
        pollSimulationStatus();
        
    } catch (error) {
        console.error('Simulation error:', error);
        setStatus('Simulation failed: ' + error.message);
        document.getElementById('runSimBtn').disabled = false;
        state.isSimulating = false;
    }
}

async function pollSimulationStatus() {
    if (!state.simulationId) return;
    
    try {
        const response = await fetch(`/api/simulations/${state.simulationId}/status`);
        const status = await response.json();
        
        // Update progress based on status
        if (status.status === 'queued') {
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('progressText').textContent = status.message || `Queued (position ${status.queue_position})`;
            // Continue polling (slower for queue)
            setTimeout(pollSimulationStatus, 2000);
        } else {
            // Running or other status
            document.getElementById('progressFill').style.width = status.progress + '%';
            document.getElementById('progressText').textContent = status.message || 'Processing...';
            
            if (status.status === 'completed') {
                // Load results
                state.isSimulating = false;
                loadResults();
                document.getElementById('runSimBtn').disabled = false;
                document.getElementById('simProgress').style.display = 'none';
                setStatus('Simulation complete!');
            } else if (status.status === 'error' || status.status === 'cancelled') {
                state.isSimulating = false;
                document.getElementById('runSimBtn').disabled = false;
                document.getElementById('simProgress').style.display = 'none';
                setStatus(status.status === 'cancelled' ? 'Simulation cancelled' : 'Error: ' + status.message);
            } else {
                // Continue polling
                setTimeout(pollSimulationStatus, 500);
            }
        }
        
    } catch (error) {
        console.error('Status poll error:', error);
        setTimeout(pollSimulationStatus, 1000);
    }
}

async function loadResults() {
    if (!state.simulationId) return;
    
    try {
        const response = await fetch(`/api/simulations/${state.simulationId}/results`);
        const results = await response.json();
        
        state.resultData = results;
        state.hasResults = true;
        
        // Show results panel
        document.getElementById('resultsPanel').style.display = 'block';
        
        // Update result stats
        updateResultStats();
        
        // Visualize results
        visualizeResults('shrinkage');
        
    } catch (error) {
        console.error('Failed to load results:', error);
    }
}

function updateResultStats() {
    if (!state.resultData || !state.resultData.data) return;
    
    const data = state.resultData.data;
    const container = document.getElementById('resultStats');
    
    let html = `
        <div class="result-stat">
            <span>Total Voxels</span>
            <span>${data.voxel_count?.toLocaleString() || '--'}</span>
        </div>
        <div class="result-stat">
            <span>Total Volume</span>
            <span>${data.total_volume?.toFixed(2) || '--'} in³</span>
        </div>
    `;
    
    if (data.shrinkage_stats) {
        html += `
            <div class="result-stat highlight">
                <span>Defect Voxels</span>
                <span>${data.shrinkage_stats.affected_count?.toLocaleString() || '--'}</span>
            </div>
            <div class="result-stat">
                <span>Max Shrinkage</span>
                <span>${(data.shrinkage_stats.max * 100)?.toFixed(1) || '--'}%</span>
            </div>
        `;
    }
    
    if (data.hotspot_stats) {
        html += `
            <div class="result-stat">
                <span>Hot Spots</span>
                <span>${data.hotspot_stats.count?.toLocaleString() || '--'}</span>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

function visualizeResults(type) {
    if (!state.resultData || !state.resultData.visualization) {
        console.log('No visualization data available');
        return;
    }
    
    const vis = state.resultData.visualization;
    
    // Hide all cached meshes first
    for (const [key, mesh] of Object.entries(state.resultMeshes)) {
        mesh.visible = false;
    }
    
    if (type === 'model') return;
    
    // Check cache
    const cacheKey = 'result_' + type;
    if (state.resultMeshes[cacheKey]) {
        state.resultMeshes[cacheKey].visible = true;
        return;
    }
    
    const centers = vis.centers;
    const sizes = vis.sizes;
    let values;
    
    switch (type) {
        case 'shrinkage':
            values = vis.shrinkage;
            break;
        case 'hotspot':
            values = vis.hotspot;
            break;
        case 'niyama':
            values = vis.niyama;
            break;
        default:
            return;
    }
    
    if (!centers || !values) {
        console.log('Missing centers or values:', { centers: !!centers, values: !!values });
        return;
    }
    
    console.log('Visualization data:', { 
        centersLen: centers.length, 
        valuesLen: values.length,
        sizesLen: sizes ? sizes.length : 0,
        maxValue: Math.max(...values),
        minValue: Math.min(...values)
    });
    
    const threshold = document.getElementById('threshold').value / 100;
    const cutoff = threshold;

    // Count voxels in selected top percentage
    let count = 0;
    for (let i = 0; i < values.length; i++) {
        if (values[i] >= cutoff) count++;
    }
    
    console.log('Min-threshold voxel filter', { type, threshold, cutoff, count });
    
    if (count === 0) {
        document.getElementById('voxelCount').textContent = 'Voxels: 0 (try lower threshold)';
        return;
    }
    
    // Create geometry for each defect voxel as individual cubes
    const defectGroup = new THREE.Group();
    
    for (let i = 0; i < values.length; i++) {
        if (values[i] < cutoff) continue;
        
        const center = centers[i];
        const size = (sizes && sizes[i]) ? sizes[i] : 0.1;
        const value = values[i];
        
        // Create cube for this defect
        const geometry = new THREE.BoxGeometry(size * 1.5, size * 1.5, size * 1.5);
        
        // Color based on value (red = high, yellow = low)
        const hue = (1 - value) * 0.15;  // 0 (red) to 0.15 (orange/yellow)
        const color = new THREE.Color();
        color.setHSL(hue, 1, 0.5);
        
        const material = new THREE.MeshPhongMaterial({
            color: color,
            transparent: false,
            side: THREE.DoubleSide,
            depthWrite: true,
            depthTest: true
        });
        
        const cube = new THREE.Mesh(geometry, material);
        cube.position.set(center[0], center[1], center[2]);
        cube.renderOrder = 1;
        defectGroup.add(cube);
    }
    
    defectGroup.renderOrder = 1;
    state.scene.add(defectGroup);
    state.resultMeshes[cacheKey] = defectGroup;
    
    document.getElementById('voxelCount').textContent = 'Defects: ' + count.toLocaleString();
    console.log('Added', count, 'defect cubes to scene');
}

function computeTopPercentCutoff(values, topFraction) {
    if (!values || values.length === 0) return Infinity;
    if (topFraction <= 0) return Infinity;
    if (topFraction >= 1) return -Infinity;

    const finiteValues = [];
    for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (Number.isFinite(v)) {
            finiteValues.push(v);
        }
    }

    if (finiteValues.length === 0) return Infinity;

    finiteValues.sort((a, b) => a - b);
    const keepCount = Math.max(1, Math.ceil(topFraction * finiteValues.length));
    const cutoffIndex = finiteValues.length - keepCount;
    return finiteValues[cutoffIndex];
}

function updateResultDisplay() {
    const type = document.getElementById('resultDisplay').value;
    
    // Show/hide fraction liquid controls and threshold controls
    const flControls = document.getElementById('fractionLiquidControls');
    const thresholdControls = document.getElementById('thresholdControls');
    
    if (type === 'fraction_liquid') {
        flControls.style.display = 'block';
        thresholdControls.style.display = 'none';
        visualizeFractionLiquid();
    } else {
        flControls.style.display = 'none';
        thresholdControls.style.display = 'block';
        visualizeResults(type);
    }
}

// Debounce timer for fraction liquid slider
let flDebounceTimer = null;

// Animation state for fraction liquid
let flAnimationId = null;
let flAnimationSpeed = 1.0;
let flIsPlaying = false;

function toggleFLAnimation() {
    if (flIsPlaying) {
        stopFLAnimation();
    } else {
        startFLAnimation();
    }
}

function startFLAnimation() {
    flIsPlaying = true;
    const playIcon = document.getElementById('flPlayIcon');
    if (playIcon) {
        playIcon.className = 'fas fa-pause';
    }
    animateFractionLiquid();
}

function stopFLAnimation() {
    flIsPlaying = false;
    const playIcon = document.getElementById('flPlayIcon');
    if (playIcon) {
        playIcon.className = 'fas fa-play';
    }
    if (flAnimationId) {
        cancelAnimationFrame(flAnimationId);
        flAnimationId = null;
    }
}

function resetFLAnimation() {
    stopFLAnimation();
    const timeSlider = document.getElementById('flTimeSlider');
    if (timeSlider) {
        timeSlider.value = 0;
        updateFractionLiquidTime();
    }
}

function updateFLSpeed() {
    const speedSlider = document.getElementById('flSpeedSlider');
    const speedLabel = document.getElementById('flSpeedValue');
    if (speedSlider && speedLabel) {
        // Map 1-10 to 0.25x - 3.0x
        const val = parseInt(speedSlider.value);
        flAnimationSpeed = 0.25 + (val - 1) * (2.75 / 9);
        speedLabel.textContent = flAnimationSpeed.toFixed(1) + 'x';
    }
}

let lastAnimationTime = 0;
function animateFractionLiquid() {
    if (!flIsPlaying) return;
    
    const now = performance.now();
    const elapsed = now - lastAnimationTime;
    
    // Update every 50ms base, adjusted by speed
    const interval = 50 / flAnimationSpeed;
    
    if (elapsed >= interval) {
        lastAnimationTime = now;
        
        const timeSlider = document.getElementById('flTimeSlider');
        if (timeSlider) {
            let currentValue = parseInt(timeSlider.value);
            currentValue += 1;
            
            if (currentValue > 100) {
                // Loop back to start
                currentValue = 0;
            }
            
            timeSlider.value = currentValue;
            updateFractionLiquidTime();
        }
    }
    
    flAnimationId = requestAnimationFrame(animateFractionLiquid);
}

function getFractionLiquidPlaybackTime(vis, sliderPercent) {
    if (!vis || !vis.fraction_liquid_times || vis.fraction_liquid_times.length === 0) {
        return null;
    }

    const times = vis.fraction_liquid_times;
    const minTime = times[0];
    const maxTime = times[times.length - 1];
    return minTime + (sliderPercent / 100.0) * (maxTime - minTime);
}

function getFractionLiquidSnapshotAtTime(vis, targetTime) {
    if (!vis || !vis.fraction_liquid_history || vis.fraction_liquid_history.length === 0) {
        return null;
    }

    const history = vis.fraction_liquid_history;
    const times = vis.fraction_liquid_times;
    if (!times || times.length !== history.length || times.length === 0) {
        const snapshotIndex = Math.min(
            Math.floor((targetTime / 100.0) * history.length),
            history.length - 1
        );
        return history[snapshotIndex];
    }

    if (targetTime <= times[0]) {
        return history[0];
    }
    if (targetTime >= times[times.length - 1]) {
        return history[history.length - 1];
    }

    let idxAfter = 0;
    while (idxAfter < times.length && times[idxAfter] < targetTime) {
        idxAfter += 1;
    }

    const idxBefore = Math.max(idxAfter - 1, 0);
    const beforeTime = times[idxBefore];
    const afterTime = times[idxAfter];
    const beforeState = history[idxBefore];
    const afterState = history[idxAfter];

    if (afterTime <= beforeTime) {
        return beforeState;
    }

    const alpha = (targetTime - beforeTime) / (afterTime - beforeTime);
    return beforeState.map((value, index) => value * (1.0 - alpha) + afterState[index] * alpha);
}

function updateFractionLiquidTime() {
    // Get time slider - support both old and new IDs
    const timeSlider = document.getElementById('flTimeSlider') || document.getElementById('flSlider');
    if (!timeSlider) {
        console.log('Time slider not found');
        return;
    }
    
    const timeValue = parseInt(timeSlider.value);
    const vis = state.resultData && state.resultData.visualization
        ? state.resultData.visualization
        : null;
    const actualTime = getFractionLiquidPlaybackTime(vis, timeValue);
    let stateDesc;
    if (timeValue === 0) stateDesc = "Just Poured";
    else if (timeValue < 25) stateDesc = "Mostly Liquid";
    else if (timeValue < 50) stateDesc = "Solidifying";
    else if (timeValue < 75) stateDesc = "Mostly Solid";
    else if (timeValue < 100) stateDesc = "Nearly Solid";
    else stateDesc = "Fully Solid";
    
    // Update label if it exists
    const timeLabel = document.getElementById('flTimeValue') || document.getElementById('flValue');
    if (timeLabel) {
        if (actualTime !== null) {
            timeLabel.textContent = actualTime.toFixed(1) + 's - ' + stateDesc;
        } else {
            timeLabel.textContent = timeValue + '% - ' + stateDesc;
        }
    }
    
    // Debounce visualization update
    if (flDebounceTimer) clearTimeout(flDebounceTimer);
    flDebounceTimer = setTimeout(() => {
        visualizeFractionLiquid();
        flDebounceTimer = null;
    }, 50);
}

function updateFractionLiquidThreshold() {
    const thresholdSlider = document.getElementById('flThresholdSlider');
    if (!thresholdSlider) {
        // If no threshold slider, just update visualization
        visualizeFractionLiquid();
        return;
    }
    
    const thresholdValue = parseInt(thresholdSlider.value);
    const thresholdLabel = document.getElementById('flThresholdValue');
    if (thresholdLabel) {
        thresholdLabel.textContent = thresholdValue + '%';
    }
    
    // Debounce visualization update
    if (flDebounceTimer) clearTimeout(flDebounceTimer);
    flDebounceTimer = setTimeout(() => {
        visualizeFractionLiquid();
        flDebounceTimer = null;
    }, 50);
}

function updateFractionLiquid() {
    // Legacy function - call visualization directly
    visualizeFractionLiquid();
}

function visualizeFractionLiquid() {
    if (!state.resultData || !state.resultData.visualization) {
        console.log('No result data for fraction liquid');
        return;
    }
    
    const vis = state.resultData.visualization;
    const centers = vis.centers;
    const sizes = vis.sizes;
    
    // Get time slider value (0-100) - support both old and new IDs
    const timeSlider = document.getElementById('flTimeSlider') || document.getElementById('flSlider');
    const thresholdSlider = document.getElementById('flThresholdSlider');
    
    // Default values if sliders not found
    const timeValue = timeSlider ? parseInt(timeSlider.value) : 0;
    const thresholdValue = thresholdSlider ? parseInt(thresholdSlider.value) : 1;
    const threshold = thresholdValue / 100.0;  // 0.01 to 1.0
    
    console.log('FL visualization: time=' + timeValue + '%, threshold=' + thresholdValue + '%');
    
    const targetTime = getFractionLiquidPlaybackTime(vis, timeValue);

    // Get fraction liquid data - use history for time-accurate animation
    let fractionLiquid = null;
    let usingHistory = false;
    
    if (vis.fraction_liquid_history && vis.fraction_liquid_history.length > 0) {
        usingHistory = true;
        fractionLiquid = getFractionLiquidSnapshotAtTime(
            vis,
            targetTime !== null ? targetTime : timeValue
        );
        console.log('Using time-accurate FL history at', targetTime !== null ? targetTime : timeValue);
    } else if (vis.fraction_liquid) {
        // Use final state - time slider will simulate progression
        fractionLiquid = vis.fraction_liquid;
        console.log('Using final fraction_liquid state, length=' + fractionLiquid.length);
    } else if (vis.solidification_times) {
        // Fallback: convert solidification times to fraction liquid
        const maxTime = Math.max(...vis.solidification_times);
        if (maxTime > 0) {
            fractionLiquid = vis.solidification_times.map(t => 1.0 - (t / maxTime));
        }
        console.log('Using solidification_times fallback');
    }
    
    if (!centers || !fractionLiquid) {
        console.log('No fraction liquid data available');
        console.log('vis keys:', Object.keys(vis));
        return;
    }
    
    // Single cache key for fraction liquid - reuse same mesh
    const cacheKey = 'fl_current';
    
    // Hide other result meshes
    for (const [key, mesh] of Object.entries(state.resultMeshes)) {
        if (!key.startsWith('fl_')) {
            mesh.visible = false;
        }
    }
    
    // Remove old fraction liquid mesh to free memory
    if (state.resultMeshes[cacheKey]) {
        state.scene.remove(state.resultMeshes[cacheKey]);
        if (state.resultMeshes[cacheKey].geometry) state.resultMeshes[cacheKey].geometry.dispose();
        if (state.resultMeshes[cacheKey].material) state.resultMeshes[cacheKey].material.dispose();
        delete state.resultMeshes[cacheKey];
    }
    
    // Find liquid voxels
    // Time slider: 0% = just poured (show all liquid), 100% = fully solid (show nothing)
    // Threshold slider: only show voxels with fraction liquid > threshold
    const liquidIndices = [];
    const timeFactor = timeValue / 100.0;  // 0.0 to 1.0
    
    for (let i = 0; i < fractionLiquid.length; i++) {
        const fl = fractionLiquid[i];
        
        // If using history, time is already factored in - just use threshold
        // If using final state, simulate time by requiring higher fl for later times
        let effectiveThreshold = threshold;
        if (!usingHistory) {
            // Simulate time: at time=0 show all liquid > threshold
            // At time=100, show only voxels with fl > 1.0 (none)
            effectiveThreshold = threshold + timeFactor * (1.0 - threshold);
        }
        
        if (fl > effectiveThreshold) {
            liquidIndices.push(i);
        }
    }
    
    console.log('Liquid voxels found:', liquidIndices.length, 'of', fractionLiquid.length);
    
    const voxelCountEl = document.getElementById('voxelCount');
    if (liquidIndices.length === 0) {
        if (voxelCountEl) voxelCountEl.textContent = 'Liquid: 0';
        return;
    }
    
    // Use instanced mesh for performance
    const baseSize = (sizes && sizes[0]) ? sizes[0] : 0.1;
    const geometry = new THREE.BoxGeometry(baseSize * 1.5, baseSize * 1.5, baseSize * 1.5);
    
    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        emissive: new THREE.Color(0.3, 0.1, 0.0),
        emissiveIntensity: 0.3,
        transparent: false,
        side: THREE.DoubleSide,
        depthWrite: true
    });
    
    const instancedMesh = new THREE.InstancedMesh(geometry, material, liquidIndices.length);
    const dummy = new THREE.Object3D();
    const colors = new Float32Array(liquidIndices.length * 3);
    
    for (let j = 0; j < liquidIndices.length; j++) {
        const i = liquidIndices[j];
        const center = centers[i];
        const fl = fractionLiquid[i];
        
        dummy.position.set(center[0], center[1], center[2]);
        dummy.updateMatrix();
        instancedMesh.setMatrixAt(j, dummy.matrix);
        
        // Lava color: yellow (hot) -> orange -> red (cooling)
        let r, g, b;
        if (fl > 0.7) {
            r = 1.0;
            g = 0.6 + (fl - 0.7) * 1.33;
            b = (fl - 0.7) * 0.67;
        } else if (fl > 0.3) {
            r = 1.0;
            g = (fl - 0.3) * 1.5;
            b = 0;
        } else {
            r = 0.4 + fl * 2.0;
            g = fl * 0.3;
            b = 0;
        }
        
        colors[j * 3] = r;
        colors[j * 3 + 1] = g;
        colors[j * 3 + 2] = b;
    }
    
    instancedMesh.instanceMatrix.needsUpdate = true;
    geometry.setAttribute('color', new THREE.InstancedBufferAttribute(colors, 3));
    
    instancedMesh.renderOrder = 1;
    state.scene.add(instancedMesh);
    state.resultMeshes[cacheKey] = instancedMesh;
    
    if (voxelCountEl) voxelCountEl.textContent = 'Liquid: ' + liquidIndices.length.toLocaleString();
}

function updateThreshold() {
    const value = document.getElementById('threshold').value;
    document.getElementById('thresholdValue').textContent = value + '%';
    
    const type = document.getElementById('resultDisplay').value;
    
    // Threshold changed - need to rebuild visualization (clear cache for this type)
    const cacheKey = 'result_' + type;
    if (state.resultMeshes[cacheKey]) {
        state.scene.remove(state.resultMeshes[cacheKey]);
        delete state.resultMeshes[cacheKey];
    }
    
    visualizeResults(type);
}

// ============================================
// Viewport Controls
// ============================================

function resetCamera() {
    state.camera.position.set(50, -50, 50);
    state.camera.up.set(0, 0, 1);
    state.camera.lookAt(0, 0, 0);
    state.controls.target.set(0, 0, 0);
    state.controls.update();
}

function setView(view) {
    const distance = 100;
    
    // Z is up coordinate system
    switch (view) {
        case 'front':
            state.camera.position.set(0, -distance, 0);
            break;
        case 'top':
            state.camera.position.set(0, 0, distance);
            break;
        case 'side':
            state.camera.position.set(distance, 0, 0);
            break;
        case 'iso':
            state.camera.position.set(distance * 0.7, -distance * 0.7, distance * 0.7);
            break;
    }
    
    state.camera.up.set(0, 0, 1);
    state.camera.lookAt(0, 0, 0);
    state.controls.target.set(0, 0, 0);
    state.controls.update();
}

function fitCameraToModel() {
    if (state.parts.length === 0) return;
    
    // Calculate bounding box of all parts
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    
    for (const part of state.parts) {
        if (!part.meshData || !part.meshData.bounds) continue;
        
        const bounds = part.meshData.bounds;
        minX = Math.min(minX, bounds.x[0]);
        maxX = Math.max(maxX, bounds.x[1]);
        minY = Math.min(minY, bounds.y[0]);
        maxY = Math.max(maxY, bounds.y[1]);
        minZ = Math.min(minZ, bounds.z[0]);
        maxZ = Math.max(maxZ, bounds.z[1]);
    }
    
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const distance = size * 2;
    
    // Z is up - position camera to look at model from front-ish angle
    state.camera.position.set(
        centerX + distance * 0.7,
        centerY - distance * 0.7,
        centerZ + distance * 0.5
    );
    state.camera.up.set(0, 0, 1);
    state.camera.lookAt(centerX, centerY, centerZ);
    state.controls.target.set(centerX, centerY, centerZ);
    state.controls.update();
}

function toggleWireframe() {
    state.showWireframe = !state.showWireframe;
    
    for (const mesh of Object.values(state.partMeshes)) {
        if (mesh.userData.wireframe) {
            mesh.userData.wireframe.visible = state.showWireframe;
        }
    }
}

function toggleModelVisibility() {
    const show = document.getElementById('showModel').checked;
    
    for (const mesh of Object.values(state.partMeshes)) {
        mesh.visible = show;
    }
}

function toggleGrid(fromCheckbox = false) {
    const checkbox = document.getElementById('showGrid');
    
    // If called from toolbar button, toggle the checkbox
    if (!fromCheckbox && checkbox) {
        checkbox.checked = !checkbox.checked;
    }
    
    const show = checkbox ? checkbox.checked : true;
    
    if (state.gridHelper) {
        state.gridHelper.visible = show;
        console.log('Grid visibility:', show);
    } else {
        console.log('Grid helper not found');
    }
}

function toggleAxes(fromCheckbox = false) {
    const checkbox = document.getElementById('showAxes');
    
    // If called from toolbar button, toggle the checkbox
    if (!fromCheckbox && checkbox) {
        checkbox.checked = !checkbox.checked;
    }
    
    const show = checkbox ? checkbox.checked : !state.axesHelper?.visible;
    
    if (state.axesHelper) {
        state.axesHelper.visible = show;
    }
}

function updateModelOpacity() {
    const opacity = document.getElementById('modelOpacity').value / 100;
    document.getElementById('opacityValue').textContent = Math.round(opacity * 100) + '%';
    
    for (const mesh of Object.values(state.partMeshes)) {
        mesh.material.opacity = opacity;
        // When model is very transparent, allow depth writing for defects
        mesh.material.depthWrite = opacity > 0.5;
    }
}

function updateColorMap() {
    // Color maps would be applied to result visualization
    const type = document.getElementById('resultDisplay').value;
    if (state.hasResults) {
        visualizeResults(type);
    }
}

function toggleColorBar() {
    // Would show/hide color bar legend
    const show = document.getElementById('showColorBar').checked;
    // Implementation depends on color bar DOM element
}

// ============================================
// Cross Section
// ============================================

function toggleCrossSection() {
    state.crossSectionEnabled = document.getElementById('enableCrossSection').checked;
    
    if (state.crossSectionEnabled) {
        updateCrossSection();
    } else {
        // Remove clipping from all meshes
        for (const mesh of Object.values(state.partMeshes)) {
            mesh.material.clippingPlanes = [];
        }
        for (const mesh of Object.values(state.resultMeshes)) {
            mesh.material.clippingPlanes = [];
        }
    }
}

function updateCrossSection() {
    if (!state.crossSectionEnabled) return;
    
    const axis = document.getElementById('crossSectionAxis').value;
    const pos = document.getElementById('crossSectionPos').value / 100;
    
    document.getElementById('crossSectionValue').textContent = Math.round(pos * 100) + '%';
    
    // Calculate plane position based on model bounds
    let planeNormal, planeConstant;
    
    if (state.parts.length > 0 && state.parts[0].meshData) {
        const bounds = state.parts[0].meshData.bounds;
        
        switch (axis) {
            case 'x':
                planeNormal = new THREE.Vector3(-1, 0, 0);
                planeConstant = bounds.x[0] + (bounds.x[1] - bounds.x[0]) * pos;
                break;
            case 'y':
                planeNormal = new THREE.Vector3(0, -1, 0);
                planeConstant = bounds.y[0] + (bounds.y[1] - bounds.y[0]) * pos;
                break;
            case 'z':
                planeNormal = new THREE.Vector3(0, 0, -1);
                planeConstant = bounds.z[0] + (bounds.z[1] - bounds.z[0]) * pos;
                break;
        }
    } else {
        planeNormal = new THREE.Vector3(0, 0, -1);
        planeConstant = pos * 20 - 10;
    }
    
    state.clippingPlane.normal.copy(planeNormal);
    state.clippingPlane.constant = planeConstant;
    
    // Apply to all meshes
    for (const mesh of Object.values(state.partMeshes)) {
        mesh.material.clippingPlanes = [state.clippingPlane];
    }
    for (const mesh of Object.values(state.resultMeshes)) {
        mesh.material.clippingPlanes = [state.clippingPlane];
    }
}

// ============================================
// Measurement
// ============================================

function startMeasurement() {
    state.measureMode = !state.measureMode;
    state.measurePoints = [];
    
    const btn = document.getElementById('measureBtn');
    if (state.measureMode) {
        btn.classList.add('btn-primary');
        setStatus('Click two points to measure distance');
    } else {
        btn.classList.remove('btn-primary');
        setStatus('Ready');
        closeMeasurement();
    }
}

function onViewportClick(event) {
    if (!state.measureMode) return;
    
    const rect = state.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, state.camera);
    
    const meshes = Object.values(state.partMeshes);
    const intersects = raycaster.intersectObjects(meshes);
    
    if (intersects.length > 0) {
        const point = intersects[0].point;
        state.measurePoints.push(point.clone());
        
        // Add marker
        const markerGeometry = new THREE.SphereGeometry(0.2, 16, 16);
        const markerMaterial = new THREE.MeshBasicMaterial({ 
            color: state.measurePoints.length === 1 ? 0x00ff00 : 0xff0000 
        });
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.position.copy(point);
        state.scene.add(marker);
        
        if (state.measurePoints.length === 2) {
            // Calculate distance
            const distance = state.measurePoints[0].distanceTo(state.measurePoints[1]);
            
            // Draw line
            const lineGeometry = new THREE.BufferGeometry().setFromPoints(state.measurePoints);
            const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2 });
            state.measureLine = new THREE.Line(lineGeometry, lineMaterial);
            state.scene.add(state.measureLine);
            
            // Show result
            document.getElementById('measurementValue').textContent = distance.toFixed(3) + '"';
            document.getElementById('measurementPopup').style.display = 'block';
            
            // Exit measure mode
            state.measureMode = false;
            document.getElementById('measureBtn').classList.remove('btn-primary');
        }
    }
}

function closeMeasurement() {
    document.getElementById('measurementPopup').style.display = 'none';
    
    // Clear measurement visuals
    if (state.measureLine) {
        state.scene.remove(state.measureLine);
        state.measureLine = null;
    }
    
    // Clear markers (simplified - would need to track them)
    state.measurePoints = [];
}

// ============================================
// Screenshot
// ============================================

function takeScreenshot() {
    state.renderer.render(state.scene, state.camera);
    const dataUrl = state.renderer.domElement.toDataURL('image/png');
    
    const link = document.createElement('a');
    link.download = 'casting_screenshot_' + Date.now() + '.png';
    link.href = dataUrl;
    link.click();
}

// ============================================
// Save/Load
// ============================================

function openSaveDialog() {
    document.getElementById('saveName').value = 'Simulation ' + new Date().toLocaleDateString();
    document.getElementById('saveDescription').value = '';
    document.getElementById('saveModal').classList.add('active');
}

function closeSaveDialog() {
    document.getElementById('saveModal').classList.remove('active');
}

async function saveSimulation() {
    const name = document.getElementById('saveName').value || 'Untitled';
    const description = document.getElementById('saveDescription').value;
    
    const simData = {
        name: name,
        description: description,
        alloy: document.getElementById('alloySelect').value,
        voxel_size: parseFloat(document.getElementById('voxelSize').value) || 0.1,
        refinement_level: parseInt(document.getElementById('refinementLevel').value) || 4,
        analysis_options: {
            shrinkage: document.getElementById('optShrinkage').checked,
            hotspot: document.getElementById('optHotspot').checked,
            niyama: document.getElementById('optNiyama').checked,
            hot_tear: document.getElementById('optHotTear').checked,
            sand_burn_in: document.getElementById('optSandBurnIn').checked,
            fraction_liquid: document.getElementById('optFractionLiquid').checked,
            sand_restraint_index: (parseInt(document.getElementById('sandRestraintIndex').value, 10) || 50) / 100
        }
    };
    
    try {
        // Create or update simulation
        let response;
        if (state.simulationId) {
            response = await fetch(`/api/simulations/${state.simulationId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(simData)
            });
        } else {
            response = await fetch('/api/simulations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(simData)
            });
            const result = await response.json();
            state.simulationId = result.id;
            
            // Add parts
            for (const part of state.parts) {
                await fetch(`/api/simulations/${state.simulationId}/parts`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: part.name,
                        part_type: part.type,
                        filepath: part.filepath,
                        position_x: part.position.x,
                        position_y: part.position.y,
                        position_z: part.position.z,
                        scale: part.scale
                    })
                });
            }
        }
        
        closeSaveDialog();
        setStatus('Saved: ' + name);
        
    } catch (error) {
        console.error('Save error:', error);
        setStatus('Save failed');
    }
}

async function openLoadDialog() {
    document.getElementById('loadModal').classList.add('active');
    
    // Load simulation list
    try {
        const response = await fetch('/api/simulations');
        const simulations = await response.json();
        
        const container = document.getElementById('simulationList');
        
        if (simulations.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-folder-open"></i>
                    <p>No saved simulations</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        for (const sim of simulations) {
            const date = new Date(sim.created_at).toLocaleDateString();
            html += `
                <div class="simulation-item" onclick="loadSimulationById('${sim.id}')">
                    <div class="simulation-item-info">
                        <h4>${sim.name}</h4>
                        <p>${sim.description || 'No description'}</p>
                    </div>
                    <div class="simulation-item-meta">
                        <div>${sim.alloy}</div>
                        <div>${date}</div>
                        <div>${sim.status}</div>
                    </div>
                </div>
            `;
        }
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Load error:', error);
    }
}

function closeLoadDialog() {
    document.getElementById('loadModal').classList.remove('active');
}

function openHelpDialog() {
    if (!state.materials || state.materials.length === 0) {
        loadMaterials()
            .then(() => {
                renderHelpAlloyTable();
            })
            .catch(() => {
                renderHelpAlloyTable([]);
            });
    } else {
        renderHelpAlloyTable();
    }
    
    document.getElementById('helpModal').classList.add('active');
}

function closeHelpDialog() {
    document.getElementById('helpModal').classList.remove('active');
}

const HELP_PRESET_CONFIGS = {
    'thin-walls': {
        label: 'Thin wall / fins',
        alloy: 'WCB',
        autoMesh: true,
        visibility: {
            showModel: true,
            showGrid: true,
            showAxes: true,
            showColorBar: false
        },
        analysis: {
            shrinkage: true,
            hotspot: false,
            niyama: false,
            hotTear: false,
            fractionLiquid: true,
            sandBurnIn: false
        },
        sandRestraint: 0.45,
        resultDisplay: 'shrinkage',
        threshold: 25,
        modelOpacity: 30,
        crossSection: false
    },
    'thick-bosses': {
        label: 'Thick bosses / heavy sections',
        alloy: 'C958',
        autoMesh: false,
        visibility: {
            showModel: true,
            showGrid: false,
            showAxes: false,
            showColorBar: false
        },
        voxelSize: 0.06,
        refinement: 4,
        analysis: {
            shrinkage: true,
            hotspot: true,
            niyama: false,
            hotTear: false,
            fractionLiquid: false,
            sandBurnIn: false
        },
        sandRestraint: 0.55,
        resultDisplay: 'hotspot',
        threshold: 20,
        modelOpacity: 40,
        crossSection: false
    },
    'complex-junctions': {
        label: 'Complex junctions / internal pockets',
        alloy: 'CF8',
        autoMesh: true,
        visibility: {
            showModel: true,
            showGrid: true,
            showAxes: true,
            showColorBar: true
        },
        analysis: {
            shrinkage: true,
            hotspot: true,
            niyama: false,
            hotTear: false,
            fractionLiquid: true,
            sandBurnIn: false
        },
        sandRestraint: 0.50,
        resultDisplay: 'fraction_liquid',
        modelOpacity: 22,
        crossSection: true,
        crossSectionAxis: 'z',
        crossSectionPos: 55,
        fractionLiquid: {
            time: 25,
            threshold: 1
        }
    },
    'high-risk-alloy': {
        label: 'Foundry-risk alloy / difficult to cast alloys',
        alloy: 'CF8',
        autoMesh: true,
        analysis: {
            shrinkage: true,
            hotspot: false,
            niyama: true,
            hotTear: false,
            fractionLiquid: false,
            sandBurnIn: true
        },
        sandRestraint: 0.35,
        visibility: {
            showModel: true,
            showGrid: false,
            showAxes: false,
            showColorBar: true
        },
        resultDisplay: 'shrinkage',
        threshold: 18,
        modelOpacity: 35,
        crossSection: true,
        crossSectionAxis: 'x',
        crossSectionPos: 52
    }
};

function applyHelpPreset(presetKey) {
    const preset = HELP_PRESET_CONFIGS[presetKey];
    if (!preset) {
        console.error('Unknown preset:', presetKey);
        return;
    }

    const alloySelect = document.getElementById('alloySelect');
    if (alloySelect && preset.alloy) {
        const optionExists = [...alloySelect.options].some((opt) => opt.value === preset.alloy);
        if (optionExists) {
            alloySelect.value = preset.alloy;
            updateMaterialInfo();
        }
    }

    const autoMeshToggle = document.getElementById('autoMesh');
        if (autoMeshToggle && typeof preset.autoMesh === 'boolean') {
            autoMeshToggle.checked = preset.autoMesh;
            toggleAutoMesh();
        }

        if (preset.visibility) {
            const visibility = preset.visibility;

            const modelCheckbox = document.getElementById('showModel');
            if (modelCheckbox && typeof visibility.showModel === 'boolean') {
                modelCheckbox.checked = visibility.showModel;
                toggleModelVisibility();
            }

            const gridCheckbox = document.getElementById('showGrid');
            if (gridCheckbox && typeof visibility.showGrid === 'boolean') {
                gridCheckbox.checked = visibility.showGrid;
                toggleGrid(true);
            }

            const axesCheckbox = document.getElementById('showAxes');
            if (axesCheckbox && typeof visibility.showAxes === 'boolean') {
                axesCheckbox.checked = visibility.showAxes;
                toggleAxes(true);
            }

            const colorBarCheckbox = document.getElementById('showColorBar');
            if (colorBarCheckbox && typeof visibility.showColorBar === 'boolean') {
                colorBarCheckbox.checked = visibility.showColorBar;
                toggleColorBar();
            }
        }

    if (preset.autoMesh === false) {
        const voxelSizeInput = document.getElementById('voxelSize');
        const refinementSelect = document.getElementById('refinementLevel');
        if (voxelSizeInput && Number.isFinite(preset.voxelSize)) {
            voxelSizeInput.value = preset.voxelSize.toFixed(3);
        }
        if (refinementSelect && Number.isInteger(preset.refinement)) {
            refinementSelect.value = String(preset.refinement);
        }
        updateMeshPreview();
    }

    const analysis = preset.analysis || {};
    const analysisChecks = {
        optShrinkage: analysis.shrinkage,
        optHotspot: analysis.hotspot,
        optNiyama: analysis.niyama,
        optHotTear: analysis.hotTear,
        optSandBurnIn: analysis.sandBurnIn,
        optFractionLiquid: analysis.fractionLiquid
    };
    Object.entries(analysisChecks).forEach(([id, value]) => {
        const cb = document.getElementById(id);
        if (cb) cb.checked = Boolean(value);
    });

    if (Number.isFinite(preset.sandRestraint)) {
        const restraint = document.getElementById('sandRestraintIndex');
        if (restraint) {
            restraint.value = Math.round(preset.sandRestraint * 100);
            updateSandRestraintLabel();
        }
    }

    const resultDisplay = document.getElementById('resultDisplay');
    if (resultDisplay && preset.resultDisplay) {
        resultDisplay.value = preset.resultDisplay;
        updateResultDisplay();
    }

    if (typeof preset.threshold === 'number' && resultDisplay && preset.resultDisplay !== 'fraction_liquid') {
        const thresholdInput = document.getElementById('threshold');
        if (thresholdInput) {
            thresholdInput.value = Math.round(preset.threshold);
            updateThreshold();
        }
    }

    if (preset.fractionLiquid && preset.resultDisplay === 'fraction_liquid') {
        const timeSlider = document.getElementById('flTimeSlider');
        const thresholdSlider = document.getElementById('flThresholdSlider');
        if (timeSlider && Number.isFinite(preset.fractionLiquid.time)) {
            timeSlider.value = Math.round(preset.fractionLiquid.time);
            updateFractionLiquidTime();
        }
        if (thresholdSlider && Number.isFinite(preset.fractionLiquid.threshold)) {
            thresholdSlider.value = Math.round(preset.fractionLiquid.threshold);
            updateFractionLiquidThreshold();
        }
    }

    const opacityInput = document.getElementById('modelOpacity');
    if (opacityInput && Number.isFinite(preset.modelOpacity)) {
        opacityInput.value = Math.max(0, Math.min(100, Math.round(preset.modelOpacity)));
        updateModelOpacity();
    }

    const crossSectionCheckbox = document.getElementById('enableCrossSection');
    if (crossSectionCheckbox && typeof preset.crossSection === 'boolean') {
        crossSectionCheckbox.checked = preset.crossSection;
        toggleCrossSection();

        if (preset.crossSection) {
            const axisSelect = document.getElementById('crossSectionAxis');
            const crossSectionPos = document.getElementById('crossSectionPos');
            if (axisSelect && preset.crossSectionAxis) {
                axisSelect.value = preset.crossSectionAxis;
            }
            if (crossSectionPos && Number.isFinite(preset.crossSectionPos)) {
                crossSectionPos.value = Math.round(preset.crossSectionPos);
            }
            updateCrossSection();
        }
    }

    const summaryEl = document.getElementById('helpPresetSummary');
    if (summaryEl) {
        summaryEl.textContent = `${preset.label}: Applied. Review settings now and run simulation when ready.`;
    }

    const activeRow = document.querySelector(`[data-preset="${presetKey}"]`);
    if (activeRow) {
        document.querySelectorAll('.help-preset-row').forEach((row) => {
            row.classList.remove('help-preset-row-active');
        });
        activeRow.classList.add('help-preset-row-active');
    }

    setStatus(`Applied Quick Start preset: ${preset.label}`);
}

function renderHelpAlloyTable(materials) {
    const tableData = Array.isArray(materials) && materials.length > 0 ? materials : state.materials;
    const body = document.getElementById('helpAlloyTableBody');
    if (!body) return;

    if (!tableData || tableData.length === 0) {
        body.innerHTML = `
            <tr>
                <td colspan="6">Alloy data unavailable right now. Please refresh and try again.</td>
            </tr>
        `;
        renderHelpShrinkageConversionTable([]);
        return;
    }

    const descriptionByAlloy = {
        WCB: 'General-purpose cast steel; highest shrinkage in this set.',
        CF8: 'Stainless steel baseline; typically harder to feed than WCB/C958.',
        C958: 'Aluminum bronze; low shrinkage and strong directional feeding behavior.',
        C964: 'Copper-nickel; relatively low shrinkage and good fluidity.',
        CA6NM: 'Medium-alloy cast steel; moderate susceptibility and good overall casting balance.',
        MONEL400: 'Nickel-copper family with moderate shrinkage and good high-temperature stability.',
        IN625: 'Nickel alloy with tighter thermal response and lower modeled shrinkage in this set.'
    };

    const ordered = [...tableData].sort((a, b) => a.name.localeCompare(b.name));
    body.innerHTML = ordered.map((m) => {
        const shrinkage = Number(m.volumetric_shrinkage);
        const feeding = Number(m.feeding_multiplier);
        const niyama = Number(m.niyama_threshold);
        const liquidus = Number(m.liquidus_temp);
        const solidus = Number(m.solidus_temp);
        const freezeRange = liquidus - solidus;
        const description = descriptionByAlloy[m.name] || 'Model-dependent alloy behavior.';

        return `
            <tr>
                <td>${m.name}</td>
                <td>${m.full_name || '--'}</td>
                <td>${shrinkage.toFixed(2)}%</td>
                <td>${feeding.toFixed(1)}x</td>
                <td>${niyama.toFixed(2)}</td>
                <td>${description}</td>
            </tr>
            <tr class="help-subrow">
                <td colspan="6">
                    Liquidus ${liquidus.toFixed(0)}°C,
                    Solidus ${solidus.toFixed(0)}°C,
                    Freeze ${freezeRange.toFixed(0)}°C,
                    Volumetric shrinkage basis for defect scoring.
                </td>
            </tr>
        `;
    }).join('');
    renderHelpShrinkageConversionTable(ordered);
}

function renderHelpShrinkageConversionTable(materials) {
    const body = document.getElementById('helpShrinkageConversionTableBody');
    if (!body) {
        return;
    }

    if (!Array.isArray(materials) || materials.length === 0) {
        body.innerHTML = `
            <tr>
                <td colspan="4">No alloy data available.</td>
            </tr>
        `;
        return;
    }

    const ordered = [...materials].sort((a, b) => a.name.localeCompare(b.name));
    body.innerHTML = ordered.map((m) => {
        const volumetric = Number(m.volumetric_shrinkage);
        const linearEquivalent = getEquivalentLinearShrinkage(volumetric);
        const formula = Number.isFinite(volumetric)
            ? `(1 - (1 - v)^(1/3)), v=${(volumetric / 100).toFixed(3)}`
            : '--';

        return `
            <tr>
                <td>${m.name}</td>
                <td>${Number.isFinite(volumetric) ? `${volumetric.toFixed(2)}%` : '--'}</td>
                <td>${linearEquivalent}</td>
                <td>${formula}</td>
            </tr>
        `;
    }).join('');
}

async function loadSimulationById(simId) {
    try {
        const response = await fetch(`/api/simulations/${simId}`);
        const data = await response.json();
        
        // Clear current state
        for (const mesh of Object.values(state.partMeshes)) {
            state.scene.remove(mesh);
        }
        state.partMeshes = {};
        state.parts = [];
        
        // Set simulation settings
        state.simulationId = simId;
        document.getElementById('alloySelect').value = data.simulation.alloy;
        document.getElementById('voxelSize').value = data.simulation.voxel_size;
        document.getElementById('refinementLevel').value = data.simulation.refinement_level;
        
        // Load analysis options
        if (data.simulation.analysis_options) {
            const opts = JSON.parse(data.simulation.analysis_options);
            document.getElementById('optShrinkage').checked = opts.shrinkage !== false;
            document.getElementById('optHotspot').checked = opts.hotspot !== false;
            document.getElementById('optNiyama').checked = opts.niyama === true;
            document.getElementById('optHotTear').checked = opts.hot_tear === true;
            document.getElementById('optSandBurnIn').checked = opts.sand_burn_in === true;
            document.getElementById('optFractionLiquid').checked = opts.fraction_liquid === true;
            const rawRestraint = Number(opts.sand_restraint_index);
            const clampedRestraint = Number.isFinite(rawRestraint)
                ? Math.max(0, Math.min(1, rawRestraint))
                : 0.5;
            document.getElementById('sandRestraintIndex').value = Math.round(clampedRestraint * 100);
            updateSandRestraintLabel();
        }
        
        // Load parts
        for (const partData of data.parts) {
            // Re-upload the file to get mesh data
            // In a real implementation, mesh data would be cached
            const part = {
                id: partData.id,
                name: partData.name,
                type: partData.part_type,
                filepath: partData.filepath,
                position: { x: partData.position_x, y: partData.position_y, z: partData.position_z },
                scale: partData.scale,
                meshData: null
            };
            
            // Try to load mesh from file
            try {
                const loader = new THREE.STLLoader();
                loader.load(partData.filepath, (geometry) => {
                    // Create mesh from geometry
                    geometry.computeVertexNormals();
                    
                    const colors = { cast: 0xb0bec5, riser: 0xf59e0b, chill: 0x3b82f6 };
                    const material = new THREE.MeshPhongMaterial({
                        color: colors[part.type],
                        transparent: true,
                        opacity: 0.7,
                        side: THREE.DoubleSide
                    });
                    
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set(part.position.x, part.position.y, part.position.z);
                    mesh.scale.setScalar(part.scale);
                    
                    state.scene.add(mesh);
                    state.partMeshes[part.id] = mesh;
                });
            } catch (e) {
                console.error('Failed to load part mesh:', e);
            }
            
            state.parts.push(part);
        }
        
        // Load results if completed
        if (data.simulation.status === 'completed') {
            loadResults();
        }
        
        updatePartsList();
        updateMaterialInfo();
        closeLoadDialog();
        setStatus('Loaded: ' + data.simulation.name);
        
    } catch (error) {
        console.error('Load simulation error:', error);
        setStatus('Failed to load simulation');
    }
}

// ============================================
// UI Utilities
// ============================================

function togglePanel(header) {
    const panel = header.parentElement;
    panel.classList.toggle('collapsed');
}

function setStatus(text) {
    document.getElementById('statusText').textContent = text;
}

function updateModelInfo() {
    if (state.parts.length === 0) {
        document.getElementById('modelInfo').textContent = 'No model loaded';
        return;
    }
    
    const part = state.parts[0];
    if (part.meshData) {
        const v = part.meshData.vertex_count.toLocaleString();
        const f = part.meshData.face_count.toLocaleString();
        document.getElementById('modelInfo').textContent = `${state.parts.length} part(s) | ${v} vertices | ${f} faces`;
    }
}

function toggleDarkMode() {
    document.body.classList.toggle('light-mode');
    
    // Update 3D scene background
    if (state.scene) {
        if (document.body.classList.contains('light-mode')) {
            state.scene.background = new THREE.Color(0xf0f0f5);
        } else {
            state.scene.background = new THREE.Color(0x0a0a12);
        }
    }
    
    // Update button icon
    const btn = document.querySelector('[onclick="toggleDarkMode()"] i');
    if (btn) {
        btn.className = document.body.classList.contains('light-mode') ? 'fas fa-sun' : 'fas fa-moon';
    }
}
