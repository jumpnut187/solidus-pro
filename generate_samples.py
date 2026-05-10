"""
Generate sample STL files for testing
Creates simple casting geometries in inches
"""

import numpy as np
from stl import mesh
import os

def create_simple_block(filename='sample_block.stl', size_inches=2.0):
    """
    Create a simple cubic block
    
    Args:
        filename: Output filename
        size_inches: Size in inches
    """
    # Define the 8 vertices of the cube
    vertices = np.array([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1]
    ]) * (size_inches / 2)
    
    # Define the 12 triangles (2 per face)
    faces = np.array([
        [0,3,1], [1,3,2],  # Bottom
        [4,5,7], [5,6,7],  # Top
        [0,1,4], [1,5,4],  # Front
        [2,3,6], [3,7,6],  # Back
        [0,4,3], [3,4,7],  # Left
        [1,2,5], [2,6,5],  # Right
    ])
    
    # Create mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[face[j]]
    
    cube.save(filename)
    print(f"✓ Created: {filename}")
    print(f"  Size: {size_inches}\" cube")


def create_thick_thin_test(filename='sample_thick_thin.stl'):
    """
    Create a test piece with thick and thin sections
    Good for testing feeding distance
    """
    # Thick section (3" x 3" x 2")
    thick_vertices = np.array([
        [-1.5, -1.5, 0], [1.5, -1.5, 0], [1.5, 1.5, 0], [-1.5, 1.5, 0],
        [-1.5, -1.5, 2], [1.5, -1.5, 2], [1.5, 1.5, 2], [-1.5, 1.5, 2]
    ])
    
    # Thin section (1" x 1" x 3")
    thin_vertices = np.array([
        [-0.5, -0.5, 2], [0.5, -0.5, 2], [0.5, 0.5, 2], [-0.5, 0.5, 2],
        [-0.5, -0.5, 5], [0.5, -0.5, 5], [0.5, 0.5, 5], [-0.5, 0.5, 5]
    ])
    
    vertices = np.vstack([thick_vertices, thin_vertices])
    
    # Create faces
    def box_faces(offset):
        return np.array([
            [0,3,1], [1,3,2],  # Bottom
            [4,5,7], [5,6,7],  # Top
            [0,1,4], [1,5,4],  # Front
            [2,3,6], [3,7,6],  # Back
            [0,4,3], [3,4,7],  # Left
            [1,2,5], [2,6,5],  # Right
        ]) + offset
    
    faces = np.vstack([box_faces(0), box_faces(8)])
    
    # Create mesh
    test_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            test_mesh.vectors[i][j] = vertices[face[j]]
    
    test_mesh.save(filename)
    print(f"✓ Created: {filename}")
    print(f"  Thick section (3\" cube) + thin section (1\" x 3\" tall)")


def create_l_bracket(filename='sample_l_bracket.stl'):
    """
    Create an L-shaped bracket
    Tests corner feeding and edge effects
    """
    # Base (4" x 2" x 0.5")
    base_h = 0.5
    base_w = 2.0
    base_l = 4.0
    
    # Vertical (2" x 3" x 0.5")
    vert_h = 3.0
    vert_w = 2.0
    vert_t = 0.5
    
    # Base vertices
    base_v = np.array([
        [0, 0, 0], [base_l, 0, 0], [base_l, base_w, 0], [0, base_w, 0],
        [0, 0, base_h], [base_l, 0, base_h], [base_l, base_w, base_h], [0, base_w, base_h]
    ])
    
    # Vertical part vertices
    vert_v = np.array([
        [0, 0, base_h], [vert_t, 0, base_h], [vert_t, vert_w, base_h], [0, vert_w, base_h],
        [0, 0, base_h + vert_h], [vert_t, 0, base_h + vert_h], 
        [vert_t, vert_w, base_h + vert_h], [0, vert_w, base_h + vert_h]
    ])
    
    vertices = np.vstack([base_v, vert_v])
    
    # Create faces
    def box_faces(offset):
        return np.array([
            [0,3,1], [1,3,2],  # Bottom
            [4,5,7], [5,6,7],  # Top
            [0,1,4], [1,5,4],  # Front
            [2,3,6], [3,7,6],  # Back
            [0,4,3], [3,4,7],  # Left
            [1,2,5], [2,6,5],  # Right
        ]) + offset
    
    faces = np.vstack([box_faces(0), box_faces(8)])
    
    # Create mesh
    bracket_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            bracket_mesh.vectors[i][j] = vertices[face[j]]
    
    bracket_mesh.save(filename)
    print(f"✓ Created: {filename}")
    print(f"  L-bracket: 4\" x 2\" x 0.5\" base + 2\" x 3\" vertical")


def create_cylinder(filename='sample_cylinder.stl', 
                   diameter_inches=2.0, 
                   height_inches=3.0, 
                   segments=32):
    """
    Create a cylindrical casting
    
    Args:
        filename: Output filename
        diameter_inches: Diameter in inches
        height_inches: Height in inches
        segments: Number of segments around circle
    """
    radius = diameter_inches / 2
    
    # Generate vertices
    vertices = []
    
    # Bottom circle
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, 0])
    
    # Top circle
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, height_inches])
    
    # Center points
    vertices.append([0, 0, 0])  # Bottom center
    vertices.append([0, 0, height_inches])  # Top center
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    
    # Side faces
    for i in range(segments):
        next_i = (i + 1) % segments
        # Triangle 1
        faces.append([i, next_i, i + segments])
        # Triangle 2
        faces.append([next_i, next_i + segments, i + segments])
    
    # Bottom cap
    bottom_center = len(vertices) - 2
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([bottom_center, next_i, i])
    
    # Top cap
    top_center = len(vertices) - 1
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([top_center, i + segments, next_i + segments])
    
    faces = np.array(faces)
    
    # Create mesh
    cyl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cyl_mesh.vectors[i][j] = vertices[face[j]]
    
    cyl_mesh.save(filename)
    print(f"✓ Created: {filename}")
    print(f"  Cylinder: {diameter_inches}\" diameter x {height_inches}\" height")


def main():
    """Generate all sample STL files"""
    print("\n" + "="*60)
    print("Generating Sample STL Files (in inches)")
    print("="*60)
    print()
    
    # Create samples directory
    samples_dir = 'sample_stl_files'
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        print(f"Created directory: {samples_dir}/")
        print()
    
    os.chdir(samples_dir)
    
    # Generate various test geometries
    create_simple_block('block_2inch.stl', size_inches=2.0)
    create_thick_thin_test('thick_thin_test.stl')
    create_l_bracket('l_bracket.stl')
    create_cylinder('cylinder.stl', diameter_inches=2.0, height_inches=3.0)
    create_simple_block('small_block.stl', size_inches=1.0)
    
    print()
    print("="*60)
    print(f"✓ Generated 5 sample STL files in '{samples_dir}/'")
    print()
    print("Recommendations:")
    print("  • Start with: block_2inch.stl (simple, fast)")
    print("  • Test feeding: thick_thin_test.stl (shows hot spots)")
    print("  • Test corners: l_bracket.stl (edge effects)")
    print("  • Cylindrical: cylinder.stl (radial solidification)")
    print("="*60)
    print()


if __name__ == '__main__':
    main()
