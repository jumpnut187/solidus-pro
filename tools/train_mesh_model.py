#!/usr/bin/env python3
"""
Train mesh recommendation model from logged simulation data.
Run this after accumulating 20+ simulation runs in data/mesh_runs.csv
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor
    import joblib
except ImportError:
    print("Error: scikit-learn not installed. Install with:")
    print("  pip3 install scikit-learn")
    sys.exit(1)

# Paths - relative to project root (one level up from tools/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_PATH = os.path.join(DATA_DIR, "mesh_runs.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "mesh_recommender.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "mesh_scaler.joblib")

# Outlier filtering thresholds (based on analysis of mesh_runs.csv)
MAX_MESH_GENERATION_TIME = 120.0  # Exclude runs taking > 2 minutes for mesh
MAX_SIMULATION_TIME = 90.0  # Exclude runs taking > 1.5 minutes for simulation
MIN_VOXEL_COUNT = 1000  # Exclude runs with very few voxels
MAX_VOXEL_COUNT = 600_000  # Exclude runs that exceeded reasonable limits


def load_and_prepare_data(csv_path):
    """Load CSV and prepare features/targets with outlier filtering."""
    if not os.path.exists(csv_path):
        print(f"Error: No training data found at {csv_path}")
        print("Run simulations first to collect training data.")
        return None, None, None
    
    # Read with error handling for malformed rows (some error messages contain commas)
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except TypeError:
        # Older pandas versions use error_bad_lines
        df = pd.read_csv(csv_path, error_bad_lines=False)
    
    print(f"Loaded {len(df)} simulation runs from {csv_path}")
    
    # Filter successful runs only
    initial_count = len(df)
    df = df[df['success'] == True].copy()
    print(f"  {len(df)} successful runs (filtered {initial_count - len(df)} failed)")
    
    # Apply outlier filtering
    pre_filter = len(df)
    
    # Filter by mesh generation time
    if 'mesh_generation_time_s' in df.columns:
        df = df[df['mesh_generation_time_s'].fillna(0) <= MAX_MESH_GENERATION_TIME]
    
    # Filter by simulation time
    if 'simulation_time_s' in df.columns:
        df = df[df['simulation_time_s'].fillna(0) <= MAX_SIMULATION_TIME]
    
    # Filter by voxel count range
    if 'voxel_count' in df.columns:
        df = df[(df['voxel_count'] >= MIN_VOXEL_COUNT) & 
                (df['voxel_count'] <= MAX_VOXEL_COUNT)]
    
    # Filter out runs with zero or null values in critical columns
    critical_cols = ['base_voxel_size', 'voxel_count', 'mesh_max_dim']
    for col in critical_cols:
        if col in df.columns:
            df = df[df[col].notna() & (df[col] > 0)]
    
    print(f"  {len(df)} runs after outlier filtering (removed {pre_filter - len(df)} outliers)")
    
    if len(df) < 10:
        print("Error: Need at least 10 valid runs to train a model.")
        print("  Tip: Run more simulations with reasonable settings")
        return None, None, None
    
    # Feature columns (must match recommend_mesh_settings feature_vector)
    feature_cols = [
        'mesh_max_dim',
        'mesh_surface_to_volume',
        'mesh_complexity',
        'mesh_fill_ratio',
        'mesh_triangle_count',
        'hw_physical_cores',
        'hw_available_memory_gb',
    ]
    
    # Check if all features exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing feature columns: {missing}")
        return None, None, None
    
    X = df[feature_cols].values
    
    # Target: predict voxel_size and refinement_level
    # We want to learn what settings led to good performance
    y_voxel_size = df['base_voxel_size'].values
    y_refinement = df['refinement_level'].values
    
    # Calculate a "quality score" based on performance
    # This weights the training toward settings that produced good results
    # Lower time = better, reasonable voxel count = better
    sim_times = df['simulation_time_s'].fillna(60).values
    mesh_times = df['mesh_generation_time_s'].fillna(30).values
    total_times = sim_times + mesh_times
    
    # Time score: exponential decay so faster is much better
    time_scores = np.exp(-total_times / 60.0)  # 60s half-life
    
    # Voxel count score: prefer 50k-300k range (sweet spot for quality/speed)
    voxel_counts = df['voxel_count'].values
    voxel_scores = np.where(
        (voxel_counts >= 50_000) & (voxel_counts <= 300_000),
        1.0,
        np.where(
            (voxel_counts >= 30_000) & (voxel_counts <= 400_000),
            0.7,
            0.4
        )
    )
    
    # Combined quality score
    quality_scores = time_scores * voxel_scores
    
    # Normalize weights
    sample_weights = quality_scores / quality_scores.sum()
    
    y = np.column_stack([y_voxel_size, y_refinement])
    
    # Print summary statistics
    print(f"\n  Data summary:")
    print(f"    Voxel size range: {y_voxel_size.min():.3f}\" - {y_voxel_size.max():.3f}\"")
    print(f"    Refinement range: {int(y_refinement.min())} - {int(y_refinement.max())}")
    print(f"    Max dim range: {df['mesh_max_dim'].min():.1f}\" - {df['mesh_max_dim'].max():.1f}\"")
    print(f"    Avg simulation time: {sim_times.mean():.1f}s")
    
    return X, y, sample_weights


def train_model(X, y, sample_weights):
    """Train gradient boosting regressor."""
    print("\nTraining model...")
    
    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multi-output model
    base_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    model = MultiOutputRegressor(base_model)
    model.fit(X_train_scaled, y_train, sample_weight=w_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\n  Training R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n  Voxel size MAE: {np.mean(np.abs(y_pred[:, 0] - y_test[:, 0])):.4f}\"")
    print(f"  Refinement MAE: {np.mean(np.abs(y_pred[:, 1] - y_test[:, 1])):.2f}")
    
    return model, scaler


def save_model(model, scaler, model_path, scaler_path):
    """Save trained model and scaler."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")
    print("\nThe ML mesh recommendation will now use this trained model!")


def main():
    print("="*70)
    print("MESH RECOMMENDATION MODEL TRAINER")
    print("="*70)
    
    # Load data
    X, y, weights = load_and_prepare_data(DATASET_PATH)
    if X is None:
        return 1
    
    # Train
    model, scaler = train_model(X, y, weights)
    
    # Save
    save_model(model, scaler, MODEL_PATH, SCALER_PATH)
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Run more simulations to continue improving the model")
    print("  2. Re-run this script periodically to retrain with new data")
    print("  3. The 'Apply Machine Learning Mesh' button will now use this model")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
