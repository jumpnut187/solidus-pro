"""
Simulation Results Comparison and Save/Load Module

Quick Win #4: Save/load simulation results to compare before/after riser changes.

Features:
- Save simulation results to JSON/pickle files
- Load and compare multiple simulation results
- Generate comparison reports
- Export comparison data for visualization

Usage:
    from core.results_manager import ResultsManager
    
    # Save current simulation
    manager = ResultsManager()
    manager.save_results(simulation.export_results(), "baseline.json")
    
    # Load and compare
    baseline = manager.load_results("baseline.json")
    optimized = manager.load_results("optimized.json")
    comparison = manager.compare_results(baseline, optimized)

Author: GitHub Copilot
Date: December 2025
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ResultsSummary:
    """Summary of key simulation metrics for comparison"""
    name: str
    timestamp: str
    alloy: str
    voxel_count: int
    simulation_time: float
    
    # Shrinkage metrics
    shrinkage_volume: float = 0.0
    shrinkage_percentage: float = 0.0
    affected_voxels: int = 0
    
    # Hot spot metrics  
    hotspot_count: int = 0
    
    # Niyama metrics
    niyama_defect_count: int = 0
    
    # Feeding metrics
    feeding_blocked_count: int = 0
    fed_fraction: float = 0.0
    
    # Stress metrics
    max_stress_mpa: float = 0.0
    
    # Microstructure metrics
    avg_sdas_um: float = 0.0


@dataclass
class ComparisonResult:
    """Results of comparing two simulations"""
    baseline_name: str
    comparison_name: str
    
    # Shrinkage changes
    shrinkage_volume_change: float = 0.0
    shrinkage_volume_change_pct: float = 0.0
    
    # Hot spot changes
    hotspot_count_change: int = 0
    
    # Feeding changes
    feeding_improvement_pct: float = 0.0
    
    # Overall assessment
    improved: bool = False
    improvement_score: float = 0.0  # 0-100, higher = better


class ResultsManager:
    """
    Manages saving, loading, and comparing simulation results.
    """
    
    def __init__(self, results_dir: str = None):
        """
        Initialize results manager.
        
        Args:
            results_dir: Directory to store results (default: data/results/)
        """
        if results_dir is None:
            # Default to data/results/ in project root
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(module_dir, 'data', 'results')
        
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Cache of loaded results
        self._cache: Dict[str, Dict] = {}
    
    def save_results(self, 
                     results: Dict, 
                     name: str,
                     include_arrays: bool = False) -> str:
        """
        Save simulation results to file.
        
        Args:
            results: Results dictionary from simulation.export_results()
            name: Name for the saved results (without extension)
            include_arrays: Whether to include full numpy arrays (larger file)
            
        Returns:
            Path to saved file
        """
        # Create summary
        summary = self._create_summary(results, name)
        
        # Prepare data for saving
        save_data = {
            'summary': asdict(summary),
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }
        
        # Copy scalar values and small arrays
        for key, value in results.items():
            if isinstance(value, (int, float, str, bool)):
                save_data['results'][key] = value
            elif isinstance(value, dict):
                save_data['results'][key] = value
            elif isinstance(value, np.ndarray):
                if include_arrays:
                    # Convert to list for JSON serialization
                    if value.size < 100000:  # Only small arrays
                        save_data['results'][key] = value.tolist()
                    else:
                        save_data['results'][key] = f"<array: {value.shape}>"
                else:
                    # Just save shape info
                    save_data['results'][key] = f"<array: {value.shape}>"
        
        # Save to JSON
        filepath = os.path.join(self.results_dir, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Cache it
        self._cache[name] = save_data
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def save_full_results(self, results: Dict, name: str) -> str:
        """
        Save full simulation results including all arrays using pickle.
        
        Args:
            results: Full results dictionary
            name: Name for the saved results
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.results_dir, f"{name}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Full results saved to: {filepath}")
        return filepath
    
    def load_results(self, name: str) -> Dict:
        """
        Load simulation results from file.
        
        Args:
            name: Name of saved results (with or without extension)
            
        Returns:
            Results dictionary
        """
        # Check cache first
        base_name = name.replace('.json', '').replace('.pkl', '')
        if base_name in self._cache:
            return self._cache[base_name]
        
        # Try JSON first
        json_path = os.path.join(self.results_dir, f"{base_name}.json")
        pkl_path = os.path.join(self.results_dir, f"{base_name}.pkl")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            self._cache[base_name] = data
            return data
        elif os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            self._cache[base_name] = {'results': data}
            return self._cache[base_name]
        else:
            raise FileNotFoundError(f"Results not found: {name}")
    
    def list_saved_results(self) -> List[str]:
        """
        List all saved result files.
        
        Returns:
            List of result names (without extensions)
        """
        results = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json') or filename.endswith('.pkl'):
                name = filename.rsplit('.', 1)[0]
                if name not in results:
                    results.append(name)
        return sorted(results)
    
    def compare_results(self,
                        baseline: Dict,
                        comparison: Dict,
                        baseline_name: str = "Baseline",
                        comparison_name: str = "Comparison") -> ComparisonResult:
        """
        Compare two simulation results.
        
        Args:
            baseline: Baseline results dictionary
            comparison: Comparison results dictionary
            baseline_name: Name for baseline
            comparison_name: Name for comparison
            
        Returns:
            ComparisonResult with changes
        """
        # Extract results sub-dict if present
        base_r = baseline.get('results', baseline)
        comp_r = comparison.get('results', comparison)
        
        # Calculate changes
        base_shrink = base_r.get('shrinkage_volume', 0)
        comp_shrink = comp_r.get('shrinkage_volume', 0)
        
        shrink_change = comp_shrink - base_shrink
        shrink_change_pct = (shrink_change / base_shrink * 100) if base_shrink > 0 else 0
        
        base_hotspot = base_r.get('hotspot_count', 0)
        comp_hotspot = comp_r.get('hotspot_count', 0)
        hotspot_change = comp_hotspot - base_hotspot
        
        base_fed = base_r.get('fed_fraction', 0.5)
        comp_fed = comp_r.get('fed_fraction', 0.5)
        feeding_improvement = (comp_fed - base_fed) * 100
        
        # Calculate improvement score
        # Negative shrinkage change = improvement
        # Negative hotspot change = improvement
        # Positive feeding change = improvement
        score = 0
        if base_shrink > 0:
            score += max(0, -shrink_change_pct)  # Up to ~100 for major reduction
        if base_hotspot > 0:
            score += max(0, -hotspot_change / base_hotspot * 30)  # Up to 30 points
        score += max(0, feeding_improvement * 0.5)  # Up to 50 points
        
        improved = shrink_change < 0 or hotspot_change < 0 or feeding_improvement > 0
        
        return ComparisonResult(
            baseline_name=baseline_name,
            comparison_name=comparison_name,
            shrinkage_volume_change=shrink_change,
            shrinkage_volume_change_pct=shrink_change_pct,
            hotspot_count_change=hotspot_change,
            feeding_improvement_pct=feeding_improvement,
            improved=improved,
            improvement_score=min(100, score)
        )
    
    def generate_comparison_report(self,
                                    comparison: ComparisonResult) -> str:
        """
        Generate a text report comparing two simulations.
        
        Args:
            comparison: ComparisonResult object
            
        Returns:
            Formatted report string
        """
        lines = [
            "="*60,
            "SIMULATION COMPARISON REPORT",
            "="*60,
            "",
            f"Baseline: {comparison.baseline_name}",
            f"Comparison: {comparison.comparison_name}",
            "",
            "SHRINKAGE:",
            f"  Volume change: {comparison.shrinkage_volume_change:+.4f} in³",
            f"  Percentage change: {comparison.shrinkage_volume_change_pct:+.1f}%",
            "",
            "HOT SPOTS:",
            f"  Count change: {comparison.hotspot_count_change:+d}",
            "",
            "FEEDING:",
            f"  Coverage improvement: {comparison.feeding_improvement_pct:+.1f}%",
            "",
            "="*60,
            "OVERALL ASSESSMENT:",
            f"  Improved: {'YES ✓' if comparison.improved else 'NO ✗'}",
            f"  Improvement Score: {comparison.improvement_score:.0f}/100",
            "="*60
        ]
        
        if comparison.improved:
            lines.append("")
            lines.append("Recommendations:")
            if comparison.shrinkage_volume_change_pct < -10:
                lines.append("  ✓ Significant shrinkage reduction achieved")
            if comparison.hotspot_count_change < 0:
                lines.append("  ✓ Hot spot reduction achieved")
            if comparison.feeding_improvement_pct > 5:
                lines.append("  ✓ Feeding coverage improved")
        else:
            lines.append("")
            lines.append("Suggestions:")
            if comparison.shrinkage_volume_change_pct > 0:
                lines.append("  • Consider adding/enlarging risers")
            if comparison.hotspot_count_change > 0:
                lines.append("  • Add chills to reduce hot spots")
            if comparison.feeding_improvement_pct < 0:
                lines.append("  • Review riser placement for better coverage")
        
        return "\n".join(lines)
    
    def _create_summary(self, results: Dict, name: str) -> ResultsSummary:
        """Create a ResultsSummary from full results"""
        return ResultsSummary(
            name=name,
            timestamp=datetime.now().isoformat(),
            alloy=results.get('alloy', 'Unknown'),
            voxel_count=results.get('voxel_count', 0),
            simulation_time=results.get('simulation_time', 0),
            shrinkage_volume=results.get('shrinkage_volume', 0),
            shrinkage_percentage=results.get('shrinkage_percentage', 0),
            affected_voxels=results.get('affected_voxels', 0),
            hotspot_count=results.get('hotspot_count', 0),
            niyama_defect_count=results.get('niyama_defect_count', 0),
            feeding_blocked_count=results.get('feeding_blocked_count', 0),
            fed_fraction=results.get('fed_fraction', 0),
            max_stress_mpa=results.get('max_stress_mpa', 0),
            avg_sdas_um=results.get('avg_sdas_um', 0)
        )
    
    def get_summary(self, name: str) -> ResultsSummary:
        """
        Get summary for a saved result.
        
        Args:
            name: Name of saved results
            
        Returns:
            ResultsSummary object
        """
        data = self.load_results(name)
        if 'summary' in data:
            return ResultsSummary(**data['summary'])
        else:
            return self._create_summary(data.get('results', data), name)


# Test
if __name__ == '__main__':
    print("Results Manager Module")
    print("="*60)
    print("Features:")
    print("  - Save simulation results to JSON/pickle")
    print("  - Load and compare results")
    print("  - Generate comparison reports")
    print("\nUsage:")
    print("  from core.results_manager import ResultsManager")
    print("  manager = ResultsManager()")
    print("  manager.save_results(sim.export_results(), 'baseline')")
