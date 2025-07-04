#!/usr/bin/env python3
"""
Test script for the two-phase simplex method implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import LPSolver

def test_two_phase_basic():
    """Test basic two-phase method functionality"""
    print("Testing Two-Phase Method...")
    print("=" * 50)
    
    # Test case 1: Simple maximization problem with artificial variables needed
    print("\nTest 1: Maximization with equality constraint")
    print("Maximize: 2x1 + 3x2")
    print("Subject to:")
    print("  x1 + x2 = 4")
    print("  2x1 + x2 <= 6")
    print("  x1, x2 >= 0")
    
    try:
        solver = LPSolver(
            obj_coeffs=[2, 3],
            cons_coeffs=[[1, 1], [2, 1]],
            cons_signs=["=", "≤"],
            cons_values=[4, 6],
            obj_type="maximize"
        )
        
        result = solver.solve_two_phase()
        print(f"Status: {result['status']}")
        if result['status'] == 'Optimal':
            print(f"Optimal solution: {result['optimal_solution']}")
            print(f"Optimal value: {result['optimal_value']:.2f}")
            print(f"Phase 1 iterations: {len(result['phase1_iterations'])}")
            print(f"Phase 2 iterations: {len(result['phase2_iterations'])}")
        print("✓ Test 1 passed!")
        
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    # Test case 2: Problem with >= constraint
    print("\nTest 2: Maximization with >= constraint")
    print("Maximize: x1 + 2x2")
    print("Subject to:")
    print("  x1 + x2 >= 3")
    print("  2x1 + x2 <= 6")
    print("  x1, x2 >= 0")
    
    try:
        solver = LPSolver(
            obj_coeffs=[1, 2],
            cons_coeffs=[[1, 1], [2, 1]],
            cons_signs=["≥", "≤"],
            cons_values=[3, 6],
            obj_type="maximize"
        )
        
        result = solver.solve_two_phase()
        print(f"Status: {result['status']}")
        if result['status'] == 'Optimal':
            print(f"Optimal solution: {result['optimal_solution']}")
            print(f"Optimal value: {result['optimal_value']:.2f}")
        print("✓ Test 2 passed!")
        
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
    
    # Test case 3: Minimization problem
    print("\nTest 3: Minimization problem")
    print("Minimize: 2x1 + x2")
    print("Subject to:")
    print("  x1 + x2 >= 3")
    print("  x1 + 2x2 >= 4")
    print("  x1, x2 >= 0")
    
    try:
        solver = LPSolver(
            obj_coeffs=[2, 1],
            cons_coeffs=[[1, 1], [1, 2]],
            cons_signs=["≥", "≥"],
            cons_values=[3, 4],
            obj_type="minimize"
        )
        
        result = solver.solve_two_phase()
        print(f"Status: {result['status']}")
        if result['status'] == 'Optimal':
            print(f"Optimal solution: {result['optimal_solution']}")
            print(f"Optimal value: {result['optimal_value']:.2f}")
        print("✓ Test 3 passed!")
        
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")

if __name__ == "__main__":
    test_two_phase_basic()
    print("\n" + "=" * 50)
    print("Testing complete!")
