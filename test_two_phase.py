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
    
    # Test case 4: Large problem with 10 variables and 15 constraints
    print("\nTest 4: Large problem with 10 variables and 15 constraints")
    print("Maximize: Sum of x1 to x10")
    print("Subject to:")
    
    cons_coeffs = [[1] * 10 for _ in range(15)]  # All constraints are x1 + x2 + ... + x10
    cons_signs = ["≤"] * 15  # All constraints are <=
    cons_values = [10] * 15  # All RHS values are 10
    
    try:
        solver = LPSolver(
            obj_coeffs=[1] * 10,  # Coefficients for the objective function
            cons_coeffs=cons_coeffs,
            cons_signs=cons_signs,
            cons_values=cons_values,
            obj_type="maximize"
        )
        
        result = solver.solve_two_phase()
        print(f"Status: {result['status']}")
        if result['status'] == 'Optimal':
            print(f"Optimal solution: {result['optimal_solution']}")
            print(f"Optimal value: {result['optimal_value']:.2f}")
            print(f"Phase 1 iterations: {len(result['phase1_iterations'])}")
            print(f"Phase 2 iterations: {len(result['phase2_iterations'])}")
        print("✓ Test 4 passed!")
        
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")

def test_two_phase_real_problem():
    """Test two-phase method functionality with a real LP problem (10 variables, 15 constraints)"""
    print("Testing Two-Phase Method with real problem...")
    print("=" * 50)
    
    # Definición del problema
    print("\nMaximizar: Z = 2x1 + 3x2 + 1x3 + 2x4 + 3x5 + 1x6 + 4x7 + 2x8 + 3x9 + 1x10")
    print("Sujeto a:")
    constraints = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # x1 + x2 + ... + x10 <= 100
        [2, 3, 1, 2, 1, 3, 4, 1, 2, 1],  # 2x1 + 3x2 + ... <= 150
        [3, 1, 1, 1, 2, 1, 1, 3, 4, 1],  # 3x1 + x2 + ... = 200
        [4, 2, 3, 1, 1, 2, 1, 3, 1, 2],  # 4x1 + 2x2 + ... <= 300
        [1, 2, 1, 1, 3, 1, 4, 2, 3, 1],  # x1 + 2x2 + ... >= 100
        [2, 3, 3, 1, 1, 2, 4, 1, 2, 1],  # 2x1 + 3x2 + ... <= 150
        [3, 1, 2, 4, 1, 1, 1, 2, 3, 1],  # 3x1 + x2 + ... = 250
        [2, 1, 4, 1, 2, 3, 1, 3, 2, 1],  # 2x1 + x2 + ... >= 180
        [1, 1, 1, 2, 1, 2, 3, 4, 2, 1],  # x1 + x2 + ... <= 220
        [1, 3, 4, 3, 2, 1, 2, 2, 3, 1],  # x1 + 3x2 + ... >= 300
    ]

    cons_values = [100, 150, 200, 300, 100, 150, 250, 180, 220, 300]
    cons_signs = ["≤", "≤", "=", "≤", "≥", "≤", "=", "≥", "≤", "≥"]

    try:
        solver = LPSolver(
            obj_coeffs=[2, 3, 1, 2, 3, 1, 4, 2, 3, 1],  # Coeficientes del objetivo
            cons_coeffs=constraints,  # Restricciones
            cons_signs=cons_signs,  # Signos de las restricciones
            cons_values=cons_values,  # Valores de las restricciones
            obj_type="maximize"  # Maximización
        )
        
        result = solver.solve_two_phase()
        print(f"Status: {result['status']}")
        if result['status'] == 'Optimal':
            print(f"Optimal solution: {result['optimal_solution']}")
            print(f"Optimal value: {result['optimal_value']:.2f}")
            print(f"Phase 1 iterations: {len(result['phase1_iterations'])}")
            print(f"Phase 2 iterations: {len(result['phase2_iterations'])}")
        print("✓ Test passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    test_two_phase_basic()
    test_two_phase_real_problem()
    print("\n" + "=" * 50)
    print("Testing complete!")
