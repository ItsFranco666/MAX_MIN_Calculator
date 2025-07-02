import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import fractions
from fractions import Fraction
import copy

class LinearProgrammingCalculator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Linear Programming Calculator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables to store problem data
        self.num_variables = 0
        self.num_constraints = 0
        self.is_maximization = True
        self.objective_coeffs = []
        self.constraint_coeffs = []
        self.constraint_rhs = []
        self.constraint_types = []
        self.show_fractions = False
        self.solution_steps = []
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.show_initial_screen()
    
    def show_initial_screen(self):
        self.clear_frame()
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Linear Programming Calculator", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Input frame
        input_frame = ttk.LabelFrame(self.main_frame, text="Problem Setup", padding="20")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Number of variables
        ttk.Label(input_frame, text="Number of Variables (1-10):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.var_entry = ttk.Entry(input_frame, width=10)
        self.var_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Number of constraints
        ttk.Label(input_frame, text="Number of Constraints (1-15):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.const_entry = ttk.Entry(input_frame, width=10)
        self.const_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Continue button
        continue_btn = ttk.Button(input_frame, text="Continue", command=self.validate_initial_input)
        continue_btn.grid(row=2, column=0, columnspan=2, pady=20)
    
    def validate_initial_input(self):
        try:
            num_vars = int(self.var_entry.get())
            num_consts = int(self.const_entry.get())
            
            if num_vars < 1 or num_vars > 10:
                messagebox.showerror("Invalid Input", "Number of variables must be between 1 and 10.")
                return
            
            if num_consts < 1 or num_consts > 15:
                messagebox.showerror("Invalid Input", "Number of constraints must be between 1 and 15.")
                return
            
            self.num_variables = num_vars
            self.num_constraints = num_consts
            self.show_problem_input_screen()
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integer values.")
    
    def show_problem_input_screen(self):
        self.clear_frame()
        
        # Create scrollable frame
        canvas = tk.Canvas(self.main_frame)
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="Problem Input", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=4, pady=10)
        
        # Objective function frame
        obj_frame = ttk.LabelFrame(scrollable_frame, text="Objective Function", padding="10")
        obj_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10, padx=10)
        
        # Maximization/Minimization
        self.obj_type = tk.StringVar(value="max")
        ttk.Radiobutton(obj_frame, text="Maximize", variable=self.obj_type, 
                       value="max").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(obj_frame, text="Minimize", variable=self.obj_type, 
                       value="min").grid(row=0, column=1, sticky=tk.W)
        
        # Objective coefficients
        ttk.Label(obj_frame, text="Z = ").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.obj_entries = []
        for i in range(self.num_variables):
            entry = ttk.Entry(obj_frame, width=8)
            entry.grid(row=1, column=i+1, padx=2, pady=5)
            self.obj_entries.append(entry)
            if i < self.num_variables - 1:
                ttk.Label(obj_frame, text=f"x{i+1} + ").grid(row=1, column=i+2, sticky=tk.W)
            else:
                ttk.Label(obj_frame, text=f"x{i+1}").grid(row=1, column=i+2, sticky=tk.W)
        
        # Constraints frame
        const_frame = ttk.LabelFrame(scrollable_frame, text="Constraints", padding="10")
        const_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10, padx=10)
        
        self.constraint_entries = []
        self.constraint_type_vars = []
        self.rhs_entries = []
        
        for i in range(self.num_constraints):
            row_entries = []
            # Constraint coefficients
            for j in range(self.num_variables):
                entry = ttk.Entry(const_frame, width=8)
                entry.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(entry)
                if j < self.num_variables - 1:
                    ttk.Label(const_frame, text=f"x{j+1} + ").grid(row=i, column=self.num_variables+j, sticky=tk.W)
                else:
                    ttk.Label(const_frame, text=f"x{j+1}").grid(row=i, column=self.num_variables+j, sticky=tk.W)
            
            self.constraint_entries.append(row_entries)
            
            # Constraint type
            constraint_type = tk.StringVar(value="≤")
            combo = ttk.Combobox(const_frame, textvariable=constraint_type, 
                               values=["≤", "≥", "="], width=3, state="readonly")
            combo.grid(row=i, column=self.num_variables*2, padx=5, pady=2)
            self.constraint_type_vars.append(constraint_type)
            
            # RHS
            rhs_entry = ttk.Entry(const_frame, width=8)
            rhs_entry.grid(row=i, column=self.num_variables*2+1, padx=5, pady=2)
            self.rhs_entries.append(rhs_entry)
        
        # Buttons frame
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=20)
        
        ttk.Button(btn_frame, text="Back", command=self.show_initial_screen).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Solve with Graphical Method", 
                  command=self.solve_graphical).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Solve with Two-Phase Method", 
                  command=self.solve_two_phase).grid(row=0, column=2, padx=5)
        
        # Pack scrollable components
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
    
    def validate_problem_input(self):
        try:
            # Validate objective coefficients
            self.objective_coeffs = []
            for entry in self.obj_entries:
                val = float(entry.get()) if entry.get() else 0.0
                self.objective_coeffs.append(val)
            
            # Validate constraints
            self.constraint_coeffs = []
            self.constraint_rhs = []
            self.constraint_types = []
            
            for i in range(self.num_constraints):
                row_coeffs = []
                for j in range(self.num_variables):
                    val = float(self.constraint_entries[i][j].get()) if self.constraint_entries[i][j].get() else 0.0
                    row_coeffs.append(val)
                self.constraint_coeffs.append(row_coeffs)
                
                rhs_val = float(self.rhs_entries[i].get()) if self.rhs_entries[i].get() else 0.0
                self.constraint_rhs.append(rhs_val)
                
                self.constraint_types.append(self.constraint_type_vars[i].get())
            
            self.is_maximization = self.obj_type.get() == "max"
            return True
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for all coefficients.")
            return False
    
    def solve_graphical(self):
        if self.num_variables != 2:
            messagebox.showerror("Invalid Method", "Graphical method is only available for exactly 2 variables.")
            return
        
        if not self.validate_problem_input():
            return
        
        self.show_graphical_solution()
    
    def solve_two_phase(self):
        if not self.validate_problem_input():
            return
        
        self.show_two_phase_solution()
    
    def show_graphical_solution(self):
        self.clear_frame()
        
        # Create solution display
        solution_frame = ttk.Frame(self.main_frame)
        solution_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(solution_frame, text="Graphical Method Solution", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot constraints and find feasible region
        x_range = np.linspace(-1, 20, 400)
        feasible_points = []
        
        # Add non-negativity constraints
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        # Plot each constraint
        constraint_lines = []
        for i, (coeffs, rhs, c_type) in enumerate(zip(self.constraint_coeffs, self.constraint_rhs, self.constraint_types)):
            if coeffs[1] != 0:  # Can be expressed as y = mx + b
                y_values = (rhs - coeffs[0] * x_range) / coeffs[1]
                ax.plot(x_range, y_values, label=f'Constraint {i+1}')
                constraint_lines.append((coeffs, rhs, c_type))
            elif coeffs[0] != 0:  # Vertical line
                ax.axvline(x=rhs/coeffs[0], label=f'Constraint {i+1}')
                constraint_lines.append((coeffs, rhs, c_type))
        
        # Find corner points of feasible region
        corner_points = [(0, 0)]  # Origin
        
        # Intersection with axes
        for coeffs, rhs, c_type in constraint_lines:
            if coeffs[0] != 0:
                x_intercept = rhs / coeffs[0]
                if x_intercept >= 0:
                    corner_points.append((x_intercept, 0))
            if coeffs[1] != 0:
                y_intercept = rhs / coeffs[1]
                if y_intercept >= 0:
                    corner_points.append((0, y_intercept))
        
        # Intersection between constraints
        for i in range(len(constraint_lines)):
            for j in range(i+1, len(constraint_lines)):
                coeffs1, rhs1, _ = constraint_lines[i]
                coeffs2, rhs2, _ = constraint_lines[j]
                
                # Solve system of equations
                try:
                    A = np.array([coeffs1, coeffs2])
                    b = np.array([rhs1, rhs2])
                    point = np.linalg.solve(A, b)
                    if point[0] >= 0 and point[1] >= 0:
                        corner_points.append(tuple(point))
                except:
                    continue
        
        # Remove duplicates and filter feasible points
        unique_points = []
        for point in corner_points:
            is_feasible = True
            # Check if point satisfies all constraints
            for coeffs, rhs, c_type in constraint_lines:
                value = coeffs[0] * point[0] + coeffs[1] * point[1]
                if c_type == "≤" and value > rhs + 1e-10:
                    is_feasible = False
                    break
                elif c_type == "≥" and value < rhs - 1e-10:
                    is_feasible = False
                    break
                elif c_type == "=" and abs(value - rhs) > 1e-10:
                    is_feasible = False
                    break
            
            if is_feasible:
                # Check for duplicates
                is_duplicate = False
                for existing_point in unique_points:
                    if abs(point[0] - existing_point[0]) < 1e-10 and abs(point[1] - existing_point[1]) < 1e-10:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_points.append(point)
        
        # Plot feasible region
        if len(unique_points) >= 3:
            # Sort points to form proper polygon
            from math import atan2
            center_x = sum(p[0] for p in unique_points) / len(unique_points)
            center_y = sum(p[1] for p in unique_points) / len(unique_points)
            
            def angle_from_center(point):
                return atan2(point[1] - center_y, point[0] - center_x)
            
            unique_points.sort(key=angle_from_center)
            
            polygon = Polygon(unique_points, alpha=0.3, color='lightblue')
            ax.add_patch(polygon)
        
        # Find optimal solution
        optimal_point = None
        optimal_value = float('-inf') if self.is_maximization else float('inf')
        
        for point in unique_points:
            value = self.objective_coeffs[0] * point[0] + self.objective_coeffs[1] * point[1]
            if self.is_maximization and value > optimal_value:
                optimal_value = value
                optimal_point = point
            elif not self.is_maximization and value < optimal_value:
                optimal_value = value
                optimal_point = point
        
        # Plot optimal point
        if optimal_point:
            ax.plot(optimal_point[0], optimal_point[1], 'ro', markersize=10, label='Optimal Solution')
            
            # Plot objective function line through optimal point
            if self.objective_coeffs[1] != 0:
                y_obj = (optimal_value - self.objective_coeffs[0] * x_range) / self.objective_coeffs[1]
                ax.plot(x_range, y_obj, '--', color='red', alpha=0.7, label='Objective Function')
        
        # Plot corner points
        for point in unique_points:
            ax.plot(point[0], point[1], 'bo', markersize=6)
            ax.annotate(f'({point[0]:.2f}, {point[1]:.2f})', 
                       (point[0], point[1]), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'{"Maximize" if self.is_maximization else "Minimize"} Z = {self.objective_coeffs[0]}x1 + {self.objective_coeffs[1]}x2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, max(10, max(p[0] for p in unique_points) + 2) if unique_points else 10)
        ax.set_ylim(-1, max(10, max(p[1] for p in unique_points) + 2) if unique_points else 10)
        
        # Embed plot in tkinter
        canvas_widget = FigureCanvasTkAgg(fig, solution_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        
        # Results text
        result_frame = ttk.Frame(solution_frame)
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        result_text = scrolledtext.ScrolledText(result_frame, height=10, width=80)
        result_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Display solution steps
        steps = f"GRAPHICAL METHOD SOLUTION\n{'='*50}\n\n"
        steps += f"Problem: {'Maximize' if self.is_maximization else 'Minimize'} Z = "
        steps += " + ".join([f"{coeff}x{i+1}" for i, coeff in enumerate(self.objective_coeffs)]) + "\n\n"
        
        steps += "Subject to:\n"
        for i, (coeffs, rhs, c_type) in enumerate(zip(self.constraint_coeffs, self.constraint_rhs, self.constraint_types)):
            constraint_str = " + ".join([f"{coeff}x{j+1}" for j, coeff in enumerate(coeffs)])
            steps += f"{constraint_str} {c_type} {rhs}\n"
        steps += "x1, x2 ≥ 0\n\n"
        
        steps += "Corner Points Found:\n"
        for i, point in enumerate(unique_points):
            value = self.objective_coeffs[0] * point[0] + self.objective_coeffs[1] * point[1]
            steps += f"Point {i+1}: ({point[0]:.4f}, {point[1]:.4f}), Z = {value:.4f}\n"
        
        if optimal_point:
            steps += f"\nOPTIMAL SOLUTION:\n"
            steps += f"x1 = {optimal_point[0]:.4f}\n"
            steps += f"x2 = {optimal_point[1]:.4f}\n"
            steps += f"Optimal value = {optimal_value:.4f}\n"
        else:
            steps += "\nNo feasible solution found.\n"
        
        result_text.insert(tk.END, steps)
        result_text.config(state=tk.DISABLED)
        
        # Buttons
        btn_frame = ttk.Frame(solution_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Back to Input", 
                  command=self.show_problem_input_screen).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Toggle Fractions/Decimals", 
                  command=self.toggle_number_format).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="New Problem", 
                  command=self.show_initial_screen).grid(row=0, column=2, padx=5)
    
    def show_two_phase_solution(self):
        self.clear_frame()
        
        # Create solution display with tabs
        notebook = ttk.Notebook(self.main_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Phase 1 tab
        phase1_frame = ttk.Frame(notebook)
        notebook.add(phase1_frame, text="Phase 1")
        
        # Phase 2 tab
        phase2_frame = ttk.Frame(notebook)
        notebook.add(phase2_frame, text="Phase 2")
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Final Results")
        
        # Solve using two-phase method
        solution_data = self.solve_two_phase_method()
        self.current_solution_data = solution_data  # Store for refresh
        
        # Display Phase 1
        self.display_phase_results(phase1_frame, solution_data['phase1'], "Phase 1")
        
        # Display Phase 2
        if solution_data['phase2']:
            self.display_phase_results(phase2_frame, solution_data['phase2'], "Phase 2")
        
        # Display final results
        self.display_final_results(results_frame, solution_data)
        
        # Control buttons
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.grid(row=1, column=0, pady=10)
        
        ttk.Button(btn_frame, text="Back to Input", 
                  command=self.show_problem_input_screen).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Toggle Fractions/Decimals", 
                  command=self.toggle_number_format).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="New Problem", 
                  command=self.show_initial_screen).grid(row=0, column=2, padx=5)
    
    def solve_two_phase_method(self):
        # Convert to standard form and solve using two-phase method
        # This is a simplified implementation
        
        # Initialize tableau
        tableau_data = self.setup_initial_tableau()
        
        phase1_iterations = []
        phase2_iterations = []
        
        # Phase 1: Remove artificial variables
        if tableau_data['has_artificial']:
            phase1_iterations = self.phase1_simplex(tableau_data)
        
        # Phase 2: Optimize original objective
        if not tableau_data.get('infeasible', False):
            phase2_iterations = self.phase2_simplex(tableau_data)
        
        return {
            'phase1': phase1_iterations,
            'phase2': phase2_iterations,
            'final_tableau': tableau_data,
            'optimal_solution': self.extract_solution(tableau_data),
            'is_feasible': not tableau_data.get('infeasible', False)
        }
    
    def setup_initial_tableau(self):
        # Convert problem to standard form
        num_slack = 0
        num_surplus = 0
        num_artificial = 0
        
        # Count variables needed
        for constraint_type in self.constraint_types:
            if constraint_type == "≤":
                num_slack += 1
            elif constraint_type == "≥":
                num_surplus += 1
                num_artificial += 1
            else:  # "="
                num_artificial += 1
        
        total_vars = self.num_variables + num_slack + num_surplus + num_artificial
        
        # Create initial tableau
        tableau = np.zeros((self.num_constraints + 1, total_vars + 1))
        
        # Fill constraint coefficients
        col_idx = 0
        
        # Original variables
        for i in range(self.num_constraints):
            for j in range(self.num_variables):
                tableau[i, j] = self.constraint_coeffs[i][j]
        col_idx = self.num_variables
        
        # Add slack, surplus, and artificial variables
        slack_idx = 0
        surplus_idx = 0
        artificial_idx = 0
        
        for i, constraint_type in enumerate(self.constraint_types):
            if constraint_type == "≤":
                tableau[i, col_idx + slack_idx] = 1
                slack_idx += 1
            elif constraint_type == "≥":
                tableau[i, col_idx + num_slack + surplus_idx] = -1
                tableau[i, col_idx + num_slack + num_surplus + artificial_idx] = 1
                surplus_idx += 1
                artificial_idx += 1
            else:  # "="
                tableau[i, col_idx + num_slack + num_surplus + artificial_idx] = 1
                artificial_idx += 1
        
        # RHS values
        for i in range(self.num_constraints):
            tableau[i, -1] = self.constraint_rhs[i]
        
        # Objective function (Phase 1 if artificial variables exist)
        has_artificial = num_artificial > 0
        if has_artificial:
            # Phase 1: minimize sum of artificial variables
            artificial_start = self.num_variables + num_slack + num_surplus
            for j in range(artificial_start, artificial_start + num_artificial):
                tableau[-1, j] = 1
        else:
            # No artificial variables, use original objective
            for j in range(self.num_variables):
                coeff = self.objective_coeffs[j]
                if self.is_maximization:
                    coeff = -coeff
                tableau[-1, j] = coeff
        
        return {
            'tableau': tableau,
            'basic_vars': list(range(self.num_variables + num_slack + num_surplus, total_vars)),
            'has_artificial': has_artificial,
            'num_slack': num_slack,
            'num_surplus': num_surplus,
            'num_artificial': num_artificial,
            'phase': 1 if has_artificial else 2
        }
    
    def phase1_simplex(self, tableau_data):
        iterations = []
        tableau = tableau_data['tableau'].copy()
        basic_vars = tableau_data['basic_vars'].copy()
        
        # Make artificial variables basic and eliminate them from objective
        artificial_start = self.num_variables + tableau_data['num_slack'] + tableau_data['num_surplus']
        
        for i, var_idx in enumerate(basic_vars):
            if var_idx >= artificial_start:
                # Eliminate this artificial variable from objective row
                multiplier = tableau[-1, var_idx]
                for j in range(tableau.shape[1]):
                    tableau[-1, j] -= multiplier * tableau[i, j]
        
        iteration = 0
        while True:
            # Store current iteration
            iterations.append({
                'iteration': iteration,
                'tableau': tableau.copy(),
                'basic_vars': basic_vars.copy(),
                'step_description': f"Phase 1 - Iteration {iteration}"
            })
            
            # Check optimality (all coefficients in objective row ≥ 0)
            entering_col = -1
            min_ratio = float('inf')
            
            for j in range(tableau.shape[1] - 1):
                if tableau[-1, j] < -1e-10:
                    entering_col = j
                    break
            
            if entering_col == -1:
                # Optimal solution found for Phase 1
                break
            
            # Find leaving variable (minimum ratio test)
            leaving_row = -1
            min_ratio = float('inf')
            
            for i in range(tableau.shape[0] - 1):
                if tableau[i, entering_col] > 1e-10:
                    ratio = tableau[i, -1] / tableau[i, entering_col]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leaving_row = i
            
            if leaving_row == -1:
                # Unbounded solution
                tableau_data['infeasible'] = True
                break
            
            # Pivot operation
            pivot_element = tableau[leaving_row, entering_col]
            
            # Scale pivot row
            for j in range(tableau.shape[1]):
                tableau[leaving_row, j] /= pivot_element
            
            # Eliminate other entries in pivot column
            for i in range(tableau.shape[0]):
                if i != leaving_row and abs(tableau[i, entering_col]) > 1e-10:
                    multiplier = tableau[i, entering_col]
                    for j in range(tableau.shape[1]):
                        tableau[i, j] -= multiplier * tableau[leaving_row, j]
            
            # Update basic variables
            basic_vars[leaving_row] = entering_col
            iteration += 1
            
            if iteration > 100:  # Prevent infinite loops
                break
        
        # Check if Phase 1 solution is feasible
        if abs(tableau[-1, -1]) > 1e-10:
            tableau_data['infeasible'] = True
        
        # Update tableau data for Phase 2
        tableau_data['tableau'] = tableau
        tableau_data['basic_vars'] = basic_vars
        
        return iterations
    
    def phase2_simplex(self, tableau_data):
        iterations = []
        tableau = tableau_data['tableau'].copy()
        basic_vars = tableau_data['basic_vars'].copy()
        
        # Replace objective function with original objective
        tableau[-1, :] = 0
        for j in range(self.num_variables):
            coeff = self.objective_coeffs[j]
            if self.is_maximization:
                coeff = -coeff
            tableau[-1, j] = coeff
        
        # Eliminate basic variables from objective row
        for i, var_idx in enumerate(basic_vars):
            if var_idx < self.num_variables and abs(tableau[-1, var_idx]) > 1e-10:
                multiplier = tableau[-1, var_idx]
                for j in range(tableau.shape[1]):
                    tableau[-1, j] -= multiplier * tableau[i, j]
        
        iteration = 0
        while True:
            # Store current iteration
            iterations.append({
                'iteration': iteration,
                'tableau': tableau.copy(),
                'basic_vars': basic_vars.copy(),
                'step_description': f"Phase 2 - Iteration {iteration}"
            })
            
            # Check optimality
            entering_col = -1
            for j in range(tableau.shape[1] - 1):
                if tableau[-1, j] < -1e-10:
                    entering_col = j
                    break
            
            if entering_col == -1:
                break
            
            # Find leaving variable
            leaving_row = -1
            min_ratio = float('inf')
            
            for i in range(tableau.shape[0] - 1):
                if tableau[i, entering_col] > 1e-10:
                    ratio = tableau[i, -1] / tableau[i, entering_col]
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leaving_row = i
            
            if leaving_row == -1:
                tableau_data['unbounded'] = True
                break
            
            # Pivot operation
            pivot_element = tableau[leaving_row, entering_col]
            
            for j in range(tableau.shape[1]):
                tableau[leaving_row, j] /= pivot_element
            
            for i in range(tableau.shape[0]):
                if i != leaving_row and abs(tableau[i, entering_col]) > 1e-10:
                    multiplier = tableau[i, entering_col]
                    for j in range(tableau.shape[1]):
                        tableau[i, j] -= multiplier * tableau[leaving_row, j]
            
            basic_vars[leaving_row] = entering_col
            iteration += 1
            
            if iteration > 100:
                break
        
        tableau_data['tableau'] = tableau
        tableau_data['basic_vars'] = basic_vars
        
        return iterations
    
    def extract_solution(self, tableau_data):
        if tableau_data.get('infeasible', False):
            return None
        
        tableau = tableau_data['tableau']
        basic_vars = tableau_data['basic_vars']
        
        solution = [0.0] * self.num_variables
        
        for i, var_idx in enumerate(basic_vars):
            if var_idx < self.num_variables:
                solution[var_idx] = tableau[i, -1]
        
        objective_value = -tableau[-1, -1] if self.is_maximization else tableau[-1, -1]
        
        return {
            'variables': solution,
            'objective_value': objective_value
        }
    
    def display_phase_results(self, parent_frame, iterations, phase_name):
        # Title
        title_label = ttk.Label(parent_frame, text=f"{phase_name} Results", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # Create frame with horizontal scrollbar for tableau
        canvas = tk.Canvas(parent_frame)
        h_scrollbar = ttk.Scrollbar(parent_frame, orient="horizontal", command=canvas.xview)
        v_scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Calculate variable counts for headers
        num_slack = sum(1 for ct in self.constraint_types if ct == "≤")
        num_surplus = sum(1 for ct in self.constraint_types if ct == "≥")
        num_artificial = sum(1 for ct in self.constraint_types if ct in ["≥", "="])
        
        # Display each iteration
        for iter_data in iterations:
            iter_frame = ttk.LabelFrame(scrollable_frame, text=iter_data['step_description'], padding="10")
            iter_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Create tableau display
            tableau = iter_data['tableau']
            basic_vars = iter_data['basic_vars']
            
            # Headers
            headers = [f"x{i+1}" for i in range(self.num_variables)]
            
            # Add slack/surplus/artificial variable headers
            total_vars = tableau.shape[1] - 1
            for i in range(self.num_variables, total_vars):
                if i < self.num_variables + num_slack:
                    headers.append(f"s{i - self.num_variables + 1}")
                elif i < self.num_variables + num_slack + num_surplus:
                    headers.append(f"e{i - self.num_variables - num_slack + 1}")
                else:
                    headers.append(f"a{i - self.num_variables - num_slack - num_surplus + 1}")
            
            headers.append("RHS")
            
            # Create treeview for table display
            tree = ttk.Treeview(iter_frame, columns=headers, show='headings', height=min(10, tableau.shape[0] + 1))
            
            # Configure columns
            for header in headers:
                tree.heading(header, text=header)
                tree.column(header, width=80, anchor='center')
            
            # Add constraint rows
            for i in range(tableau.shape[0] - 1):
                row_data = []
                for j in range(tableau.shape[1]):
                    if self.show_fractions:
                        frac = Fraction(tableau[i, j]).limit_denominator(1000)
                        row_data.append(str(frac))
                    else:
                        row_data.append(f"{tableau[i, j]:.4f}")
                
                basic_var_name = ""
                if i < len(basic_vars):
                    var_idx = basic_vars[i]
                    if var_idx < self.num_variables:
                        basic_var_name = f"x{var_idx + 1}"
                    else:
                        basic_var_name = headers[var_idx]
                
                tree.insert('', tk.END, values=row_data, tags=(f"basic_{basic_var_name}",))
            
            # Add objective row
            obj_row = []
            for j in range(tableau.shape[1]):
                if self.show_fractions:
                    frac = Fraction(tableau[-1, j]).limit_denominator(1000)
                    obj_row.append(str(frac))
                else:
                    obj_row.append(f"{tableau[-1, j]:.4f}")
            
            tree.insert('', tk.END, values=obj_row, tags=("objective",))
            
            # Configure row tags
            tree.tag_configure("objective", background="lightgray")
            
            tree.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbars to treeview
            tree_v_scroll = ttk.Scrollbar(iter_frame, orient="vertical", command=tree.yview)
            tree_h_scroll = ttk.Scrollbar(iter_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=tree_v_scroll.set, xscrollcommand=tree_h_scroll.set)
            
            tree_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            tree_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Pack scrollable components
        canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)
    
    def display_final_results(self, parent_frame, solution_data):
        # Title
        title_label = ttk.Label(parent_frame, text="Final Results", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # Results text
        result_text = scrolledtext.ScrolledText(parent_frame, height=20, width=80)
        result_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Generate results summary
        results = "TWO-PHASE METHOD SOLUTION\n"
        results += "=" * 50 + "\n\n"
        
        results += f"Original Problem:\n"
        results += f"{'Maximize' if self.is_maximization else 'Minimize'} Z = "
        results += " + ".join([f"{coeff}x{i+1}" for i, coeff in enumerate(self.objective_coeffs)]) + "\n\n"
        
        results += "Subject to:\n"
        for i, (coeffs, rhs, c_type) in enumerate(zip(self.constraint_coeffs, self.constraint_rhs, self.constraint_types)):
            constraint_str = " + ".join([f"{coeff}x{j+1}" for j, coeff in enumerate(coeffs)])
            results += f"{constraint_str} {c_type} {rhs}\n"
        results += "All variables ≥ 0\n\n"
        
        # Solution status
        if not solution_data['is_feasible']:
            results += "RESULT: The problem has NO FEASIBLE SOLUTION\n\n"
            results += "The Phase 1 of the two-phase method did not yield a feasible solution.\n"
            results += "This means the constraint set is inconsistent.\n"
        elif solution_data['final_tableau'].get('unbounded', False):
            results += "RESULT: The problem has an UNBOUNDED SOLUTION\n\n"
            results += "The objective function can be improved indefinitely.\n"
        else:
            solution = solution_data['optimal_solution']
            results += "RESULT: OPTIMAL SOLUTION FOUND\n\n"
            
            results += "Optimal Variable Values:\n"
            for i, value in enumerate(solution['variables']):
                if self.show_fractions:
                    frac = Fraction(value).limit_denominator(1000)
                    results += f"x{i+1} = {frac}\n"
                else:
                    results += f"x{i+1} = {value:.6f}\n"
            
            if self.show_fractions:
                obj_frac = Fraction(solution['objective_value']).limit_denominator(1000)
                results += f"\nOptimal Objective Value: Z = {obj_frac}\n"
            else:
                results += f"\nOptimal Objective Value: Z = {solution['objective_value']:.6f}\n"
        
        # Add solution steps explanation
        results += "\n" + "=" * 50 + "\n"
        results += "SOLUTION PROCESS:\n\n"
        
        results += "1. PROBLEM CONVERSION TO STANDARD FORM:\n"
        slack_count = sum(1 for ct in self.constraint_types if ct == "≤")
        surplus_count = sum(1 for ct in self.constraint_types if ct == "≥")
        artificial_count = sum(1 for ct in self.constraint_types if ct in ["≥", "="])
        
        if slack_count > 0:
            results += f"   - Added {slack_count} slack variable(s) for ≤ constraints\n"
        if surplus_count > 0:
            results += f"   - Added {surplus_count} surplus variable(s) for ≥ constraints\n"
        if artificial_count > 0:
            results += f"   - Added {artificial_count} artificial variable(s) for ≥ and = constraints\n"
        
        if artificial_count > 0:
            results += "\n2. PHASE 1:\n"
            results += "   - Objective: Minimize sum of artificial variables\n"
            results += f"   - Performed {len(solution_data['phase1'])} iteration(s)\n"
            if solution_data['is_feasible']:
                results += "   - Result: Feasible solution found, artificial variables eliminated\n"
            else:
                results += "   - Result: No feasible solution exists\n"
        
        if solution_data['phase2'] and solution_data['is_feasible']:
            results += "\n3. PHASE 2:\n"
            results += "   - Objective: Optimize original objective function\n"
            results += f"   - Performed {len(solution_data['phase2'])} iteration(s)\n"
            results += "   - Result: Optimal solution found\n"
        
        result_text.insert(tk.END, results)
        result_text.config(state=tk.DISABLED)
        
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)
    
    def toggle_number_format(self):
        self.show_fractions = not self.show_fractions
        # Refresh current display
        if hasattr(self, 'current_solution_data'):
            self.show_two_phase_solution()
    
    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = LinearProgrammingCalculator()
    app.run()