import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
import fractions
from fractions import Fraction
import copy

class LinearProgrammingCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Linear Programming Solver")
        self.geometry("1200x900")

        # App state variables
        self.num_vars = tk.IntVar(value=2)
        self.num_constraints = tk.IntVar(value=1)
        self.problem_type = tk.StringVar(value="Maximize")
        self.objective_coeffs = []
        self.constraint_coeffs = []
        self.constraint_signs = []
        self.constraint_rhs = []

        # Main container
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (StartPage, InputPage, ResultsPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")
    
    def show_frame(self, page_name):
        """Show a frame for the given page name."""
        frame = self.frames[page_name]
        if hasattr(frame, 'on_show'):
            frame.on_show()
        frame.tkraise()
    
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
        self.mainloop()

class StartPage(tk.Frame):
    """Initial page to get the number of variables and constraints."""
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Problem Setup", font=("Arial", 16, "bold"))
        label.pack(pady=20)

        # Frame for inputs
        input_frame = tk.Frame(self)
        input_frame.pack(pady=20, padx=20)

        tk.Label(input_frame, text="Number of Variables:").grid(row=0, column=0, padx=5, pady=10, sticky="e")
        self.vars_entry = tk.Entry(input_frame, textvariable=self.controller.num_vars, width=5)
        self.vars_entry.grid(row=0, column=1, padx=5, pady=10)

        tk.Label(input_frame, text="Number of Constraints:").grid(row=1, column=0, padx=5, pady=10, sticky="e")
        self.constraints_entry = tk.Entry(input_frame, textvariable=self.controller.num_constraints, width=5)
        self.constraints_entry.grid(row=1, column=1, padx=5, pady=10)

        next_button = tk.Button(self, text="Next", command=self.validate_and_proceed)
        next_button.pack(pady=20)

    def validate_and_proceed(self):
        """Validates inputs and proceeds to the next page."""
        try:
            num_vars_str = self.vars_entry.get().strip()
            num_constraints_str = self.constraints_entry.get().strip()
            num_vars = int(num_vars_str)
            num_constraints = int(num_constraints_str)

            if not (1 <= num_vars <= 10):
                messagebox.showerror("Invalid Input", "Number of variables must be between 1 and 10.")
                return
            if not (1 <= num_constraints <= 15):
                messagebox.showerror("Invalid Input", "Number of constraints must be between 1 and 15.")
                return

            self.controller.num_vars.set(num_vars)
            self.controller.num_constraints.set(num_constraints)
            self.controller.show_frame("InputPage")

        except ValueError:
            messagebox.showerror("Invalid Input", "Por favor, ingresa números enteros válidos en ambos campos.")

class InputPage(tk.Frame):
    """Page for inputting the objective function and constraints."""
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.canvas = None

    def on_show(self):
        """Dynamically create input fields based on user selection."""
        if self.canvas:
            # Desvincular el scrollbar antes de destruir el canvas
            for child in self.canvas.master.winfo_children():
                if isinstance(child, ttk.Scrollbar):
                    child.config(command=None)
            self.canvas.destroy()
        self._create_widgets()

    def _create_widgets(self):
        num_vars = self.controller.num_vars.get()
        num_constraints = self.controller.num_constraints.get()

        # Main frame with a scrollbar
        main_frame = tk.Frame(self)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        scrollable_frame = ttk.Frame(self.canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # --- Objective Function ---
        obj_frame = tk.LabelFrame(scrollable_frame, text="Objective Function", padx=10, pady=10)
        obj_frame.pack(pady=10, padx=20, fill="x")

        tk.Radiobutton(obj_frame, text="Maximize", variable=self.controller.problem_type, value="Maximize").pack(side="left")
        tk.Radiobutton(obj_frame, text="Minimize", variable=self.controller.problem_type, value="Minimize").pack(side="left", padx=20)

        func_frame = tk.Frame(obj_frame)
        func_frame.pack(pady=10)
        
        tk.Label(func_frame, text="Z = ").pack(side="left")
        self.controller.objective_coeffs = []
        for i in range(num_vars):
            entry = tk.Entry(func_frame, width=5)
            entry.pack(side="left")
            self.controller.objective_coeffs.append(entry)
            tk.Label(func_frame, text=f"x{i+1} " + ("+" if i < num_vars - 1 else "")).pack(side="left")

        # --- Constraints ---
        constraints_frame = tk.LabelFrame(scrollable_frame, text="Constraints", padx=10, pady=10)
        constraints_frame.pack(pady=10, padx=20, fill="x")
        
        self.controller.constraint_coeffs = []
        self.controller.constraint_signs = []
        self.controller.constraint_rhs = []

        for i in range(num_constraints):
            row_frame = tk.Frame(constraints_frame)
            row_frame.pack(fill="x", pady=5)
            coeffs_row = []
            for j in range(num_vars):
                entry = tk.Entry(row_frame, width=5)
                entry.pack(side="left")
                coeffs_row.append(entry)
                tk.Label(row_frame, text=f"x{j+1} " + ("+" if j < num_vars - 1 else "")).pack(side="left")
            self.controller.constraint_coeffs.append(coeffs_row)

            sign_var = tk.StringVar(value="<=")
            sign_menu = tk.OptionMenu(row_frame, sign_var, "<=", ">=", "=")
            sign_menu.pack(side="left", padx=5)
            self.controller.constraint_signs.append(sign_var)

            rhs_entry = tk.Entry(row_frame, width=5)
            rhs_entry.pack(side="left")
            self.controller.constraint_rhs.append(rhs_entry)

        # --- Navigation and Solve Buttons ---
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(pady=20, padx=20)
        
        back_button = tk.Button(button_frame, text="Back", command=lambda: self.controller.show_frame("StartPage"))
        back_button.pack(side="left", padx=10)
        
        self.graphical_button = tk.Button(button_frame, text="Solve with Graphical Method", command=self.solve_graphical)
        self.graphical_button.pack(side="left", padx=10)
        
        two_phase_button = tk.Button(button_frame, text="Solve with Two-Phase Method", command=self.solve_two_phase)
        two_phase_button.pack(side="left", padx=10)

        # Enable graphical button only if num_vars is 2
        if num_vars != 2:
            self.graphical_button.config(state="disabled")

    def _get_problem_data(self):
        """Helper to parse and validate all input fields."""
        try:
            problem = {}
            problem['type'] = self.controller.problem_type.get()
            problem['obj_coeffs'] = [float(e.get()) for e in self.controller.objective_coeffs]
            
            problem['constraints'] = []
            for i in range(self.controller.num_constraints.get()):
                constraint = {
                    'coeffs': [float(e.get()) for e in self.controller.constraint_coeffs[i]],
                    'sign': self.controller.constraint_signs[i].get(),
                    'rhs': float(self.controller.constraint_rhs[i].get())
                }
                problem['constraints'].append(constraint)
            return problem
        except (ValueError, IndexError):
            messagebox.showerror("Input Error", "Please ensure all fields are filled with valid numbers.")
            return None

    def solve_graphical(self):
        problem_data = self._get_problem_data()
        if problem_data:
            self.controller.problem_data = problem_data
            self.controller.solve_method = "Graphical"
            self.controller.show_frame("ResultsPage")

    def solve_two_phase(self):
        problem_data = self._get_problem_data()
        if problem_data:
            self.controller.problem_data = problem_data
            self.controller.solve_method = "Two-Phase"
            self.controller.show_frame("ResultsPage")


class ResultsPage(tk.Frame):
    """Page to display the solution and steps."""
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.results_frame = None

    def on_show(self):
        """Clear previous results and run the selected solver."""
        if self.results_frame:
            self.results_frame.destroy()
        
        self.results_frame = tk.Frame(self)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

        back_button = tk.Button(self.results_frame, text="Back to Inputs", command=lambda: self.controller.show_frame("InputPage"))
        back_button.pack(pady=10)

        if self.controller.solve_method == "Graphical":
            self.display_graphical_solution()
        elif self.controller.solve_method == "Two-Phase":
            self.display_two_phase_solution()

    def display_graphical_solution(self):
        """Calculates and displays the graphical method solution."""
        problem = self.controller.problem_data
        
        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot constraints
        d = np.linspace(0, 50, 2000)
        x, y = np.meshgrid(d, d)
        
        feasible_region = np.ones(x.shape, dtype=bool)
        
        explanation = "### Solution Process (Graphical Method)\n\n"
        explanation += "1.  **Plotting Constraints:** Each constraint is plotted as a line on the graph.\n"

        for i, c in enumerate(problem['constraints']):
            coeffs = c['coeffs']
            rhs = c['rhs']
            sign = c['sign']
            
            # Ensure we don't divide by zero
            if coeffs[1] != 0:
                y_vals = (rhs - coeffs[0] * d) / coeffs[1]
                ax.plot(d, y_vals, label=f'Constraint {i+1}')
                
                # Shading the feasible region
                if sign == '<=':
                    feasible_region &= (coeffs[0]*x + coeffs[1]*y <= rhs)
                elif sign == '>=':
                    feasible_region &= (coeffs[0]*x + coeffs[1]*y >= rhs)
            elif coeffs[0] != 0: # Vertical line
                x_val = rhs / coeffs[0]
                ax.axvline(x=x_val, label=f'Constraint {i+1}')
                if sign == '<=':
                    feasible_region &= (x <= x_val)
                elif sign == '>=':
                    feasible_region &= (x >= x_val)
            
            explanation += f"    - **Constraint {i+1}:** `{coeffs[0]}x1 + {coeffs[1]}x2 {sign} {rhs}`\n"


        # Non-negativity constraints
        feasible_region &= (x >= 0)
        feasible_region &= (y >= 0)

        # Shade the feasible region
        ax.imshow(feasible_region.T, extent=(0, 50, 0, 50), origin='lower', cmap="Greys", alpha=0.3)
        explanation += "\n2.  **Identifying Feasible Region:** The area satisfying all constraints simultaneously (including non-negativity x1, x2 >= 0) is shaded.\n"

        # Find corner points
        # This is a simplified approach; a full implementation requires finding all intersections.
        # For this example, we'll test intersections of all pairs of constraints.
        corner_points = [(0, 0)]
        constraints = problem['constraints']
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                A = np.array([constraints[i]['coeffs'], constraints[j]['coeffs']])
                b = np.array([constraints[i]['rhs'], constraints[j]['rhs']])
                try:
                    point = np.linalg.solve(A, b)
                    if np.all(point >= 0):
                        corner_points.append(tuple(point))
                except np.linalg.LinAlgError:
                    continue # Parallel lines

        # Intersections with axes
        for c in constraints:
            if c['coeffs'][0] != 0:
                p = (c['rhs']/c['coeffs'][0], 0)
                if p[0] >= 0: corner_points.append(p)
            if c['coeffs'][1] != 0:
                p = (0, c['rhs']/c['coeffs'][1])
                if p[1] >= 0: corner_points.append(p)

        # Filter points that are actually feasible
        feasible_points = []
        for p in corner_points:
            is_feasible = True
            for c in constraints:
                val = c['coeffs'][0]*p[0] + c['coeffs'][1]*p[1]
                if (c['sign'] == '<=' and val > c['rhs'] + 1e-6) or \
                   (c['sign'] == '>=' and val < c['rhs'] - 1e-6):
                    is_feasible = False
                    break
            if is_feasible:
                feasible_points.append(p)
        
        feasible_points = sorted(list(set(feasible_points))) # Remove duplicates

        explanation += "\n3.  **Finding Corner Points:** The vertices (corners) of the feasible region are calculated by finding the intersection of the constraint lines.\n"
        explanation += f"    - Feasible corner points found: {feasible_points}\n"

        if not feasible_points:
            result_text = "No feasible solution found."
        else:
            # Evaluate objective function at each corner point
            obj_coeffs = problem['obj_coeffs']
            obj_values = [obj_coeffs[0]*p[0] + obj_coeffs[1]*p[1] for p in feasible_points]
            
            if problem['type'] == "Maximize":
                opt_idx = np.argmax(obj_values)
            else:
                opt_idx = np.argmin(obj_values)
                
            opt_point = feasible_points[opt_idx]
            opt_value = obj_values[opt_idx]

            explanation += f"\n4.  **Evaluating Objective Function:** The objective function `Z = {obj_coeffs[0]}x1 + {obj_coeffs[1]}x2` is evaluated at each corner point.\n"
            explanation += f"    - The optimal value is found by {'maximizing' if problem['type'] == 'Maximize' else 'minimizing'} Z.\n"
            
            # Plot optimal point
            ax.plot(opt_point[0], opt_point[1], 'r*', markersize=15, label=f'Optimal Solution')
            
            result_text = f"**Optimal Solution Found:**\n\n"
            result_text += f"**Objective Value (Z):** {opt_value:.4f}\n"
            result_text += f"**Variables:**\n - x1 = {opt_point[0]:.4f}\n - x2 = {opt_point[1]:.4f}"

        # Final UI display
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        info_frame = tk.Frame(self.results_frame, width=300)
        info_frame.pack(side="right", fill="both", expand=True, padx=10)

        tk.Label(info_frame, text="Solution Details", font=("Arial", 14, "bold")).pack(anchor="w")
        
        explanation_label = tk.Message(info_frame, text=explanation, width=300, justify="left", font=("Arial", 10))
        explanation_label.pack(pady=10, anchor="w")
        
        result_label = tk.Label(info_frame, text=result_text, justify="left", font=("Arial", 12, "bold"))
        result_label.pack(pady=20, anchor="w")

    def display_two_phase_solution(self):
        """Calculates and displays the two-phase method solution."""
        # --- This is a placeholder for the complex Two-Phase Simplex logic ---
        # A full implementation is extensive. This sets up the UI to display the results.
        
        explanation_frame = tk.LabelFrame(self.results_frame, text="Solution Process (Two-Phase Method)", padx=10, pady=10)
        explanation_frame.pack(pady=10, fill="x")

        explanation_text = (
            "The Two-Phase Simplex method is used for problems with '>=' or '=' constraints.\n\n"
            "**Phase 1:** An artificial objective function (sum of artificial variables) is minimized to find a basic feasible solution. "
            "If the minimum is 0, a feasible solution exists and we proceed to Phase 2. Otherwise, the problem is infeasible.\n"
            "   - Slack variables ('s') are added for '<=' constraints.\n"
            "   - Surplus ('s') and artificial ('a') variables are added for '>=' constraints.\n"
            "   - Artificial variables ('a') are added for '=' constraints.\n\n"
            "**Phase 2:** The original objective function is optimized using the standard Simplex method, starting from the feasible solution found in Phase 1."
        )
        tk.Message(explanation_frame, text=explanation_text, width=700).pack()

        # --- Placeholder for Iteration Tables ---
        table_frame = tk.LabelFrame(self.results_frame, text="Iterations", padx=10, pady=10)
        table_frame.pack(pady=10, fill="both", expand=True)
        
        # Add a horizontal scrollbar for the table
        x_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal")
        
        # Example: Creating a placeholder Treeview to show what an iteration table would look like
        cols = ('Basis', 'x1', 'x2', 's1', 's2', 'a1', 'RHS') # Example columns
        tree = ttk.Treeview(table_frame, columns=cols, show='headings', xscrollcommand=x_scrollbar.set)
        x_scrollbar.config(command=tree.xview)

        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=80)
        
        # Placeholder data
        tree.insert("", "end", values=('a1', '1', '2', '-1', '0', '1', '10'))
        tree.insert("", "end", values=('s2', '3', '1', '0', '1', '0', '15'))
        tree.insert("", "end", values=('Z', '-4', '-3', '1', '1', '0', '0'))
        
        x_scrollbar.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        
        tk.Label(table_frame, text="NOTE: The Two-Phase algorithm logic is not implemented. This is a UI placeholder.").pack(pady=10)

        # --- Final Result ---
        result_frame = tk.LabelFrame(self.results_frame, text="Final Result", padx=10, pady=10)
        result_frame.pack(pady=10, fill="x")
        
        result_text = "**Optimal Solution Found (Example):**\n\n"
        result_text += "**Objective Value (Z):** 75.0\n"
        result_text += "**Variables:**\n - x1 = 15.0\n - x2 = 5.0"
        tk.Label(result_frame, text=result_text, justify="left", font=("Arial", 12, "bold")).pack()
        
        # Add toggle for decimal/fraction
        # Note: The logic for conversion is not included in this placeholder
        toggle_button = tk.Button(result_frame, text="Toggle Decimal/Fraction")
        toggle_button.pack(pady=10)

if __name__ == "__main__":
    app = LinearProgrammingCalculator()
    app.run()