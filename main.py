import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import itertools
import math
import copy
import sys

# Configure CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Añadir función para rutas de recursos compatible con PyInstaller
def resource_path(relative_path):
    """Obtiene la ruta absoluta al recurso, compatible con PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class LinearProgrammingCalculator:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Investigacion de Operaciones - 303")
        self.root.geometry("1200x800")
        self.root.configure(fg_color="#1a1a1a")

        # colours
        self.colors = {"primary": "#FF5500", "background": "#1a1a1a",
                       "surface": "#2b2b2b", "text": "#ffffff",
                       "text_secondary": "#cccccc"}

        # session state
        self.session = {
            "num_variables": 2,
            "num_constraints": 1,
            "objective_coeffs": [],
            "constraint_coeffs": [],
            "constraint_signs": [],
            "constraint_values": [],
            "objective_type": "maximize",
            "solver": None,
            "graphical_result": None,
            "twophase_result": None,
        }

        self.setup_initial_screen()
        
    def setup_initial_screen(self):
        """Create the initial entry screen with logo and input form"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['background'])
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Logo section
        logo_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        logo_frame.pack(pady=(20, 10))
        
        # Cargar y mostrar el logo real centrado
        try:
            logo_img = Image.open(resource_path(AppConfig.LOGO_PATH)).convert("RGBA")
            logo_img = logo_img.resize((150, 150), Image.LANCZOS)
            self.logo_photo = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(150, 150))
            logo_label = ctk.CTkLabel(
                logo_frame,
                image=self.logo_photo,
                text="",
                width=150,
                height=150
            )
            logo_label.pack(anchor="center")
        except Exception as e:
            print("Error cargando el logo:", e)
            logo_placeholder = ctk.CTkLabel(
                logo_frame,
                text="UNIVERSIDAD DISTRITAL",
                font=ctk.CTkFont(size=20, weight="bold"),
                text_color=self.colors['primary']
            )
            logo_placeholder.pack(anchor="center")
        
        # Title section
        title_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        title_frame.pack(pady=(10, 5))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="Calculadora de Métodos Gráfico y Dos Fases",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors['text']
        )
        title_label.pack()

        # Poner Integrantes
        title_label = ctk.CTkLabel(
            title_frame,
            text="\nIntegrantes:",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors['text']
        )
        title_label.pack()
        
        # Developer names
        dev_label = ctk.CTkLabel(
            title_frame,
            text="Andres Felipe Franco Tellez (20221978031)  -  Gabriela Moreno Rojas (20221978026)  -  Andres Felipe Rincon Sanchez (20221978013)",
            font=ctk.CTkFont(size=15),
            text_color=self.colors['text_secondary']
        )
        dev_label.pack(pady=(5, 0))
        
        # Input form section
        form_frame = ctk.CTkFrame(main_frame, fg_color=self.colors['surface'])
        form_frame.pack(pady=(30, 20), padx=50, fill="x")
        
        # Variables input
        var_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        var_frame.pack(pady=15, padx=15, fill="x", anchor="center")
        var_frame.grid_columnconfigure(0, weight=1)
        var_frame.grid_columnconfigure(1, weight=1)
        
        var_label = ctk.CTkLabel(
            var_frame,
            text="Número de Variables (2-10):",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['text']
        )
        var_label.grid(row=0, column=0, sticky="e", padx=(0, 10))
        
        self.var_entry = ctk.CTkEntry(
            var_frame,
            width=120,
            font=ctk.CTkFont(size=18),
            placeholder_text="2"
        )
        self.var_entry.grid(row=0, column=1, sticky="w")
        self.var_entry.insert(0, str(self.session['num_variables']))
        
        # Constraints input
        const_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        const_frame.pack(pady=15, padx=15, fill="x", anchor="center")
        const_frame.grid_columnconfigure(0, weight=1)
        const_frame.grid_columnconfigure(1, weight=1)
        
        const_label = ctk.CTkLabel(
            const_frame,
            text="Número de Restricciones (2-15):",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['text']
        )
        const_label.grid(row=0, column=0, sticky="e", padx=(0, 10))
        
        self.const_entry = ctk.CTkEntry(
            const_frame,
            width=120,
            font=ctk.CTkFont(size=18),
            placeholder_text="2"
        )
        self.const_entry.grid(row=0, column=1, sticky="w")
        # Si el valor actual es menor que 2, poner 2 por defecto
        valor_default = max(2, self.session['num_constraints'])
        self.const_entry.insert(0, str(valor_default))
        
        # Submit button
        submit_btn = ctk.CTkButton(
            form_frame,
            text="Continuar",
            command=self.validate_and_continue,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=self.colors['primary'],
            hover_color="#e04400",
            height=50,
            width=240
        )
        submit_btn.pack(pady=20)
        
    def validate_and_continue(self):
        """Validar entradas y pasar a la pantalla de ingreso de datos"""
        try:
            num_vars = int(self.var_entry.get())
            num_consts = int(self.const_entry.get())
            
            if not (2 <= num_vars <= 10):
                messagebox.showerror("Entrada inválida", "El número de variables debe estar entre 2 y 10")
                return
                
            if not (2 <= num_consts <= 15):
                messagebox.showerror("Entrada inválida", "El número de restricciones debe estar entre 2 y 15")
                return
                
            # Update session data
            self.session['num_variables'] = num_vars
            self.session['num_constraints'] = num_consts
            
            # Initialize coefficient arrays
            self.session['objective_coeffs'] = [0] * num_vars
            self.session['constraint_coeffs'] = [[0] * num_vars for _ in range(num_consts)]
            self.session['constraint_signs'] = ['≤'] * num_consts
            self.session['constraint_values'] = [0] * num_consts
            
            self.setup_data_entry_screen()
            
        except ValueError:
            messagebox.showerror("Entrada inválida", "Por favor ingrese números válidos")
            
    def setup_data_entry_screen(self):
        """Create the data entry screen with objective function and constraints"""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main container with scrollable frame
        main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['background'])
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Configuración del Problema de Programación Lineal",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors['text']
        )
        title_label.pack()
        
        # Back button
        back_btn = ctk.CTkButton(
            header_frame,
            text="← Volver al Inicio",
            command=self.setup_initial_screen,
            font=ctk.CTkFont(size=15),
            fg_color="transparent",
            text_color=self.colors['primary'],
            hover_color="#2b2b2b",
            width=100
        )
        back_btn.pack(anchor="w", pady=(10, 0))
        
        # Scrollable frame for inputs
        scroll_frame = ctk.CTkScrollableFrame(main_frame, fg_color=self.colors['surface'])
        scroll_frame.pack(fill="both", expand=True)
        
        # Objective function section
        self.setup_objective_section(scroll_frame)
        
        # Constraints section
        self.setup_constraints_section(scroll_frame)
        
        # Solution buttons
        self.setup_solution_buttons(main_frame)
        
    def setup_objective_section(self, parent):
        """Setup objective function input section"""
        obj_frame = ctk.CTkFrame(parent, fg_color=self.colors['background'])
        obj_frame.pack(fill="x", pady=15, padx=15)
        obj_frame.grid_columnconfigure(tuple(range(self.session['num_variables']*2+1)), weight=1)
        
        # Objective function header
        obj_header = ctk.CTkLabel(
            obj_frame,
            text="Función Objetivo",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors['primary']
        )
        obj_header.pack(pady=(15, 10))
        
        # Objective type selection
        obj_type_frame = ctk.CTkFrame(obj_frame, fg_color="transparent")
        obj_type_frame.pack(pady=10)
        
        self.obj_type_var = tk.StringVar(value=self.session['objective_type'])
        
        max_radio = ctk.CTkRadioButton(
            obj_type_frame,
            text="Maximizar",
            variable=self.obj_type_var,
            value="maximize",
            font=ctk.CTkFont(size=16),
            text_color=self.colors['text']
        )
        max_radio.pack(side="left", padx=(0, 20))
        
        min_radio = ctk.CTkRadioButton(
            obj_type_frame,
            text="Minimizar",
            variable=self.obj_type_var,
            value="minimize",
            font=ctk.CTkFont(size=16),
            text_color=self.colors['text']
        )
        min_radio.pack(side="left")
        
        # Objective coefficients
        coeffs_frame = ctk.CTkFrame(obj_frame, fg_color="transparent")
        coeffs_frame.pack(pady=15, padx=15, fill="x")
        coeffs_frame.grid_columnconfigure(tuple(range(self.session['num_variables']*2+1)), weight=1)
        
        obj_label = ctk.CTkLabel(
            coeffs_frame,
            text="Z = ",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['text']
        )
        obj_label.pack(side="left")
        
        self.obj_entries = []
        for i in range(self.session['num_variables']):
            if i > 0:
                plus_label = ctk.CTkLabel(
                    coeffs_frame,
                    text=" + ",
                    font=ctk.CTkFont(size=16),
                    text_color=self.colors['text']
                )
                plus_label.pack(side="left")
                
            entry = ctk.CTkEntry(
                coeffs_frame,
                width=80,
                font=ctk.CTkFont(size=16),
                placeholder_text="0"
            )
            entry.pack(side="left", padx=2)
            entry.insert(0, str(self.session['objective_coeffs'][i]))
            self.obj_entries.append(entry)
            
            var_label = ctk.CTkLabel(
                coeffs_frame,
                text=f"x{i+1}",
                font=ctk.CTkFont(size=16),
                text_color=self.colors['text']
            )
            var_label.pack(side="left", padx=(2, 5))
            
    def setup_constraints_section(self, parent):
        """Setup constraints input section"""
        const_frame = ctk.CTkFrame(parent, fg_color=self.colors['background'])
        const_frame.pack(fill="x", pady=15, padx=15, anchor="center")
        
        # Constraints header
        const_header = ctk.CTkLabel(
            const_frame,
            text="Restricciones",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.colors['primary']
        )
        const_header.pack(pady=(15, 10))
        
        # Constraint inputs
        self.constraint_entries = []
        self.constraint_signs = []
        self.constraint_values = []
        
        for i in range(self.session['num_constraints']):
            constraint_row = ctk.CTkFrame(const_frame, fg_color="transparent")
            constraint_row.pack(fill="x", pady=5, padx=15)
            constraint_row.grid_columnconfigure(tuple(range(self.session['num_variables']*2+3)), weight=1)
            
            # Constraint coefficients
            row_entries = []
            for j in range(self.session['num_variables']):
                if j > 0:
                    plus_label = ctk.CTkLabel(
                        constraint_row,
                        text=" + ",
                        font=ctk.CTkFont(size=16),
                        text_color=self.colors['text']
                    )
                    plus_label.pack(side="left")
                    
                entry = ctk.CTkEntry(
                    constraint_row,
                    width=80,
                    font=ctk.CTkFont(size=16),
                    placeholder_text="0"
                )
                entry.pack(side="left", padx=2)
                entry.insert(0, str(self.session['constraint_coeffs'][i][j]))
                row_entries.append(entry)
                
                var_label = ctk.CTkLabel(
                    constraint_row,
                    text=f"x{j+1}",
                    font=ctk.CTkFont(size=16),
                    text_color=self.colors['text']
                )
                var_label.pack(side="left", padx=(2, 5))
            
            self.constraint_entries.append(row_entries)
            
            # Inequality sign dropdown
            sign_dropdown = ctk.CTkComboBox(
                constraint_row,
                values=["≤", "≥", "="],
                width=80,
                font=ctk.CTkFont(size=16),
                state="readonly"
            )
            sign_dropdown.pack(side="left", padx=10)
            sign_dropdown.set(self.session['constraint_signs'][i])
            self.constraint_signs.append(sign_dropdown)
            
            # Constraint value
            value_entry = ctk.CTkEntry(
                constraint_row,
                width=100,
                font=ctk.CTkFont(size=16),
                placeholder_text="0"
            )
            value_entry.pack(side="left", padx=10)
            value_entry.insert(0, str(self.session['constraint_values'][i]))
            self.constraint_values.append(value_entry)
            
    def setup_solution_buttons(self, parent):
        """Setup solution method buttons"""
        button_frame = ctk.CTkFrame(parent, fg_color="transparent")
        button_frame.pack(fill="x", pady=20)
        
        # Graphical Method button
        self.graphical_btn = ctk.CTkButton(
            button_frame,
            text="Método Gráfico",
            command=self.show_graphical_solution,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=self.colors['primary'] if self.session['num_variables'] == 2 else "#666666",
            hover_color="#e04400" if self.session['num_variables'] == 2 else "#666666",
            state="normal" if self.session['num_variables'] == 2 else "disabled",
            height=40,
            width=180
        )
        self.graphical_btn.pack(side="left", padx=(20, 10))
        
        # Two-Phase Method button
        self.twophase_btn = ctk.CTkButton(
            button_frame,
            text="Método Dos Fases",
            command=self.show_twophase_solution,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=self.colors['primary'],
            hover_color="#e04400",
            height=40,
            width=180
        )
        self.twophase_btn.pack(side="left", padx=10)
        
        # Update button states based on variable count
        if self.session['num_variables'] != 2:
            tooltip_label = ctk.CTkLabel(
                button_frame,
                text="(El método gráfico solo está disponible para 2 variables)",
                font=ctk.CTkFont(size=10),
                text_color=self.colors['text_secondary']
            )
            tooltip_label.pack(side="left", padx=10)
            
    def collect_input_data(self):
        """Collect and validate all input data"""
        try:
            # Collect objective coefficients
            obj_coeffs = []
            for entry in self.obj_entries:
                obj_coeffs.append(float(entry.get()) if entry.get() else 0)
                
            # Collect constraint data
            constraint_coeffs = []
            constraint_signs = []
            constraint_values = []
            
            for i in range(self.session['num_constraints']):
                row_coeffs = []
                for entry in self.constraint_entries[i]:
                    row_coeffs.append(float(entry.get()) if entry.get() else 0)
                constraint_coeffs.append(row_coeffs)
                
                constraint_signs.append(self.constraint_signs[i].get())
                constraint_values.append(float(self.constraint_values[i].get()) if self.constraint_values[i].get() else 0)
                
            # Update session data
            self.session['objective_coeffs'] = obj_coeffs
            self.session['constraint_coeffs'] = constraint_coeffs
            self.session['constraint_signs'] = constraint_signs
            self.session['constraint_values'] = constraint_values
            self.session['objective_type'] = self.obj_type_var.get()
            
            return True
            
        except ValueError:
            messagebox.showerror("Entrada inválida", "Por favor ingrese valores numéricos válidos")
            return False
            
    def show_graphical_solution(self):
        """Display graphical method solution"""
        if not self.collect_input_data():
            return
            
        # Instantiate solver
        solver = LPSolver(
            self.session['objective_coeffs'],
            self.session['constraint_coeffs'],
            self.session['constraint_signs'],
            self.session['constraint_values'],
            self.session['objective_type']
        )
        
        try:
            result = solver.solve_graphical()
            self.session['graphical_result'] = result

            if result.get('status') != "Optimal" or 'feasible_points' not in result:
                messagebox.showerror("Método Gráfico", "No se encontró una región factible o el problema no tiene solución gráfica.")
                return
        except ValueError as e:
            messagebox.showerror("Error en Método Gráfico", str(e))
            return
            
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['background'])
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Solución por Método Gráfico",
            font=ctk.CTkFont(size=25, weight="bold"),
            text_color=self.colors['primary']
        )
        title_label.pack()
        
        # Back button
        back_btn = ctk.CTkButton(
            header_frame,
            text="← Volver a Datos",
            command=self.setup_data_entry_screen,
            font=ctk.CTkFont(size=15),
            fg_color="transparent",
            text_color=self.colors['primary'],
            hover_color="#2b2b2b",
            width=120
        )
        back_btn.pack(anchor="w", pady=(10, 0))
        
        # Content frame
        content_frame = ctk.CTkFrame(main_frame, fg_color=self.colors['surface'])
        content_frame.pack(fill="both", expand=True)
        
        # Plot section
        plot_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        plot_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        
        # Determinar el rango global de x para graficar todas las restricciones
        all_points = result['feasible_points'][:]
        for a, b, c in result['constraint_lines']:
            if a != 0:
                all_points.append((c / a, 0))
            if b != 0:
                all_points.append((0, c / b))
        if all_points:
            x_vals = [p[0] for p in all_points]
            y_vals = [p[1] for p in all_points]
            x_min_plot = min(0, min(x_vals))
            x_max_plot = max(5, max(x_vals) * 1.2)
            y_min_plot = min(0, min(y_vals))
            y_max_plot = max(5, max(y_vals) * 1.2)
        else:
            x_min_plot, x_max_plot = 0, 10
            y_min_plot, y_max_plot = 0, 10
        x = np.linspace(x_min_plot, x_max_plot, 400)
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min_plot, y_max_plot)

        # Graficar restricciones en todo el rango de x
        for i, (a, b, c) in enumerate(result['constraint_lines']):
            if self.session['constraint_signs'][i] == "≥":
                # Invertir la desigualdad para representarlo como "≤"
                a, b, c = -a, -b, -c

            if b == 0 and a != 0:  # Línea vertical x = c/a
                x_val = c / a
                ax.axvline(x=x_val, color=f'C{i}', linestyle='--', label=f'Restricción {i+1}')
            elif b != 0:
                # Verifica que el rango de x sea adecuado para calcular y
                if a != 0:
                    y = (c - a * x) / b
                    ax.plot(x, y, color=f'C{i}', linestyle='--', label=f'Restricción {i+1}')
                else:
                    # Para el caso donde b == 0, si a también es 0, no hace sentido graficar
                    pass

        # Dibujar la región factible usando ConvexHull si hay suficientes puntos
        feasible = result['feasible_points']
        if result['status'] == "Optimal" and len(feasible) >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(feasible)
                hull_points = [feasible[v] for v in hull.vertices]
                hull_points.append(hull_points[0])
                feasible_x = [p[0] for p in hull_points]
                feasible_y = [p[1] for p in hull_points]
                ax.fill(feasible_x, feasible_y, alpha=0.3, color=self.colors['primary'], label='Región Factible')
            except ImportError:
                # Si no hay scipy, usar orden angular como antes
                cx = np.mean([p[0] for p in feasible])
                cy = np.mean([p[1] for p in feasible])
                feasible_sorted = sorted(feasible, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
                feasible_x = [p[0] for p in feasible_sorted] + [feasible_sorted[0][0]]
                feasible_y = [p[1] for p in feasible_sorted] + [feasible_sorted[0][1]]
                ax.fill(feasible_x, feasible_y, alpha=0.3, color=self.colors['primary'], label='Región Factible')
        elif result['status'] == "Optimal" and len(feasible) == 2:
            feasible_x = [p[0] for p in feasible]
            feasible_y = [p[1] for p in feasible]
            ax.plot(feasible_x, feasible_y, color=self.colors['primary'], alpha=0.5, label='Región Factible')

        # Línea objetivo
        if result['status'] == "Optimal" and result['optimal_point']:
            opt_x, opt_y = result['optimal_point']
            c = self.session['objective_coeffs']
            z_line_val = c[0] * opt_x + c[1] * opt_y
            if len(c) >= 2:
                if c[1] != 0:
                    # La línea objetivo debe pasar por el punto óptimo
                    y_obj = (z_line_val - c[0] * x) / c[1]
                    ax.plot(x, y_obj, 'k--', label='Línea Objetivo (Z)')
                elif c[0] != 0:
                    x_obj = z_line_val / c[0]
                    ax.axvline(x=x_obj, color='k', linestyle='--', label='Línea Objetivo (Z)')

        # Punto óptimo
        if result['status'] == "Optimal" and result['optimal_point']:
            opt_x, opt_y = result['optimal_point']
            ax.plot(opt_x, opt_y, 'go', markersize=10, label=f'Solución Óptima ({opt_x:.2f}, {opt_y:.2f})')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3) # x2 >= 0
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3) # x1 >= 0
        
        ax.set_xlabel('x₁', color='white')
        ax.set_ylabel('x₂', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3)
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Results table section
        table_frame = ctk.CTkFrame(content_frame, fg_color=self.colors['background'])
        table_frame.pack(side="right", fill="y", padx=(0, 15), pady=15)
        
        table_title = ctk.CTkLabel(
            table_frame,
            text="Puntos de Intersección",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['primary']
        )
        table_title.pack(pady=(10, 15))
        
        # Display intersection points and their objective values
        points_data = []
        if result['status'] == "Optimal":
            for pt in result['feasible_points']:
                z_val = self.session['objective_coeffs'][0] * pt[0] + self.session['objective_coeffs'][1] * pt[1]
                points_data.append((f"({pt[0]:.2f}, {pt[1]:.2f})", f"{z_val:.2f}"))
            
            # Mark optimal point
            optimal_point_str = f"({result['optimal_point'][0]:.2f}, {result['optimal_point'][1]:.2f})"
            for i, (point_str, val_str) in enumerate(points_data):
                if point_str == optimal_point_str:
                    points_data[i] = (point_str, val_str + "*")

        if points_data:
            for point, value in points_data:
                point_frame = ctk.CTkFrame(table_frame, fg_color="transparent")
                point_frame.pack(fill="x", pady=2, padx=10)
                
                point_label = ctk.CTkLabel(
                    point_frame,
                    text=f"Punto: {point}",
                    font=ctk.CTkFont(size=11),
                    text_color=self.colors['text']
                )
                point_label.pack(side="left")
                
                value_label = ctk.CTkLabel(
                    point_frame,
                    text=f"Z = {value}",
                    font=ctk.CTkFont(size=11, weight="bold" if "*" in value else "normal"),
                    text_color=self.colors['primary'] if "*" in value else self.colors['text']
                )
                value_label.pack(side="right")
        else:
            no_points_label = ctk.CTkLabel(
                table_frame,
                text="No se encontraron puntos factibles o el problema es no acotado.",
                font=ctk.CTkFont(size=11),
                text_color=self.colors['text_secondary']
            )
            no_points_label.pack(pady=10)


        # Optimal solution highlight
        optimal_frame = ctk.CTkFrame(table_frame, fg_color=self.colors['primary'])
        optimal_frame.pack(fill="x", pady=(15, 10), padx=10)
        
        if result['status'] == "Optimal":
            optimal_label = ctk.CTkLabel(
                optimal_frame,
                text=f"Solución Óptima: ({result['optimal_point'][0]:.2f}, {result['optimal_point'][1]:.2f})\nValor Objetivo: {result['optimal_value']:.2f}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="white"
            )
        else:
            optimal_label = ctk.CTkLabel(
                optimal_frame,
                text=f"Estado: {result['status']}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="white"
            )
        optimal_label.pack(pady=10)
        
    def show_twophase_solution(self):
        """Display two-phase method solution"""
        if not self.collect_input_data():
            return
            
        # Validate input data
        try:
            # Check for valid coefficients
            if not any(abs(c) > 1e-10 for c in self.session['objective_coeffs']):
                messagebox.showwarning("Advertencia", "La función objetivo tiene todos los coeficientes iguales a cero.")
            
            # Check for valid constraints
            if not any(any(abs(c) > 1e-10 for c in row) for row in self.session['constraint_coeffs']):
                messagebox.showerror("Error", "Todas las restricciones tienen coeficientes iguales a cero.")
                return
                
        except Exception as e:
            messagebox.showerror("Error de validación", f"Error validando datos: {str(e)}")
            return
            
        # Instantiate solver
        try:
            solver = LPSolver(
                self.session['objective_coeffs'],
                self.session['constraint_coeffs'],
                self.session['constraint_signs'],
                self.session['constraint_values'],
                self.session['objective_type']
            )
        except Exception as e:
            messagebox.showerror("Error de inicialización", f"Error inicializando el solver: {str(e)}")
            return

        try:
            result = solver.solve_two_phase()
            self.session['twophase_result'] = result
        except ValueError as e:
            messagebox.showerror("Error en Método Dos Fases", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error inesperado", f"Error inesperado en el método dos fases: {str(e)}")
            return
            
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=self.colors['background'])
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Solución por Método Dos Fases",
            font=ctk.CTkFont(size=25, weight="bold"),
            text_color=self.colors['primary']
        )
        title_label.pack()
        
        # Back button
        back_btn = ctk.CTkButton(
            header_frame,
            text="← Volver a Datos",
            command=self.setup_data_entry_screen,
            font=ctk.CTkFont(size=15),
            fg_color="transparent",
            text_color=self.colors['primary'],
            hover_color="#2b2b2b",
            width=120
        )
        back_btn.pack(anchor="w", pady=(10, 0))
        
        # Tabview for phases
        tabview = ctk.CTkTabview(main_frame, fg_color=self.colors['surface'])
        tabview.pack(fill="both", expand=True)
        
        # Phase 1 tab
        phase1_tab = tabview.add("Fase 1")
        if result.get('phase1_iterations'):
            var_names_phase1 = result.get('variable_names_phase1', [])
            self.create_phase_table(phase1_tab, "Fase 1: Encontrar Solución Básica Factible Inicial", result['phase1_iterations'], var_names_phase1)
        else:
            status_text = result.get('status', 'Desconocido')
            ctk.CTkLabel(phase1_tab, text=f"Estado de Fase 1: {status_text}\nNo hay iteraciones que mostrar.", text_color=self.colors['text']).pack(pady=20)
        
        # Phase 2 tab
        phase2_tab = tabview.add("Fase 2")
        if result.get('phase2_iterations'):
            var_names_phase2 = result.get('variable_names_phase2', [])
            self.create_phase_table(phase2_tab, "Fase 2: Optimizar Problema Original", result['phase2_iterations'], var_names_phase2)
        else:
            status_text = result.get('status', 'Desconocido')
            if result['status'] in ['Infactible', 'No acotado']:
                ctk.CTkLabel(phase2_tab, text=f"Estado: {status_text}\nNo se pudo completar la Fase 2.", text_color=self.colors['text']).pack(pady=20)
            else:
                ctk.CTkLabel(phase2_tab, text=f"Estado de Fase 2: {status_text}\nNo hay iteraciones que mostrar.", text_color=self.colors['text']).pack(pady=20)
        
        # Summary tab
        summary_tab = tabview.add("Resumen")
        self.create_summary_section(summary_tab, result)
        
    def create_phase_table(self, parent, title, iterations_data, var_names):
        """Create a phase table with simplex iterations using a grid of labels."""
        title_label = ctk.CTkLabel(
            parent,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['primary']
        )
        title_label.pack(pady=(10, 15))

        scrollable_area = ctk.CTkScrollableFrame(parent, fg_color=self.colors['background'])
        scrollable_area.pack(fill="both", expand=True, padx=10, pady=10)

        if not iterations_data:
            ctk.CTkLabel(scrollable_area, text="No hay iteraciones para mostrar.", text_color=self.colors['text']).pack(pady=20)
            return

        for iteration_num, tableau_state, basic_vars_indices, enter_col, leave_row in iterations_data:
            iteration_frame = ctk.CTkFrame(scrollable_area, fg_color=self.colors['surface'])
            iteration_frame.pack(fill="x", pady=(10, 5), padx=5)

            iteration_title = ctk.CTkLabel(
                iteration_frame,
                text=f"Iteración {iteration_num}",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.colors['primary']
            )
            iteration_title.pack(pady=5)

            table_grid_frame = ctk.CTkFrame(iteration_frame, fg_color="transparent")
            table_grid_frame.pack(fill="x", expand=True, padx=10, pady=5)

            m, n = tableau_state.shape
            num_vars = n - 1
            header = ['Base'] + (var_names[:num_vars] if var_names else [f'x{i+1}' for i in range(num_vars)]) + ['RHS']

            # Create Header
            for j, col_name in enumerate(header):
                is_entering_col = (enter_col is not None and j == enter_col + 1)
                header_color = "#e04400" if is_entering_col else self.colors['primary']
                
                header_label = ctk.CTkLabel(
                    table_grid_frame,
                    text=col_name,
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=header_color,
                    fg_color=self.colors['background'],
                    padx=5, pady=5
                )
                header_label.grid(row=0, column=j, sticky="nsew", padx=1, pady=1)
                table_grid_frame.grid_columnconfigure(j, weight=1)

            # Populate table with data
            for i in range(m):
                # Base Variable Name
                is_pivot_row = (leave_row is not None and i == leave_row)
                row_fg_color = self.colors['surface']

                base_var_name = 'Z'
                if i < m - 1: # If not the Z-row
                    base_var_idx = basic_vars_indices[i]
                    base_var_name = var_names[base_var_idx] if 0 <= base_var_idx < len(var_names) else '?'
                
                base_label = ctk.CTkLabel(table_grid_frame, text=base_var_name, fg_color=row_fg_color, font=ctk.CTkFont(size=10))
                base_label.grid(row=i + 1, column=0, sticky="nsew", padx=1, pady=1)

                # Data cells
                for j in range(n):
                    cell_val = tableau_state[i, j]
                    is_pivot = (is_pivot_row and enter_col is not None and j == enter_col)
                    is_entering_col = (enter_col is not None and j == enter_col)

                    cell_font = ctk.CTkFont(size=10, weight="bold" if is_pivot else "normal")
                    cell_fg_color = "#552200" # Pivot background
                    if not is_pivot:
                         cell_fg_color = "#3a3a3a" if is_entering_col else row_fg_color
                    
                    # Correct sign for Z value display
                    if i == m - 1 and j == n - 1: # Z-row RHS value
                        if "Fase 2" in title: # Only flip sign in phase 2
                           cell_val *= -1

                    cell_label = ctk.CTkLabel(
                        table_grid_frame,
                        text=f"{cell_val:.3f}",
                        fg_color=cell_fg_color,
                        font=cell_font
                    )
                    cell_label.grid(row=i + 1, column=j + 1, sticky="nsew", padx=1, pady=1)
                        
    def create_summary_section(self, parent, result):
        """Create summary section with final results"""
        summary_frame = ctk.CTkFrame(parent, fg_color=self.colors['background'])
        summary_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        solution_frame = ctk.CTkFrame(summary_frame, fg_color=self.colors['primary'])
        solution_frame.pack(fill="x", pady=(0, 20))
        
        solution_title = ctk.CTkLabel(
            solution_frame,
            text="Solución Final",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        )
        solution_title.pack(pady=(15, 10))
        
        if result.get('status') == "Optimal":
            solution_text = "Solución Óptima Encontrada:\n"
            if result.get('optimal_solution'):
                for var, val in result['optimal_solution'].items():
                    if var.startswith('x'):
                        solution_text += f"{var} = {val:.3f}\n"
            solution_text += f"Valor Objetivo (Z) = {result.get('optimal_value', 0):.3f}\n"
            solution_text += f"Estado: {result.get('status')}"
        else:
            solution_text = f"Estado: {result.get('status', 'Desconocido')}\n"
            if result.get('status') == "Infeasible":
                solution_text += "El problema no tiene solución factible."
            elif result.get('status') == "Unbounded":
                solution_text += "La función objetivo es no acotada."
        
        solution_label = ctk.CTkLabel(
            solution_frame,
            text=solution_text,
            font=ctk.CTkFont(size=12),
            text_color="white",
            justify="center"
        )
        solution_label.pack(pady=(0, 15))

        # Phase summary
        phase_summary_frame = ctk.CTkFrame(summary_frame, fg_color=self.colors['surface'])
        phase_summary_frame.pack(fill="x", pady=10)
        
        phase_title = ctk.CTkLabel(
            phase_summary_frame,
            text="Resumen de Fases",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=self.colors['primary']
        )
        phase_title.pack(pady=(15, 10))
        
        p1_iters = len(result.get('phase1_iterations', []))
        p2_iters = len(result.get('phase2_iterations', []))

        phase1_text = f"Fase 1: Completada en {p1_iters} iteraciones."
        phase2_text = f"Fase 2: Completada en {p2_iters} iteraciones."
        if result.get('status') != "Optimal":
            phase2_text = f"Fase 2: No se completó. Estado: {result.get('status')}"

        ctk.CTkLabel(phase_summary_frame, text=phase1_text, font=ctk.CTkFont(size=11), text_color=self.colors['text']).pack(anchor="w", padx=15)
        ctk.CTkLabel(phase_summary_frame, text=phase2_text, font=ctk.CTkFont(size=11), text_color=self.colors['text']).pack(anchor="w", padx=15, pady=(0,15))
                
    def run(self):
        """Start the application"""
        self.root.mainloop()

# Additional utility functions for future implementation
class LPSolver:
    """Linear‑Programming solver providing:

    • Graphical method for exactly two decision variables.
    • Two‑phase Simplex for 2–10 variables and 1–15 constraints.
    """

    TOL = 1e-8  # numerical tolerance

    def __init__(self, obj_coeffs, cons_coeffs, cons_signs, cons_values,
                 obj_type="maximize"):
        self.c = np.asarray(obj_coeffs, dtype=float)
        self.A = np.asarray(cons_coeffs, dtype=float)
        self.signs = cons_signs[:]
        self.b = np.asarray(cons_values, dtype=float)
        self.obj_type = obj_type  # "maximize" or "minimize"

        self.n = len(self.c)      # original variables
        self.m = len(self.b)      # constraints

        if self.A.shape != (self.m, self.n):
            raise ValueError("Coefficient matrix dimensions mismatch")

    # --------------------------  Graphical Method  -------------------------

    def solve_graphical(self):
        if self.n != 2:
            raise ValueError("Graphical method only supports 2 variables")

        # Build augmented list of constraints including non‑negativity
        constraints = [(*self.A[i], self.signs[i], self.b[i])
                       for i in range(self.m)]
        constraints += [(1, 0, "≥", 0), (0, 1, "≥", 0)]  # x1>=0, x2>=0

        # Convert '= / ≥' to '≤' by multiplying by −1 where necessary
        half_planes = []  # (a, b, c) for ax + by ≤ c
        for a, b, s, rhs in constraints:
            if s == "=":
                half_planes.append(( a,  b,  rhs))
                half_planes.append((-a, -b, -rhs))
            elif s == "≤":
                half_planes.append((a, b, rhs))
            elif s == "≥":
                half_planes.append((-a, -b, -rhs))
            else:
                raise ValueError("Unknown sign")

        # Enumerate all pair‑wise intersections of bounding lines
        lines = [(a, b, c) for a, b, c in half_planes]
        pts = []
        for i, (a1, b1, c1) in enumerate(lines):
            for j, (a2, b2, c2) in enumerate(lines):
                if i >= j: continue # Avoid duplicate pairs and self-intersections
                
                det = a1 * b2 - a2 * b1
                if abs(det) < self.TOL:
                    continue  # parallel or same line
                x = (c1 * b2 - c2 * b1) / det
                y = (a1 * c2 - a2 * c1) / det
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                pts.append((x, y))

        # Filter feasible points
        feasible = []
        for x, y in pts:
            # Check non-negativity first explicitly as it's part of the problem formulation
            if x < -self.TOL or y < -self.TOL:
                continue
            
            is_feasible = True
            for a, b, s, rhs in constraints:
                # Need to check original constraints, not just converted half-planes,
                # as half_planes might have reversed inequalities.
                # A direct check against original constraints is safer.
                val = a * x + b * y
                if s == "≤" and val > rhs + self.TOL:
                    is_feasible = False
                    break
                elif s == "≥" and val < rhs - self.TOL:
                    is_feasible = False
                    break
                elif s == "=" and abs(val - rhs) > self.TOL:
                    is_feasible = False
                    break
            if is_feasible:
                feasible.append((x, y))

        if not feasible:
            # Check for unbounded case or simple infeasibility if no points found
            # A more rigorous check for unboundedness in graphical method would involve
            # checking if objective can be moved infinitely in feasible direction.
            # For simplicity here, we'll return Infeasible if no feasible points.
            return {"status": "Infeasible"}

        # Objective evaluation
        z_vals = [self.c[0] * x + self.c[1] * y for x, y in feasible]
        
        if not z_vals: # No feasible points, could be truly infeasible
            return {"status": "Infeasible"}

        if self.obj_type == "maximize":
            idx = int(np.argmax(z_vals))
        else: # minimize
            idx = int(np.argmin(z_vals))
        
        opt_pt = feasible[idx]
        opt_val = z_vals[idx]

        # Sort feasible vertices for polygon shading (angle sort)
        if len(feasible) > 2: # Need at least 3 points for a polygon
            cx = np.mean([p[0] for p in feasible])
            cy = np.mean([p[1] for p in feasible])
            feasible_sorted = sorted(feasible, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        else:
            feasible_sorted = feasible # Can't form a polygon with less than 3 points
            

        return {
            "status": "Optimal",
            "optimal_point": opt_pt,
            "optimal_value": opt_val,
            "feasible_points": feasible,
            "feasible_region": feasible_sorted,
            "constraint_lines": [(a, b, rhs) for a, b, s, rhs in constraints if s != "≥"], # only <= and = for plotting lines
            "all_constraint_equations": constraints # All original constraints for plotting lines
        }

    # --------------------------  Two‑Phase Simplex  ------------------------

    def _standard_form(self):
        """Return tableau, basic variable list and variable names in standard
        form suitable for two‑phase simplex.
        """
        # Ensure RHS positive by row multiplication when needed
        A_prime, b_prime, signs_prime = [], [], []
        for row, s, rhs in zip(self.A, self.signs, self.b):
            row = list(row)
            rhs = float(rhs)
            if rhs < 0:
                row = [-v for v in row]
                rhs = -rhs
                # Flip sign if original constraint was <= or >=
                if s == "≤": s = "≥"
                elif s == "≥": s = "≤"
            A_prime.append(row)
            b_prime.append(rhs)
            signs_prime.append(s)

        A_prime = np.asarray(A_prime, dtype=float)
        b_prime = np.asarray(b_prime, dtype=float)

        var_names = [f"x{i+1}" for i in range(self.n)]
        
        # Determine total number of variables for the tableau including original, slack, surplus, artificial
        num_slack_surplus = sum(1 for s in signs_prime if s in ["≤", "≥"])
        num_artificial = sum(1 for s in signs_prime if s in ["=", "≥"])
        
        total_vars = self.n + num_slack_surplus + num_artificial
        tableau = np.zeros((self.m, total_vars + 1)) # +1 for RHS

        basic_vars_indices = [] # Indices of basic variables in the full tableau's var_names
        current_slack_surplus_idx = 0
        current_artificial_idx = 0
        
        # Populate tableau with original coefficients and RHS
        tableau[:, :self.n] = A_prime
        tableau[:, -1] = b_prime

        # Add slack, surplus, and artificial variables
        col_offset = self.n
        for i, s in enumerate(signs_prime):
            if s == "≤":
                tableau[i, col_offset] = 1 # Slack variable
                var_names.append(f"s{current_slack_surplus_idx+1}")
                basic_vars_indices.append(col_offset) # Slack vars are initially basic
                col_offset += 1
                current_slack_surplus_idx += 1
            elif s == "≥":
                tableau[i, col_offset] = -1 # Surplus variable
                var_names.append(f"s{current_slack_surplus_idx+1}")
                col_offset += 1
                current_slack_surplus_idx += 1

                tableau[i, col_offset] = 1 # Artificial variable
                var_names.append(f"a{current_artificial_idx+1}")
                basic_vars_indices.append(col_offset) # Artificial vars are initially basic
                col_offset += 1
                current_artificial_idx += 1
            elif s == "=":
                tableau[i, col_offset] = 1 # Artificial variable
                var_names.append(f"a{current_artificial_idx+1}")
                basic_vars_indices.append(col_offset) # Artificial vars are initially basic
                col_offset += 1
                current_artificial_idx += 1
            else:
                raise ValueError("Unknown sign")

        artificial_cols = [idx for idx, name in enumerate(var_names) if name.startswith('a')]
        
        # The basic_vars passed back should be the column indices in the current tableau
        # after augmentation.
        
        return tableau, basic_vars_indices, var_names, artificial_cols

    @staticmethod
    def _pivot(mat, row, col):
        """Perform pivot on (row,col) in tableau mat in‑place"""
        m, n = mat.shape
        pivot_val = mat[row, col]
        if abs(pivot_val) < 1e-9: # Avoid division by zero or very small numbers
            raise ValueError("Elemento pivote demasiado pequeño o cero.")
        
        mat[row, :] = mat[row, :] / pivot_val
        for r in range(m):
            if r != row:
                mat[r, :] -= mat[r, col] * mat[row, :]

    def solve_two_phase(self):
        """Two-phase simplex algorithm returning a detailed solution dict."""
        tableau_initial, basic_vars_initial, var_names_full, art_cols = self._standard_form()
        
        if not art_cols:
            return self._solve_simplex_phase2_only(tableau_initial, basic_vars_initial, var_names_full)

        # --- Phase 1 Setup ---
        tableau = copy.deepcopy(tableau_initial)
        basic_vars = copy.deepcopy(basic_vars_initial)
        m, n = tableau.shape
        z_row_phase1 = np.zeros(n)
        for a_col_idx in art_cols:
            z_row_phase1[a_col_idx] = -1
        tableau = np.vstack([tableau, z_row_phase1])
        for i, b_var_idx in enumerate(basic_vars):
            if b_var_idx in art_cols:
                tableau[-1, :] += tableau[i, :]

        # --- Phase 1 Iterations ---
        it = 1
        phase1_iterations = [(it, copy.deepcopy(tableau), copy.deepcopy(basic_vars), None, None)]
        
        while True:
            current_iter_data = phase1_iterations[-1]
            current_tableau = current_iter_data[1]
            
            obj_coeffs = current_tableau[-1, :-1]
            enter_candidates = [j for j, v in enumerate(obj_coeffs) if v > self.TOL]
            if not enter_candidates:
                break # Optimal for Phase 1

            enter_col = max(enter_candidates, key=lambda j: obj_coeffs[j])
            
            ratios = [(current_tableau[i, -1] / current_tableau[i, enter_col], i) for i in range(m) if current_tableau[i, enter_col] > self.TOL]
            if not ratios:
                phase1_iterations[-1] = (current_iter_data[0], current_tableau, current_iter_data[2], enter_col, None)
                return {"status": "Infactible", "phase1_iterations": phase1_iterations, "phase2_iterations": []}

            leave_row = min(ratios, key=lambda x: x[0])[1]
            
            # Update current iteration with the pivot info before performing the pivot
            phase1_iterations[-1] = (current_iter_data[0], current_tableau, current_iter_data[2], enter_col, leave_row)

            # Create the state for the *next* iteration
            it += 1
            next_tableau = copy.deepcopy(current_tableau)
            next_basic_vars = copy.deepcopy(current_iter_data[2])
            self._pivot(next_tableau, leave_row, enter_col)
            next_basic_vars[leave_row] = enter_col
            phase1_iterations.append((it, next_tableau, next_basic_vars, None, None))

        # Check for infeasibility
        if abs(phase1_iterations[-1][1][-1, -1]) > self.TOL:
            return {"status": "Infactible", "phase1_iterations": phase1_iterations, "phase2_iterations": []}
        
        # --- Phase 2 Setup ---
        final_phase1_vars = phase1_iterations[-1][2]
        original_and_slack_surplus_vars = [idx for idx, name in enumerate(var_names_full) if not name.startswith('a')]
        new_tableau_cols = original_and_slack_surplus_vars + [n-1]
        tableau_phase2 = phase1_iterations[-1][1][:-1, new_tableau_cols] # Get final tableau, remove Z-row, filter columns
        var_names_phase2 = [var_names_full[i] for i in original_and_slack_surplus_vars]
        
        old_to_new_col_map = {old_idx: new_idx for new_idx, old_idx in enumerate(original_and_slack_surplus_vars)}
        basic_vars_phase2 = [old_to_new_col_map[b] for b in final_phase1_vars if b in old_to_new_col_map]
        
        m_phase2, n_phase2 = tableau_phase2.shape
        obj_row_phase2 = np.zeros(n_phase2)
        for i in range(self.n):
            obj_row_phase2[i] = -self.c[i]
        tableau_phase2 = np.vstack([tableau_phase2, obj_row_phase2])
        for i, basic_var_col_idx in enumerate(basic_vars_phase2):
            coeff_in_z = tableau_phase2[-1, basic_var_col_idx]
            if abs(coeff_in_z) > self.TOL:
                tableau_phase2[-1, :] -= coeff_in_z * tableau_phase2[i, :]

        # --- Phase 2 Iterations ---
        it = 1
        phase2_iterations = [(it, copy.deepcopy(tableau_phase2), copy.deepcopy(basic_vars_phase2), None, None)]
        
        while True:
            current_iter_data = phase2_iterations[-1]
            current_tableau = current_iter_data[1]

            obj_coeffs_phase2 = current_tableau[-1, :-1]
            enter_candidates = [j for j, v in enumerate(obj_coeffs_phase2) if v > self.TOL]
            if not enter_candidates:
                break # Optimal for Phase 2

            enter_col = max(enter_candidates, key=lambda j: obj_coeffs_phase2[j])
            
            ratios = [(current_tableau[i, -1] / current_tableau[i, enter_col], i) for i in range(m_phase2) if current_tableau[i, enter_col] > self.TOL]
            if not ratios:
                phase2_iterations[-1] = (current_iter_data[0], current_tableau, current_iter_data[2], enter_col, None)
                return {"status": "No acotado", "phase1_iterations": phase1_iterations, "phase2_iterations": phase2_iterations}

            leave_row = min(ratios, key=lambda x: x[0])[1]

            # Update current iteration with pivot info
            phase2_iterations[-1] = (current_iter_data[0], current_tableau, current_iter_data[2], enter_col, leave_row)

            # Create next iteration's state
            it += 1
            next_tableau = copy.deepcopy(current_tableau)
            next_basic_vars = copy.deepcopy(current_iter_data[2])
            self._pivot(next_tableau, leave_row, enter_col)
            next_basic_vars[leave_row] = enter_col
            phase2_iterations.append((it, next_tableau, next_basic_vars, None, None))

        # --- Results ---
        final_tableau = phase2_iterations[-1][1]
        final_basic_vars = phase2_iterations[-1][2]
        solution = {v: 0.0 for v in var_names_phase2}
        for i, b in enumerate(final_basic_vars):
            solution[var_names_phase2[b]] = final_tableau[i, -1]

        opt_val = final_tableau[-1, -1]
        if self.obj_type == "maximize":
            opt_val *= -1

        return {
            "status": "Optimal",
            "phase1_iterations": phase1_iterations,
            "phase2_iterations": phase2_iterations,
            "variable_names_phase1": var_names_full,
            "variable_names_phase2": var_names_phase2,
            "optimal_solution": {k:v for k,v in solution.items() if k.startswith('x')},
            "optimal_value": opt_val,
        }

    def _solve_simplex_phase2_only(self, tableau_initial, basic_vars_initial, var_names_full):
        tableau = copy.deepcopy(tableau_initial)
        basic_vars = copy.deepcopy(basic_vars_initial)
        m, n = tableau.shape

        obj_row_phase2 = np.zeros(n)
        for i in range(self.n):
            obj_row_phase2[i] = -self.c[i]
        tableau = np.vstack([tableau, obj_row_phase2])
        for i, b_var_col_idx in enumerate(basic_vars):
            coeff_in_z = tableau[-1, b_var_col_idx]
            if abs(coeff_in_z) > self.TOL:
                tableau[-1, :] -= coeff_in_z * tableau[i, :]

        it = 1
        phase2_iterations = [(it, copy.deepcopy(tableau), copy.deepcopy(basic_vars), None, None)]
        
        while True:
            current_iter_data = phase2_iterations[-1]
            current_tableau = current_iter_data[1]

            obj_coeffs = current_tableau[-1, :-1]
            enter_candidates = [j for j, v in enumerate(obj_coeffs) if v > self.TOL]
            if not enter_candidates:
                break # Optimal

            enter_col = max(enter_candidates, key=lambda j: obj_coeffs[j])

            ratios = [(current_tableau[i, -1] / current_tableau[i, enter_col], i) for i in range(m) if current_tableau[i, enter_col] > self.TOL]
            if not ratios:
                phase2_iterations[-1] = (current_iter_data[0], current_tableau, current_iter_data[2], enter_col, None)
                return {"status": "No acotado", "phase1_iterations": [], "phase2_iterations": phase2_iterations}

            leave_row = min(ratios, key=lambda x: x[0])[1]
            
            # Update current iteration with pivot info
            phase2_iterations[-1] = (current_iter_data[0], current_tableau, current_iter_data[2], enter_col, leave_row)

            # Create next iteration's state
            it += 1
            next_tableau = copy.deepcopy(current_tableau)
            next_basic_vars = copy.deepcopy(current_iter_data[2])
            self._pivot(next_tableau, leave_row, enter_col)
            next_basic_vars[leave_row] = enter_col
            phase2_iterations.append((it, next_tableau, next_basic_vars, None, None))
        
        # --- Results ---
        final_tableau = phase2_iterations[-1][1]
        final_basic_vars = phase2_iterations[-1][2]
        solution = {v: 0.0 for v in var_names_full}
        for i, b in enumerate(final_basic_vars):
            solution[var_names_full[b]] = final_tableau[i, -1]
        
        opt_val = final_tableau[-1, -1]
        if self.obj_type == "maximize":
            opt_val *= -1

        return {
            "status": "Optimal",
            "phase1_iterations": [],
            "phase2_iterations": phase2_iterations,
            "variable_names_phase1": [],
            "variable_names_phase2": var_names_full,
            "optimal_solution": {k:v for k,v in solution.items() if k.startswith('x')},
            "optimal_value": opt_val,
        }

# Error handling and validation utilities
class InputValidator:
    """Utility class for input validation"""
    
    @staticmethod
    def validate_number_range(value, min_val, max_val):
        """Validate if a number is within specified range"""
        try:
            num = float(value)
            return min_val <= num <= max_val
        except ValueError:
            return False
            
    @staticmethod
    def validate_coefficient_matrix(coeffs):
        """Validate coefficient matrix for mathematical consistency"""
        # Check if all rows have the same number of columns
        if not coeffs:
            return False
            
        num_cols = len(coeffs[0])
        return all(len(row) == num_cols for row in coeffs)
        
    @staticmethod
    def check_problem_feasibility(constraint_coeffs, constraint_signs, constraint_values):
        """Basic feasibility check for the problem"""
        # This would contain actual feasibility checking logic
        # For now, always returns True
        return True

# Configuration and settings
class AppConfig:
    """Application configuration settings"""
    
    # Visual settings
    WINDOW_MIN_WIDTH = 1000
    WINDOW_MIN_HEIGHT = 700
    DEFAULT_FONT_SIZE = 12
    TITLE_FONT_SIZE = 24
    
    # Color themes
    DARK_THEME = {
        'primary': "#FF5500",
        'background': "#1a1a1a",
        'surface': "#2b2b2b",
        'text': "#ffffff",
        'text_secondary': "#cccccc"
    }
    
    # Constraints
    MIN_VARIABLES = 2
    MAX_VARIABLES = 10
    MIN_CONSTRAINTS = 1
    MAX_CONSTRAINTS = 15
    
    # File paths
    LOGO_PATH = "assets/logo.png"
    ICON_PATH = "assets/icon.ico"

# Main application entry point
if __name__ == "__main__":
    # Initialize the application
    app = LinearProgrammingCalculator()
    
    # Set minimum window size
    app.root.minsize(AppConfig.WINDOW_MIN_WIDTH, AppConfig.WINDOW_MIN_HEIGHT)
    
    # Center the window on screen
    app.root.geometry("1200x800+100+50")
    
    # Set window icon (if available)
    try:
        if os.path.exists(AppConfig.ICON_PATH):
            app.root.iconbitmap(AppConfig.ICON_PATH)
    except:
        pass  # Icon not available, continue without it
    
    # Run the application
    app.run()