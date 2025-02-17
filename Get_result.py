import tkinter as tk
from tkinter import ttk, messagebox
import os
from collections import Counter
import threading
import time
import random
import datetime
import re

# -------------------------------
# Global Variables and Constants
# -------------------------------
BASE_DIR = r"C:\Users\user\PycharmProjects\lottrey"
FILE_COLUMNS = ["DN", "Draw Date", "L1", "L2", "L3", "L4", "L5", "L6"]
MODE_OPTIONS = [
    "6_42.txt", "6_45.txt", "6_49.txt", "6_55.txt", "6_58.txt",
    "EZ2.txt", "Swertres.txt", "4D.txt", "6D.txt"
]

loaded_file_content = ""
parsed_file_content = []
loaded_column_data = []  # Data for the selected lottery column
frequency_data = {}  # Frequency distribution for lottery line values
line_avg_diff = {}  # Average DN difference per lottery line value
most_recent_dn_data = {}  # Most recent DN for each lottery line (excluding zero)
global_max_diff = 1.0  # Maximum historical DN difference (for normalization)

# Physical constants for lottery ball and drawing machine
BALL_NOMINAL_WEIGHT = 2.6  # grams
BALL_WEIGHT_VARIANCE = 0.2  # grams
AIR_PRESSURE_BASE = 40.0  # psi (base value for scaling)

# -------------------------------
# Environmental Parameters Setup
# -------------------------------
# Default (realistic nonzero) values for environmental factors affecting lottery draws.
env_defaults = {
    "Temperature": 22.0,  # °C
    "Temperature Gradient": 0.5,  # °C variation
    "Humidity": 50.0,  # %
    "Humidity Gradient": 2.0,  # % variation
    "Air Pressure": 101.3,  # kPa
    "Wind Speed": 0.5,  # m/s
    "Airflow Direction": 0.0,  # degrees
    "Air Density": 1.2,  # kg/m³
    "Air Turbulence": 1.0,  # scale factor (>0)
    "Vibration": 0.2,  # arbitrary units
    "Acoustic Pressure Fluctuations": 0.1,  # arbitrary units
    "Micro-Seismic Activity": 0.05,  # arbitrary units
    "Altitude": 100.0,  # meters
    "Barometric Pressure Trend": 0.1,  # kPa/min
    "Dust Level": 0.1,  # mg/m³
    "Air Quality": 50.0,  # AQI
    "Solar Radiation": 500.0,  # W/m²
    "Static Electricity": 0.05,  # kV
    "Air Ion Concentration": 300.0,  # ions/cm³
    "Ambient Electrical Field": 100.0,  # V/m
    "Electromagnetic Interference": 0.1,  # arbitrary units
    "Lightning Activity": 0.1,  # arbitrary units
    "Cosmic Ray Activity": 1.0,  # arbitrary units
    "Local Gravitational Variations": 9.8,  # m/s²
    "Air Viscosity Variations": 1.8e-5,  # Pa·s
    "Precipitation/Condensation": 0.1,  # mm/hr equivalent
    "Coriolis Effect": 0.1  # arbitrary units
}

# Definitions for each environmental parameter.
env_definitions = {
    "Temperature": "Ambient operating temperature during the draw.",
    "Temperature Gradient": "Spatial variations in temperature within the drawing chamber that may create localized convection currents.",
    "Humidity": "Moisture level in the air, which can influence static charge buildup and ball behavior.",
    "Humidity Gradient": "Spatial variations in humidity within the drawing chamber that may cause localized changes in air density and static buildup.",
    "Air Pressure": "Ambient atmospheric pressure that can subtly affect ball trajectory.",
    "Wind Speed": "Any airflow or drafts inside the drawing chamber that may alter ball movement.",
    "Airflow Direction": "The directional flow of air, influencing the ball's initial trajectory.",
    "Air Density": "Variations in density impacting the aerodynamic properties of the ball.",
    "Air Turbulence": "Localized turbulent airflow patterns within the drawing environment. Adjusted values reflect the high‐pressure spin in the drawing machine.",
    "Vibration": "External vibrations or tremors (e.g., building movements) that might impact the draw.",
    "Acoustic Pressure Fluctuations": "Sound pressure variations that may induce minor air vibrations affecting the ball's movement.",
    "Micro-Seismic Activity": "Minor ground movements beyond standard vibrations that might subtly influence the drawing mechanism.",
    "Altitude": "The elevation of the drawing location, which influences air pressure and density.",
    "Barometric Pressure Trend": "The rate of change in atmospheric pressure during the draw.",
    "Dust Level": "The concentration of airborne particles that may affect friction on the ball’s surface.",
    "Air Quality": "The presence of pollutants or particulate matter in the air, potentially altering ball surface interactions.",
    "Solar Radiation": "Intensity of sunlight, which can indirectly modify ambient temperature and static conditions.",
    "Static Electricity": "Accumulation of static charge on the balls influenced by environmental conditions.",
    "Air Ion Concentration": "The concentration of ions in the air, affecting static charge accumulation on the balls.",
    "Ambient Electrical Field": "The natural electrical field present in the atmosphere that might influence the ball's charge distribution and behavior.",
    "Electromagnetic Interference": "Natural electromagnetic fields (e.g., from solar activity) that could impact sensitive components.",
    "Lightning Activity": "Nearby lightning or thunderstorm activity that may cause sudden electromagnetic disturbances.",
    "Cosmic Ray Activity": "High-energy particles from space that may cause minor ionizations in the air, subtly affecting static conditions.",
    "Local Gravitational Variations": "Slight differences in gravitational pull due to local geological or topographical conditions.",
    "Air Viscosity Variations": "Subtle changes in air viscosity, influenced by temperature and humidity, which can affect ball movement.",
    "Precipitation/Condensation": "The presence of water droplets or condensation, particularly in non-controlled environments, affecting friction and ball behavior.",
    "Coriolis Effect": "The deflection of ball movement due to Earth's rotation; typically negligible at small scales."
}

# This will hold the Entry widgets for environmental parameters.
env_entries = {}
env_param_list = [
    "Temperature", "Temperature Gradient", "Humidity", "Humidity Gradient",
    "Air Pressure", "Wind Speed", "Airflow Direction", "Air Density",
    "Air Turbulence", "Vibration", "Acoustic Pressure Fluctuations", "Micro-Seismic Activity",
    "Altitude", "Barometric Pressure Trend", "Dust Level", "Air Quality", "Solar Radiation",
    "Static Electricity", "Air Ion Concentration", "Ambient Electrical Field",
    "Electromagnetic Interference", "Lightning Activity", "Cosmic Ray Activity",
    "Local Gravitational Variations", "Air Viscosity Variations", "Precipitation/Condensation",
    "Coriolis Effect"
]


# -------------------------------
# Data Normalization Function
# -------------------------------
def normalize_data(data):
    numeric_fields = ["DN", "L1", "L2", "L3", "L4", "L5", "L6"]
    stats = {}
    for field in numeric_fields:
        try:
            values = [float(record[field]) for record in data if record[field] != "0"]
            stats[field] = (min(values), max(values)) if values else (0, 1)
        except Exception:
            stats[field] = (0, 1)
    for record in data:
        for field in numeric_fields:
            try:
                val = float(record[field])
                min_val, max_val = stats[field]
                record[field + "_norm"] = (val - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0.0
            except Exception:
                record[field + "_norm"] = 0.0
    return data


# -------------------------------
# Helper Functions for File Handling, Splitting, and Logging
# -------------------------------
def get_file_filepath(filename):
    return os.path.join(BASE_DIR, filename)


def append_log(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    console_text.insert(tk.END, timestamp + message + "\n")
    console_text.see(tk.END)


def load_file_content(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        append_log(f"Error reading file {file_path}: {e}")
        return ""


def parse_file_content(content):
    data = []
    lines = content.strip().splitlines()
    for line in lines:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) == len(FILE_COLUMNS):
            record = dict(zip(FILE_COLUMNS, fields))
            data.append(record)
        else:
            append_log(f"Parsing Warning: Line skipped due to unexpected number of fields:\n{line}")
    return data


def compute_data_splits(data):
    splits = []
    ratios = [(0.6, 0.3, 0.1), (0.7, 0.2, 0.1), (0.8, 0.1, 0.1), (0.9, 0.05, 0.05)]
    total = len(data)
    for ratio in ratios:
        train = int(total * ratio[0])
        validation = int(total * ratio[1])
        test = total - train - validation
        splits.append((train, validation, test))
    avg_train = sum(x[0] for x in splits) / len(splits)
    avg_validation = sum(x[1] for x in splits) / len(splits)
    avg_test = sum(x[2] for x in splits) / len(splits)
    append_log(f"Data Splits (avg): Train={avg_train:.1f}, Validation={avg_validation:.1f}, Test={avg_test:.1f}")
    return splits


# -------------------------------
# Machine-Specific Data Integration and Popups
# -------------------------------
machine_param_definitions = {
    "Machine ID": "A unique identifier for each lottery drawing machine.",
    "Calibration Data": "Information about the machine's calibration (e.g., last calibration date).",
    "Wear Status": "An indicator of the machine's physical condition (e.g., percentage wear).",
    "Usage Stats": "Data showing how much the machine has been used (e.g., number of draws).",
    "Temperature": "Operating temperature of the machine.",
    "Humidity": "Internal humidity level in the machine."
}


def show_machine_param_info(param):
    definition = machine_param_definitions.get(param, "No definition available.")
    messagebox.showinfo(param, definition)


# -------------------------------
# Environmental Parameter Definition Popup
# -------------------------------
def show_env_definition(param):
    definition = env_definitions.get(param, "No definition available.")
    messagebox.showinfo(param, definition)


# -------------------------------
# New: Environmental Adjustment Algorithm
# -------------------------------
def compute_env_adjustment():
    """
    Computes a weighted adjustment factor based on environmental parameters.
    For each parameter, the ratio (actual/default) is weighted and then averaged.
    A factor of ~1.0 indicates default conditions.
    """
    weights = {
        "Temperature": 0.05,
        "Temperature Gradient": 0.05,
        "Humidity": 0.05,
        "Humidity Gradient": 0.05,
        "Air Pressure": 0.1,
        "Wind Speed": 0.05,
        "Air Density": 0.05,
        "Air Turbulence": 0.15,
        "Vibration": 0.05,
        "Acoustic Pressure Fluctuations": 0.05,
        "Micro-Seismic Activity": 0.05,
        "Altitude": 0.05,
        "Barometric Pressure Trend": 0.05,
        "Dust Level": 0.05,
        "Air Quality": 0.05,
        "Solar Radiation": 0.05,
        "Static Electricity": 0.05,
        "Air Ion Concentration": 0.05,
        "Ambient Electrical Field": 0.05,
        "Electromagnetic Interference": 0.05,
        "Lightning Activity": 0.05,
        "Cosmic Ray Activity": 0.05,
        "Local Gravitational Variations": 0.05,
        "Air Viscosity Variations": 0.05,
        "Precipitation/Condensation": 0.05,
        "Coriolis Effect": 0.05
    }
    total_weight = sum(weights.values())
    sum_weighted_ratios = 0.0
    for param, weight in weights.items():
        if param in env_entries:
            try:
                actual = float(env_entries[param].get())
            except Exception:
                actual = env_defaults[param]
            default = env_defaults[param]
            ratio = actual / default if default != 0 else 1.0
            sum_weighted_ratios += weight * ratio
    return sum_weighted_ratios / total_weight


# -------------------------------
# GUI Event Functions
# -------------------------------
def on_mode_select(event=None):
    selected_mode = mode_combo.get()
    file_path = get_file_filepath(selected_mode)
    if not os.path.exists(file_path):
        try:
            with open(file_path, 'w') as f:
                pass
            append_log(f"File {file_path} created as it did not exist.")
        except Exception as e:
            append_log(f"Error creating file {file_path}: {e}")
            file_path_label.config(text=f"File path: {file_path} (Error creating file)")
            return
    file_path_label.config(text=f"File path: {file_path}")
    global loaded_file_content, parsed_file_content, loaded_column_data, frequency_data, line_avg_diff, most_recent_dn_data
    loaded_file_content = load_file_content(file_path)
    parsed_file_content = parse_file_content(loaded_file_content)
    parsed_file_content = normalize_data(parsed_file_content)
    loaded_column_data = []
    frequency_data = {}
    line_avg_diff = {}
    most_recent_dn_data = {}
    clear_prediction_tree()
    lottery_column_combo.set("L1")
    on_lottery_column_select()


def on_lottery_column_select(event=None):
    selected_column = lottery_column_combo.get()
    global loaded_column_data, frequency_data, line_avg_diff, most_recent_dn_data
    if parsed_file_content:
        loaded_column_data = [record[selected_column] for record in parsed_file_content if
                              record.get(selected_column, "0") != "0"]
        frequency_data = dict(Counter(loaded_column_data))
        append_log(f"Frequency data for {selected_column} recalculated: {frequency_data}")
        diff_map = {}
        if len(parsed_file_content) > 1:
            for i in range(1, len(parsed_file_content)):
                try:
                    prev_dn = float(parsed_file_content[i - 1]["DN"])
                    curr_dn = float(parsed_file_content[i]["DN"])
                    diff = curr_dn - prev_dn
                    if diff > 0:
                        line_val = parsed_file_content[i][selected_column]
                        if line_val != "0":
                            diff_map.setdefault(line_val, []).append(diff)
                except Exception:
                    pass
        line_avg_diff.clear()
        for line_val, diffs in diff_map.items():
            if diffs:
                line_avg_diff[line_val] = sum(diffs) / len(diffs)
        if line_avg_diff:
            append_log("Average DN differences per lottery line computed.")
        else:
            append_log("No DN difference data available; defaulting to 1.")
        most_recent_dn_data.clear()
        for record in reversed(parsed_file_content):
            val = record.get(selected_column, "0")
            if val != "0" and val not in most_recent_dn_data:
                try:
                    dn_val = float(record["DN"])
                    most_recent_dn_data[val] = dn_val
                except Exception:
                    pass
        append_log(f"Most recent DN data for {selected_column}: {most_recent_dn_data}")
    else:
        append_log("Warning: No parsed file content available. Please load a file first.")


def clear_prediction_tree():
    for item in prediction_tree.get_children():
        prediction_tree.delete(item)


# -------------------------------
# Prediction Algorithm and Multi-threading
# -------------------------------
def start_prediction():
    if not parsed_file_content or not frequency_data:
        append_log("Error: No data available for prediction. Please load a valid file and select a lottery column.")
        return
    start_button.config(state="disabled")
    clear_prediction_tree()
    # Run prediction in a separate thread so that the GUI remains responsive.
    thread = threading.Thread(target=prediction_algorithm)
    thread.daemon = True
    thread.start()


def prediction_algorithm():
    append_log("Starting prediction algorithm...")
    machine_params = {
        "Machine ID": machine_id_combo.get(),
        "Calibration Data": machine_calibration_combo.get(),
        "Wear Status": machine_wear_combo.get(),
        "Usage Stats": machine_usage_combo.get(),
        "Temperature": machine_temp_combo.get(),
        "Humidity": machine_humidity_combo.get()
    }
    append_log("Machine parameters: " + str(machine_params))

    compute_data_splits(parsed_file_content)
    total_steps = 10
    simulated_total_time = 5  # seconds
    start_time = time.time()
    for step in range(total_steps + 1):
        progress = int((step / total_steps) * 100)
        elapsed = time.time() - start_time
        remaining = max(simulated_total_time - elapsed, 0)
        eta_text = f"ETA: {remaining:.1f}s"
        root.after(0, update_progress, progress, eta_text)
        time.sleep(simulated_total_time / total_steps)

    try:
        last_dn = float(parsed_file_content[-1]["DN"]) if parsed_file_content else 0.0
    except Exception as e:
        append_log("Error: Could not determine last DN value.")
        root.after(0, lambda: start_button.config(state="normal"))
        return

    overall_diffs = []
    if len(parsed_file_content) > 1:
        for i in range(1, len(parsed_file_content)):
            try:
                diff = float(parsed_file_content[i]["DN"]) - float(parsed_file_content[i - 1]["DN"])
                if diff > 0:
                    overall_diffs.append(diff)
            except Exception:
                pass
    overall_avg_diff = sum(overall_diffs) / len(overall_diffs) if overall_diffs else 1.0

    hist_diffs = []
    if len(parsed_file_content) > 1:
        for i in range(1, len(parsed_file_content)):
            try:
                diff = float(parsed_file_content[i]["DN"]) - float(parsed_file_content[i - 1]["DN"])
                if diff > 0:
                    hist_diffs.append(diff)
            except Exception:
                pass
    max_diff = max(hist_diffs) if hist_diffs else 1.0
    global global_max_diff
    global_max_diff = max_diff

    machine_factor = get_machine_factor()  # based on machine wear

    # Compute a more advanced environmental adjustment factor.
    env_factor = compute_env_adjustment()
    append_log(f"Computed Environmental Adjustment Factor: {env_factor:.3f}")

    # Generate 10 predictions using weighted random selection and adjustments.
    alpha = 0.5  # weighting factor between overall and line-specific averages
    predictions = []
    numbers = list(frequency_data.keys())
    counts = list(frequency_data.values())
    if not numbers:
        append_log("Error: No frequency data available for prediction.")
        root.after(0, lambda: start_button.config(state="normal"))
        return
    max_count = max(counts) if counts else 1

    for i in range(10):
        predicted_line = random.choices(numbers, weights=counts, k=1)[0]
        line_avg = line_avg_diff.get(predicted_line, overall_avg_diff)
        noise = random.uniform(-0.5, 0.5)
        frequency = frequency_data.get(predicted_line, 1)
        # Adjust effective weight with machine and environmental factors.
        effective_weight = BALL_NOMINAL_WEIGHT * (1 - (frequency / max_count * 0.1))
        effective_weight *= machine_factor
        effective_weight *= env_factor
        weight_adjustment = BALL_NOMINAL_WEIGHT - effective_weight

        predicted_increment = alpha * overall_avg_diff + (1 - alpha) * line_avg + noise - weight_adjustment
        predicted_increment *= (AIR_PRESSURE_BASE / 40.0)
        predicted_increment *= env_factor  # further scaling with environmental adjustment

        predicted_dn = last_dn + predicted_increment
        if predicted_dn < last_dn + overall_avg_diff:
            predicted_dn = last_dn + overall_avg_diff

        prob_line = min(frequency / (last_dn + predicted_increment + 1) + random.uniform(0, 0.1), 1.0)
        predictions.append((predicted_dn, predicted_line, prob_line))

    predictions.sort(key=lambda x: x[2], reverse=True)
    root.after(0, update_prediction_tree, predictions)
    append_log("Prediction algorithm completed.")
    root.after(0, lambda: start_button.config(state="normal"))
    root.after(0, update_progress, 0, "ETA: 0.0s")


def update_progress(progress, eta_text):
    progress_bar["value"] = progress
    progress_label.config(text=f"{progress}%")
    eta_label.config(text=eta_text)


def update_prediction_tree(predictions):
    clear_prediction_tree()
    for pred in predictions:
        dn, line, prob = pred
        dn_formatted = f"{dn:.2f}"
        try:
            line_int = int(line)
            line_formatted = f"{line_int:02d}"
        except Exception:
            line_formatted = line
        prediction_tree.insert("", "end", values=(dn_formatted, line_formatted, f"{prob:.4f}"))


def copy_selected():
    selected_items = prediction_tree.selection()
    if not selected_items:
        return
    copied_text = ""
    for item in selected_items:
        values = prediction_tree.item(item, "values")
        copied_text += "\t".join(values) + "\n"
    root.clipboard_clear()
    root.clipboard_append(copied_text)
    append_log("Selected items copied to clipboard.")


def get_machine_factor():
    try:
        wear_str = machine_wear_combo.get()
        match = re.search(r'\((\d+)%\)', wear_str)
        if match:
            wear = float(match.group(1))
        else:
            wear = 0.0
        factor = (100 - wear) / 100.0
        return factor
    except Exception:
        return 1.0


# -------------------------------
# GUI Layout and Initialization
# -------------------------------
root = tk.Tk()
root.title("Lottery Prediction Application")
# Set window size to 1400x900 and center on screen.
width, height = 1400, 900
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (width // 2)
y = (screen_height // 2) - (height // 2)
root.geometry(f"{width}x{height}+{x}+{y}")

# -------------------------------
# Main PanedWindow (Three Columns)
# -------------------------------
main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# --- Left Panel: Controls ---
left_frame = ttk.Frame(main_paned, width=250)
main_paned.add(left_frame, weight=1)

mode_label = ttk.Label(left_frame, text="Select Lottery Mode:")
mode_label.pack(pady=5)
mode_combo = ttk.Combobox(left_frame, values=MODE_OPTIONS, state="readonly")
mode_combo.pack(pady=5)
mode_combo.bind("<<ComboboxSelected>>", on_mode_select)
default_mode = "6_42.txt"
mode_combo.set(default_mode)

file_path_label = ttk.Label(left_frame, text="File path will appear here.")
file_path_label.pack(pady=5)

lottery_column_label = ttk.Label(left_frame, text="Select Lottery Column:")
lottery_column_label.pack(pady=5)
lottery_columns = ["L1", "L2", "L3", "L4", "L5", "L6"]
lottery_column_combo = ttk.Combobox(left_frame, values=lottery_columns, state="readonly")
lottery_column_combo.pack(pady=5)
lottery_column_combo.bind("<<ComboboxSelected>>", on_lottery_column_select)
lottery_column_combo.set("L1")

# Machine-specific parameters panel (inside left panel)
machine_frame = ttk.LabelFrame(left_frame, text="Machine Parameters")
machine_frame.pack(pady=10, fill=tk.X, padx=5)
ttk.Label(machine_frame, text="Machine ID:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
machine_id_combo = ttk.Combobox(machine_frame, values=["Machine 1", "Machine 2", "Machine 3"], state="readonly")
machine_id_combo.grid(row=0, column=1, padx=2, pady=2)
machine_id_combo.set("Machine 1")
ttk.Button(machine_frame, text="?", width=2, command=lambda: show_machine_param_info("Machine ID")).grid(row=0,
                                                                                                         column=2,
                                                                                                         padx=2, pady=2)
ttk.Label(machine_frame, text="Calibration Data:").grid(row=1, column=0, sticky="e", padx=2, pady=2)
machine_calibration_combo = ttk.Combobox(machine_frame,
                                         values=["Calibrated (2025-01-01)", "Not Calibrated", "Calibration Due"],
                                         state="readonly")
machine_calibration_combo.grid(row=1, column=1, padx=2, pady=2)
machine_calibration_combo.set("Calibrated (2025-01-01)")
ttk.Button(machine_frame, text="?", width=2, command=lambda: show_machine_param_info("Calibration Data")).grid(row=1,
                                                                                                               column=2,
                                                                                                               padx=2,
                                                                                                               pady=2)
ttk.Label(machine_frame, text="Wear Status:").grid(row=2, column=0, sticky="e", padx=2, pady=2)
machine_wear_combo = ttk.Combobox(machine_frame,
                                  values=["New (0%)", "Light wear (10%)", "Moderate wear (20%)", "Heavy wear (30%)"],
                                  state="readonly")
machine_wear_combo.grid(row=2, column=1, padx=2, pady=2)
machine_wear_combo.set("New (0%)")
ttk.Button(machine_frame, text="?", width=2, command=lambda: show_machine_param_info("Wear Status")).grid(row=2,
                                                                                                          column=2,
                                                                                                          padx=2,
                                                                                                          pady=2)
ttk.Label(machine_frame, text="Usage Stats:").grid(row=3, column=0, sticky="e", padx=2, pady=2)
machine_usage_combo = ttk.Combobox(machine_frame, values=["Low Usage", "Moderate Usage", "High Usage"],
                                   state="readonly")
machine_usage_combo.grid(row=3, column=1, padx=2, pady=2)
machine_usage_combo.set("Low Usage")
ttk.Button(machine_frame, text="?", width=2, command=lambda: show_machine_param_info("Usage Stats")).grid(row=3,
                                                                                                          column=2,
                                                                                                          padx=2,
                                                                                                          pady=2)
ttk.Label(machine_frame, text="Temperature:").grid(row=4, column=0, sticky="e", padx=2, pady=2)
machine_temp_combo = ttk.Combobox(machine_frame, values=["20°C", "22°C", "25°C", "27°C", "30°C"], state="readonly")
machine_temp_combo.grid(row=4, column=1, padx=2, pady=2)
machine_temp_combo.set("22°C")
ttk.Button(machine_frame, text="?", width=2, command=lambda: show_machine_param_info("Temperature")).grid(row=4,
                                                                                                          column=2,
                                                                                                          padx=2,
                                                                                                          pady=2)
ttk.Label(machine_frame, text="Humidity:").grid(row=5, column=0, sticky="e", padx=2, pady=2)
machine_humidity_combo = ttk.Combobox(machine_frame, values=["30%", "40%", "50%", "60%", "70%"], state="readonly")
machine_humidity_combo.grid(row=5, column=1, padx=2, pady=2)
machine_humidity_combo.set("50%")
ttk.Button(machine_frame, text="?", width=2, command=lambda: show_machine_param_info("Humidity")).grid(row=5, column=2,
                                                                                                       padx=2, pady=2)

start_button = ttk.Button(left_frame, text="PREDICT", command=start_prediction)
start_button.pack(pady=10)

progress_frame = ttk.Frame(left_frame)
progress_frame.pack(pady=10, fill=tk.X)
progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode="determinate")
progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
progress_label = ttk.Label(progress_frame, text="0%")
progress_label.pack(side=tk.LEFT, padx=5)
eta_label = ttk.Label(progress_frame, text="ETA: 0.0s")
eta_label.pack(side=tk.LEFT, padx=5)

# --- Middle Panel: Prediction Output ---
middle_frame = ttk.Frame(main_paned)
main_paned.add(middle_frame, weight=3)
prediction_label = ttk.Label(middle_frame, text="Prediction Results:")
prediction_label.pack(pady=5)
columns = ("Predicted DN", "Predicted Line", "Probability")
prediction_tree = ttk.Treeview(middle_frame, columns=columns, show="headings", height=15, selectmode="extended")
for col in columns:
    prediction_tree.heading(col, text=col)
    prediction_tree.column(col, anchor="center", width=120)
prediction_tree.pack(pady=5, fill=tk.BOTH, expand=True)
copy_button = ttk.Button(middle_frame, text="Copy Selected", command=copy_selected)
copy_button.pack(pady=5)

# --- Right Panel: Environmental Parameters and Console Log ---
right_panel = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
main_paned.add(right_panel, weight=2)

env_frame_container = ttk.Frame(right_panel)
env_labelframe = ttk.LabelFrame(env_frame_container, text="Environmental Parameters")
env_labelframe.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

env_canvas = tk.Canvas(env_labelframe)
env_scrollbar = ttk.Scrollbar(env_labelframe, orient="vertical", command=env_canvas.yview)
env_inner = ttk.Frame(env_canvas)
env_inner.bind(
    "<Configure>",
    lambda e: env_canvas.configure(scrollregion=env_canvas.bbox("all"))
)
env_canvas.create_window((0, 0), window=env_inner, anchor="nw")
env_canvas.configure(yscrollcommand=env_scrollbar.set)
env_canvas.pack(side="left", fill="both", expand=True)
env_scrollbar.pack(side="right", fill="y")

for i, param in enumerate(env_param_list):
    ttk.Label(env_inner, text=param + ":").grid(row=i, column=0, sticky="w", padx=2, pady=2)
    entry = ttk.Entry(env_inner, width=10)
    entry.grid(row=i, column=1, padx=2, pady=2)
    entry.insert(0, str(env_defaults.get(param, 1)))
    env_entries[param] = entry
    ttk.Button(env_inner, text="?", width=2, command=lambda p=param: show_env_definition(p)).grid(row=i, column=2,
                                                                                                  padx=2, pady=2)

right_panel.add(env_frame_container, weight=2)

console_frame = ttk.Frame(right_panel)
console_text = tk.Text(console_frame, height=10)
console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
console_scrollbar = ttk.Scrollbar(console_frame, command=console_text.yview)
console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
console_text.configure(yscrollcommand=console_scrollbar.set)
right_panel.add(console_frame, weight=1)

append_log("Application started. Loading default mode file...")
on_mode_select()

root.mainloop()
