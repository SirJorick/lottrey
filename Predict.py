import tkinter as tk
from tkinter import ttk, messagebox
import os
import csv
import re
from datetime import datetime, timedelta
import numpy as np

# Scikit-learn imports (install via: pip install scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ------------------ Constants ------------------

BASE_DIR = r"C:\Users\user\PycharmProjects\lottrey"
FILE_COLUMNS = ["DN", "Draw Date", "L1", "L2", "L3", "L4", "L5", "L6"]
FILE_DATE_FORMAT = "%d-%b-%y"  # e.g., "01-Feb-25"
DISPLAY_DATE_FORMAT = "%b %d, %Y"  # e.g., "Feb 01, 2025"
MODE_OPTIONS = [
    "6_42.txt", "6_45.txt", "6_49.txt", "6_55.txt", "6_58.txt",
    "EZ2.txt", "Swertres.txt", "4D.txt", "6D.txt"
]
NUM_PREDICTIONS = 10  # Predict only 10 future draws


# ------------------ Utility Functions ------------------

def get_file_filepath(filename):
    return os.path.join(BASE_DIR, filename)


def get_max_number(filename):
    # For a file like "6_42.txt", returns 42.
    m = re.search(r"_(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


# ------------------ Data Loading & Historical Frequency ------------------

def load_file_data(filepath):
    """
    Load data from the given file.
    Returns a list of dictionaries with keys: DN, Draw Date, and L1.
    """
    data = []
    try:
        with open(filepath, 'r', newline='') as f:
            sample = f.read(1024)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                try:
                    draw_date = datetime.strptime(row['Draw Date'].strip(), FILE_DATE_FORMAT)
                    DN = int(row['DN'].strip())
                    L1 = int(row['L1'].strip())
                    data.append({
                        'DN': DN,
                        'Draw Date': draw_date,
                        'L1': L1
                    })
                except Exception:
                    continue
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file data: {e}")
    return data


def calculate_historical_frequency(data, max_number):
    """
    For each L1 value from 1 to max_number, count its occurrences.
    Total draws is assumed to be the maximum DN (if draws are sequential).
    Returns (freq_dict, overall_avg, total_draws).
    """
    if not data:
        return None, 0, 0
    total_draws = max(row['DN'] for row in data)
    freq_dict = {}
    for l1 in range(1, max_number + 1):
        count = sum(1 for row in data if row['L1'] == l1)
        freq = (count / total_draws * 100) if total_draws > 0 else 0
        freq_dict[l1] = (count, freq)
    overall_avg = sum(row['L1'] for row in data) / len(data)
    return freq_dict, overall_avg, total_draws


# ------------------ Model Training & Future Predictions ------------------

def train_models_on_file(data, train_percent, val_percent, test_percent):
    """
    Split the loaded file data into training, validation, and test sets.
    The splits are determined by:
      - Training: train_percent%
      - Validation: val_percent%
      - Test: test_percent%
    (The three values must sum to 100.)

    First, split the data into training (train_percent%) and remainder.
    Then, split the remainder into validation and test sets according to the given percentages.

    Train:
      - model_date: a LinearRegression model to predict Draw Date (as ordinal) from DN.
      - model_L1: a RandomForestRegressor to predict L1 from [DN, date_ordinal].
    Prints validation and test performance, then returns (model_date, model_L1).
    """
    if len(data) < 10:
        messagebox.showwarning("Not Enough Data", "Not enough data in the file to train models.")
        return None, None

    X = np.array([[row['DN'], row['Draw Date'].toordinal()] for row in data])
    y_L1 = np.array([row['L1'] for row in data])
    X_date = X[:, 0].reshape(-1, 1)
    y_date = X[:, 1]

    # Compute the remaining percentage.
    remaining = 100 - train_percent
    # Calculate the ratio of validation to the remainder.
    # For example, if train=70, remaining=30, if val_percent is chosen as 20 then test_percent must be 10.
    # We'll assume the combobox values are already set correctly (and sum to 100).
    # First split: training vs remainder.
    train_ratio = train_percent / 100.0
    X_train, X_rem, y_train, y_rem = train_test_split(X, y_L1, train_size=train_ratio, random_state=42)

    # The remainder proportion (we have val_percent and test_percent)
    rem_total = 100 - train_percent
    val_ratio = val_percent / rem_total if rem_total > 0 else 0

    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=val_ratio, random_state=42)

    X_date_train = X_train[:, 0].reshape(-1, 1)
    y_date_train = X_train[:, 1]

    model_date = LinearRegression().fit(X_date_train, y_date_train)
    model_L1 = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    # Evaluate performance (optional).
    X_date_val = X_val[:, 0].reshape(-1, 1)
    val_date_pred = model_date.predict(X_date_val)
    val_mae_date = mean_absolute_error(X_val[:, 1], val_date_pred)
    val_L1_pred = model_L1.predict(X_val)
    val_mae_L1 = mean_absolute_error(y_val, val_L1_pred)
    print("Validation MAE (Date):", val_mae_date)
    print("Validation MAE (L1):", val_mae_L1)

    X_date_test = X_test[:, 0].reshape(-1, 1)
    test_date_pred = model_date.predict(X_date_test)
    test_mae_date = mean_absolute_error(X_test[:, 1], test_date_pred)
    test_L1_pred = model_L1.predict(X_test)
    test_mae_L1 = mean_absolute_error(y_test, test_L1_pred)
    print("Test MAE (Date):", test_mae_date)
    print("Test MAE (L1):", test_mae_L1)

    return model_date, model_L1


global_predictions = []


def do_future_predictions(i, num_predictions, model_date, model_L1, data, max_number):
    """
    Compute prediction i using the trained models.
    new_dn = last DN + i.
    A noise term is added to the predicted L1 based on 0.5 * historical std.
    Append (i, new_dn, predicted_date, predicted_L1) to global_predictions.
    """
    global global_predictions
    data_sorted = sorted(data, key=lambda row: row['DN'])
    last_dn = data_sorted[-1]['DN']
    new_dn = last_dn + i

    pred_date_ord = model_date.predict(np.array([[new_dn]]))[0]
    try:
        pred_date = datetime.fromordinal(int(round(pred_date_ord)))
    except Exception:
        pred_date = data_sorted[-1]['Draw Date'] + timedelta(days=i)

    pred_L1 = model_L1.predict(np.array([[new_dn, pred_date_ord]]))[0]
    hist_L1 = np.array([row['L1'] for row in data])
    std_L1 = np.std(hist_L1)
    noise = np.random.normal(0, std_L1 * 0.5)
    pred_L1 += noise
    pred_L1 = int(round(pred_L1))
    if max_number is not None:
        pred_L1 = max(1, min(pred_L1, max_number))
    global_predictions.append((i, new_dn, pred_date.strftime(DISPLAY_DATE_FORMAT), pred_L1))

    progress = int((i / num_predictions) * 100)
    progress_bar['value'] = progress
    progress_label.config(text=f"Progress: {progress}%")
    eta = (num_predictions - i) * 0.5
    eta_label.config(text=f"ETA: {eta:.1f} sec")
    if i < num_predictions:
        root.after(100, do_future_predictions, i + 1, num_predictions, model_date, model_L1, data, max_number)
    else:
        update_predictions_tree(global_predictions)
        progress_label.config(text="Completed!")
        eta_label.config(text="")
        predict_button.config(state="normal")


def update_predictions_tree(predictions):
    for item in prediction_tree.get_children():
        prediction_tree.delete(item)
    for pred in predictions:
        prediction_tree.insert("", "end", values=pred)


def on_predict():
    """
    When the user clicks the Predict Future Draws button:
      - Load the selected file's data.
      - Split the data using the user-defined splits (Training, Validation, Test).
      - Train models.
      - Predict NUM_PREDICTIONS future draws.
    """
    global global_predictions
    selected_file = mode_combo.get()
    filepath = get_file_filepath(selected_file)
    max_num = get_max_number(selected_file)
    file_path_label.config(text=f"File path: {filepath}")

    data = load_file_data(filepath)
    if not data:
        messagebox.showerror("Error", "No valid data loaded from file.")
        return

    try:
        train_percent = float(train_combo.get())
        val_percent = float(val_combo.get())
        test_percent = float(test_combo.get())
    except:
        messagebox.showerror("Error", "Please select valid split percentages.")
        return

    if abs(train_percent + val_percent + test_percent - 100) > 0.001:
        messagebox.showerror("Error", "Training, Validation, and Test percentages must sum to 100.")
        return

    split_label.config(text=f"Splits: Training: {train_percent}% | Validation: {val_percent}% | Test: {test_percent}%")

    model_date, model_L1 = train_models_on_file(data, train_percent, val_percent, test_percent)
    if model_date is None or model_L1 is None:
        messagebox.showerror("Training Error", "Failed to train models on file data.")
        return

    global_predictions = []
    progress_bar['value'] = 0
    progress_label.config(text="Progress: 0%")
    eta_label.config(text="ETA: calculating...")
    predict_button.config(state="disabled")
    do_future_predictions(1, NUM_PREDICTIONS, model_date, model_L1, data, max_num)
    root.after(NUM_PREDICTIONS * 100 + 1000, lambda: predict_button.config(state="normal"))


def copy_predictions():
    rows = []
    headers = [prediction_tree.heading(col)["text"] for col in prediction_tree["columns"]]
    rows.append("\t".join(headers))
    for item in prediction_tree.get_children():
        values = prediction_tree.item(item, "values")
        rows.append("\t".join(str(v) for v in values))
    text = "\n".join(rows)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copied", "Predictions copied to clipboard.")


def copy_hist_summary():
    rows = []
    headers = [hist_summary_tree.heading(col)["text"] for col in hist_summary_tree["columns"]]
    rows.append("\t".join(headers))
    for item in hist_summary_tree.get_children():
        values = hist_summary_tree.item(item, "values")
        rows.append("\t".join(str(v) for v in values))
    text = "\n".join(rows)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copied", "Historical summary copied to clipboard.")


def on_calculate_historical():
    selected_file = mode_combo.get()
    filepath = get_file_filepath(selected_file)
    file_path_label.config(text=f"File path: {filepath}")
    data = load_file_data(filepath)
    if not data:
        messagebox.showerror("Error", "No valid data loaded from file.")
        return
    max_num = get_max_number(selected_file)
    freq_dict, overall_avg, total_draws = calculate_historical_frequency(data, max_num)
    for item in hist_summary_tree.get_children():
        hist_summary_tree.delete(item)
    for l1 in range(1, max_num + 1):
        count, freq = freq_dict.get(l1, (0, 0))
        if count > 0:
            hist_summary_tree.insert("", "end", values=(l1, count, f"{freq:.2f}%"))
    overall_hist_label.config(text=f"Overall Historical L1 Average: {overall_avg:.2f} | Total Draws: {total_draws}")


# ------------------ GUI Setup ------------------

root = tk.Tk()
root.title("Lottery Future Draws & Historical Frequency")

frame = ttk.Frame(root, padding=20)
frame.pack(fill=tk.BOTH, expand=True)

# File selection
mode_label = ttk.Label(frame, text="Select Lottery File:")
mode_label.pack(pady=5)
mode_combo = ttk.Combobox(frame, values=MODE_OPTIONS, state="readonly")
mode_combo.pack(pady=5)
mode_combo.set("6_42.txt")
file_path_label = ttk.Label(frame, text="File path will appear here.")
file_path_label.pack(pady=5)

# Training Splits Adjustment
split_frame = ttk.LabelFrame(frame, text="Adjust Data Splits (%)")
split_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

train_label = ttk.Label(split_frame, text="Training:")
train_label.grid(row=0, column=0, padx=5, pady=5)
train_values = [str(x) for x in range(50, 91)]
train_combo = ttk.Combobox(split_frame, values=train_values, state="readonly", width=5)
train_combo.grid(row=0, column=1, padx=5, pady=5)
train_combo.set("70")

val_label = ttk.Label(split_frame, text="Validation:")
val_label.grid(row=0, column=2, padx=5, pady=5)
# These comboboxes will be auto-adjusted; disable editing.
val_combo = ttk.Combobox(split_frame, state="disabled", width=5)
val_combo.grid(row=0, column=3, padx=5, pady=5)

test_label = ttk.Label(split_frame, text="Test:")
test_label.grid(row=0, column=4, padx=5, pady=5)
test_combo = ttk.Combobox(split_frame, state="disabled", width=5)
test_combo.grid(row=0, column=5, padx=5, pady=5)

split_label = ttk.Label(frame, text="Splits: Training: 70% | Validation: 20% | Test: 10%")
split_label.pack(pady=5)


def update_splits(event=None):
    try:
        train_percent = float(train_combo.get())
    except:
        train_percent = 70.0
    remaining = 100 - train_percent
    # We'll use a fixed ratio: Validation gets 2/3 of remainder, Test gets 1/3.
    val_percent = round(remaining * (2 / 3))
    test_percent = 100 - train_percent - val_percent
    val_combo.config(state="normal")
    test_combo.config(state="normal")
    val_combo.set(str(val_percent))
    test_combo.set(str(test_percent))
    val_combo.config(state="disabled")
    test_combo.config(state="disabled")
    split_label.config(text=f"Splits: Training: {train_percent}% | Validation: {val_percent}% | Test: {test_percent}%")


# Bind changes to the training combobox.
train_combo.bind("<<ComboboxSelected>>", update_splits)
update_splits()  # initial update

# Historical Frequency Section
hist_frame = ttk.LabelFrame(frame, text="Historical Frequency")
hist_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
calc_hist_button = ttk.Button(hist_frame, text="Calculate Historical Frequency", command=on_calculate_historical)
calc_hist_button.pack(pady=5)
hist_summary_tree = ttk.Treeview(hist_frame, columns=("L1 Value", "Count", "Frequency (%)"), show="headings", height=10)
hist_summary_tree.heading("L1 Value", text="L1 Value")
hist_summary_tree.heading("Count", text="Count")
hist_summary_tree.heading("Frequency (%)", text="Frequency (%)")
hist_summary_tree.column("L1 Value", width=100, anchor="center")
hist_summary_tree.column("Count", width=100, anchor="center")
hist_summary_tree.column("Frequency (%)", width=100, anchor="center")
hist_summary_tree.pack(pady=5, fill=tk.X)
overall_hist_label = ttk.Label(hist_frame, text="Overall Historical L1 Average: N/A")
overall_hist_label.pack(pady=5)
copy_hist_button = ttk.Button(hist_frame, text="Copy Historical Summary", command=copy_hist_summary)
copy_hist_button.pack(pady=5)

# Future Predictions Section
pred_frame = ttk.LabelFrame(frame, text="Future Draw Predictions")
pred_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
predict_button = ttk.Button(pred_frame, text="Predict Future Draws", command=on_predict)
predict_button.pack(pady=5)
progress_bar = ttk.Progressbar(pred_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=5)
progress_label = ttk.Label(pred_frame, text="Progress: 0%")
progress_label.pack(pady=2)
eta_label = ttk.Label(pred_frame, text="ETA: ")
eta_label.pack(pady=2)
prediction_tree = ttk.Treeview(pred_frame,
                               columns=("Prediction", "Predicted DN", "Predicted Date", "Predicted L1"),
                               show="headings", height=10)
prediction_tree.heading("Prediction", text="Prediction #")
prediction_tree.heading("Predicted DN", text="Predicted DN")
prediction_tree.heading("Predicted Date", text="Predicted Date")
prediction_tree.heading("Predicted L1", text="Predicted L1")
prediction_tree.column("Prediction", width=100, anchor="center")
prediction_tree.column("Predicted DN", width=120, anchor="center")
prediction_tree.column("Predicted Date", width=150, anchor="center")
prediction_tree.column("Predicted L1", width=120, anchor="center")
prediction_tree.pack(pady=10, fill=tk.X)
copy_pred_button = ttk.Button(pred_frame, text="Copy Predictions", command=copy_predictions)
copy_pred_button.pack(pady=5)

root.mainloop()
