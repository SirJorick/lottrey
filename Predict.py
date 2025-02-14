import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import csv
import re
from datetime import datetime, timedelta
import numpy as np
import sys
import threading

# Scikit-learn imports (ensure installation via: pip install scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ------------------ Constants ------------------

BASE_DIR = r"C:\Users\user\PycharmProjects\lottrey"
FILE_DATE_FORMAT = "%d-%b-%y"  # e.g., "01-Feb-25"
DISPLAY_DATE_FORMAT = "%b %d, %Y"  # e.g., "Feb 01, 2025"
MODE_OPTIONS = [
    "6_42.txt", "6_45.txt", "6_49.txt", "6_55.txt", "6_58.txt",
    "EZ2.txt", "Swertres.txt", "4D.txt", "6D.txt"
]
NUM_PREDICTIONS = 10  # Only 10 future draws

# Fixed split configurations
SPLIT_CONFIGS = [
    {"name": "70/20/10", "train": 70, "val": 20, "test": 10},
    {"name": "80/10/10", "train": 80, "val": 10, "test": 10},
    {"name": "90/5/5", "train": 90, "val": 5, "test": 5},
    {"name": "60/30/10", "train": 60, "val": 30, "test": 10},
    {"name": "75/15/10", "train": 75, "val": 15, "test": 10},
    {"name": "65/25/10", "train": 65, "val": 25, "test": 10},
]


# ------------------ Utility Functions ------------------

def get_file_filepath(filename):
    return os.path.join(BASE_DIR, filename)


def get_max_number(filename):
    m = re.search(r"_(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


# ------------------ Data Loading & Historical Frequency ------------------

def load_file_data(filepath):
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
                    data.append({'DN': DN, 'Draw Date': draw_date, 'L1': L1})
                except Exception:
                    continue
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file data: {e}")
    return data


def calculate_historical_frequency(data, max_number):
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
    if len(data) < 10:
        messagebox.showwarning("Not Enough Data", "Not enough data in the file to train models.")
        return None, None

    X = np.array([[row['DN'], row['Draw Date'].toordinal()] for row in data])
    y_L1 = np.array([row['L1'] for row in data])
    print("Total samples:", len(X), "Target samples:", len(y_L1))

    train_ratio = train_percent / 100.0
    X_train, X_rem, y_train, y_rem = train_test_split(X, y_L1, train_size=train_ratio, random_state=42)
    rem_total = 100 - train_percent
    val_ratio = val_percent / rem_total if rem_total > 0 else 0
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=val_ratio, random_state=42)

    X_date_train = X_train[:, 0].reshape(-1, 1)
    y_date_train = X_train[:, 1]

    model_date = LinearRegression().fit(X_date_train, y_date_train)
    # Use a pipeline with PolynomialFeatures and GradientBoostingRegressor for L1.
    model_L1 = make_pipeline(PolynomialFeatures(degree=2),
                             GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                       random_state=42)
                             ).fit(X_train, y_train)

    X_date_val = X_val[:, 0].reshape(-1, 1)
    val_date_pred = model_date.predict(X_date_val)
    val_mae_date = mean_absolute_error(X_val[:, 1], val_date_pred)
    val_L1_pred = model_L1.predict(X_val)
    val_mae_L1 = mean_absolute_error(y_val, val_L1_pred)
    print(f"Split {train_percent}/{val_percent}/{test_percent} - Validation MAE (Date):", val_mae_date)
    print(f"Split {train_percent}/{val_percent}/{test_percent} - Validation MAE (L1):", val_mae_L1)

    X_date_test = X_test[:, 0].reshape(-1, 1)
    test_date_pred = model_date.predict(X_date_test)
    test_mae_date = mean_absolute_error(X_test[:, 1], test_date_pred)
    test_L1_pred = model_L1.predict(X_test)
    test_mae_L1 = mean_absolute_error(y_test, test_L1_pred)
    print(f"Split {train_percent}/{val_percent}/{test_percent} - Test MAE (Date):", test_mae_date)
    print(f"Split {train_percent}/{val_percent}/{test_percent} - Test MAE (L1):", test_mae_L1)

    return model_date, model_L1


def predict_future_sync(model_date, model_L1, data, max_number):
    predictions = []
    data_sorted = sorted(data, key=lambda row: row['DN'])
    last_dn = data_sorted[-1]['DN']
    for i in range(1, NUM_PREDICTIONS + 1):
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
        predictions.append((new_dn, pred_date.strftime(DISPLAY_DATE_FORMAT), pred_L1))
    return predictions


def combine_predictions(all_predictions_dict):
    combined = []
    for split, preds in all_predictions_dict.items():
        for pred in preds:
            combined.append((split,) + pred)
    l1_counts = {}
    for entry in combined:
        l1 = entry[3]
        l1_counts[l1] = l1_counts.get(l1, 0) + 1
    mode_l1 = max(l1_counts, key=lambda k: l1_counts[k]) if l1_counts else None
    return combined, l1_counts, mode_l1


# ------------------ Console Output Redirector ------------------

class ConsoleRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, s):
        self.widget.insert(tk.END, s)
        self.widget.see(tk.END)

    def flush(self):
        pass


# ------------------ GUI Functions ------------------

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
    # Sort by frequency descending.
    sorted_freq = sorted([(l1, freq_dict[l1]) for l1 in range(1, max_num + 1) if freq_dict[l1][0] > 0],
                         key=lambda x: x[1][1], reverse=True)
    for item in hist_summary_tree.get_children():
        hist_summary_tree.delete(item)
    for l1, (count, freq) in sorted_freq:
        hist_summary_tree.insert("", "end", values=(f"{l1:02d}", count, f"{freq:.2f}%"))
    overall_hist_label.config(text=f"Overall Historical L1 Average: {overall_avg:.2f} | Total Draws: {total_draws}")


def on_predict_all():
    threading.Thread(target=_on_predict_all, daemon=True).start()


def _on_predict_all():
    selected_file = mode_combo.get()
    filepath = get_file_filepath(selected_file)
    max_num = get_max_number(selected_file)
    file_path_label.config(text=f"File path: {filepath}")
    data = load_file_data(filepath)
    if not data:
        messagebox.showerror("Error", "No valid data loaded from file.")
        return

    all_predictions = {}
    for config in SPLIT_CONFIGS:
        split_name = config["name"]
        print(f"Processing split: {split_name}\n")
        model_date, model_L1 = train_models_on_file(data, config["train"], config["val"], config["test"])
        if model_date is None or model_L1 is None:
            messagebox.showerror("Training Error", f"Failed to train models for split {split_name}.")
            return
        preds = predict_future_sync(model_date, model_L1, data, max_num)
        all_predictions[split_name] = preds

    combined_preds, l1_counts, mode_l1 = combine_predictions(all_predictions)

    # Sort combined predictions by predicted date.
    combined_preds_sorted = sorted(combined_preds, key=lambda x: datetime.strptime(x[2], DISPLAY_DATE_FORMAT))
    prediction_tree.after(0, lambda: update_predictions_tree(combined_preds_sorted))

    final_label.after(0, lambda: final_label.config(
        text=f"Final Predicted L1 (mode): {f'{mode_l1:02d}' if mode_l1 is not None else 'N/A'}"))

    total_preds = NUM_PREDICTIONS * len(SPLIT_CONFIGS)
    sorted_l1 = sorted(l1_counts.items(), key=lambda item: item[1], reverse=True)
    top10 = sorted_l1[:10]
    top10_tree.after(0, lambda: update_top10(top10, total_preds))


def update_top10(top10, total_preds):
    for item in top10_tree.get_children():
        top10_tree.delete(item)
    for l1, count in top10:
        freq = (count / total_preds * 100) if total_preds > 0 else 0
        top10_tree.insert("", "end", values=(f"{l1:02d}", count, f"{freq:.2f}%"))


def update_predictions_tree(predictions):
    for item in prediction_tree.get_children():
        prediction_tree.delete(item)
    for entry in predictions:
        split, new_dn, pred_date, pred_L1 = entry
        prediction_tree.insert("", "end", values=(split, new_dn, pred_date, f"{pred_L1:02d}"))


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


def copy_top10():
    rows = []
    headers = [top10_tree.heading(col)["text"] for col in top10_tree["columns"]]
    rows.append("\t".join(headers))
    for item in top10_tree.get_children():
        values = top10_tree.item(item, "values")
        rows.append("\t".join(str(v) for v in values))
    text = "\n".join(rows)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copied", "Top-10 prediction summary copied to clipboard.")


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


# ------------------ Console Output Redirector ------------------

class ConsoleRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, s):
        self.widget.insert(tk.END, s)
        self.widget.see(tk.END)

    def flush(self):
        pass


# ------------------ GUI Setup ------------------

root = tk.Tk()
root.title("Lottery Predictor: Multiple Splits, Final L1 Mode & Top-10")

frame = ttk.Frame(root, padding=20)
frame.pack(fill=tk.BOTH, expand=True)

# Use a PanedWindow to split left and right panels.
paned = ttk.PanedWindow(frame, orient="horizontal")
paned.pack(fill=tk.BOTH, expand=True)

left_frame = ttk.Frame(paned, padding=10)
paned.add(left_frame, weight=3)

right_frame = ttk.Frame(paned, padding=10)
paned.add(right_frame, weight=1)

# --- Left Panel ---

# File selection
mode_label = ttk.Label(left_frame, text="Select Lottery File:")
mode_label.pack(pady=5)
mode_combo = ttk.Combobox(left_frame, values=MODE_OPTIONS, state="readonly")
mode_combo.pack(pady=5)
mode_combo.set("6_42.txt")
file_path_label = ttk.Label(left_frame, text="File path will appear here.")
file_path_label.pack(pady=5)

# Historical Frequency Section
hist_frame = ttk.LabelFrame(left_frame, text="Historical Frequency")
hist_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
calc_hist_button = ttk.Button(hist_frame, text="Calculate Historical Frequency", command=on_calculate_historical)
calc_hist_button.pack(pady=5)
hist_summary_tree = ttk.Treeview(hist_frame, columns=("L1 Value", "Count", "Frequency (%)"), show="headings", height=8)
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
pred_frame = ttk.LabelFrame(left_frame, text="Future Draw Predictions (All Splits)")
pred_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
predict_all_button = ttk.Button(pred_frame, text="Predict All Splits", command=on_predict_all)
predict_all_button.pack(pady=5)
progress_bar = ttk.Progressbar(pred_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=5)
progress_label = ttk.Label(pred_frame, text="Progress: 0%")
progress_label.pack(pady=2)
eta_label = ttk.Label(pred_frame, text="ETA: ")
eta_label.pack(pady=2)
prediction_tree = ttk.Treeview(pred_frame,
                               columns=("Split", "Predicted DN", "Predicted Date", "Predicted L1"),
                               show="headings", height=10)
prediction_tree.heading("Split", text="Split")
prediction_tree.heading("Predicted DN", text="Predicted DN")
prediction_tree.heading("Predicted Date", text="Predicted Date")
prediction_tree.heading("Predicted L1", text="Predicted L1")
prediction_tree.column("Split", width=100, anchor="center")
prediction_tree.column("Predicted DN", width=120, anchor="center")
prediction_tree.column("Predicted Date", width=150, anchor="center")
prediction_tree.column("Predicted L1", width=120, anchor="center")
prediction_tree.pack(pady=10, fill=tk.X)
final_label = ttk.Label(pred_frame, text="Final Predicted L1 (mode): N/A", font=("Helvetica", 12, "bold"))
final_label.pack(pady=5)
copy_pred_button = ttk.Button(pred_frame, text="Copy Predictions", command=copy_predictions)
copy_pred_button.pack(pady=5)

# --- Right Panel: Top-10 & Console Output ---

right_paned = ttk.PanedWindow(right_frame, orient="vertical")
right_paned.pack(fill=tk.BOTH, expand=True)

top10_frame = ttk.LabelFrame(right_paned, text="Top-10 Predicted L1 Values")
right_paned.add(top10_frame, weight=1)
top10_tree = ttk.Treeview(top10_frame, columns=("L1 Value", "Count", "Frequency (%)"), show="headings", height=8)
top10_tree.heading("L1 Value", text="L1 Value")
top10_tree.heading("Count", text="Count")
top10_tree.heading("Frequency (%)", text="Frequency (%)")
top10_tree.column("L1 Value", width=100, anchor="center")
top10_tree.column("Count", width=100, anchor="center")
top10_tree.column("Frequency (%)", width=100, anchor="center")
top10_tree.pack(pady=5, fill=tk.BOTH, expand=True)
copy_top10_button = ttk.Button(top10_frame, text="Copy Top-10", command=copy_top10)
copy_top10_button.pack(pady=5)

console_frame = ttk.LabelFrame(right_paned, text="Console Output")
right_paned.add(console_frame, weight=1)
console_text = ScrolledText(console_frame, height=10, bg="black", fg="white", insertbackground="white")
console_text.pack(fill=tk.BOTH, expand=True)

# Redirect print output to console_text.
sys.stdout = ConsoleRedirector(console_text)


def copy_top10():
    rows = []
    headers = [top10_tree.heading(col)["text"] for col in top10_tree["columns"]]
    rows.append("\t".join(headers))
    for item in top10_tree.get_children():
        values = top10_tree.item(item, "values")
        rows.append("\t".join(str(v) for v in values))
    text = "\n".join(rows)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copied", "Top-10 prediction summary copied to clipboard.")


root.mainloop()
