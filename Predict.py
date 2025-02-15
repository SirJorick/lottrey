import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import os, csv, re
from datetime import datetime, timedelta
import numpy as np
import sys, threading, time

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# Try to import XGBoost and LightGBM
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

# ------------------ Constants ------------------
BASE_DIR = r"C:\Users\user\PycharmProjects\lottrey"
FILE_DATE_FORMAT = "%d-%b-%y"
DISPLAY_DATE_FORMAT = "%b %d, %Y"
MODE_OPTIONS = ["6_42.txt", "6_45.txt", "6_49.txt", "6_55.txt", "6_58.txt",
                "EZ2.txt", "Swertres.txt", "4D.txt", "6D.txt"]
NUM_PREDICTIONS = 10

SPLIT_CONFIGS = [
    {"name": "70/20/10", "train": 70, "val": 20, "test": 10},
    {"name": "80/10/10", "train": 80, "val": 10, "test": 10},
    {"name": "90/5/5", "train": 90, "val": 5, "test": 5},
    {"name": "60/30/10", "train": 60, "val": 30, "test": 10},
    {"name": "75/15/10", "train": 75, "val": 15, "test": 10},
    {"name": "65/25/10", "train": 65, "val": 25, "test": 10},
]

target_col = "L1"


# ------------------ Utility Functions ------------------
def get_file_filepath(filename):
    return os.path.join(BASE_DIR, filename)


def get_max_number(filename):
    m = re.search(r"_(\d+)", filename)
    return int(m.group(1)) if m else None


# ------------------ Data Loading ------------------
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
                    data.append({
                        'DN': DN,
                        'Draw Date': draw_date,
                        'L1': int(row['L1'].strip()),
                        'L2': int(row['L2'].strip()) if 'L2' in row and row['L2'].strip() else None,
                        'L3': int(row['L3'].strip()) if 'L3' in row and row['L3'].strip() else None,
                        'L4': int(row['L4'].strip()) if 'L4' in row and row['L4'].strip() else None,
                        'L5': int(row['L5'].strip()) if 'L5' in row and row['L5'].strip() else None,
                        'L6': int(row['L6'].strip()) if 'L6' in row and row['L6'].strip() else None,
                    })
                except Exception:
                    continue
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file data: {e}")
    return data


def calculate_historical_frequency(data, max_number):
    global target_col
    if not data:
        return None, 0, 0
    total_draws = max(row['DN'] for row in data)
    freq_dict = {}
    for l in range(1, max_number + 1):
        count = sum(1 for row in data if row[target_col] == l)
        freq = (count / total_draws * 100) if total_draws > 0 else 0
        freq_dict[l] = (count, freq)
    overall_avg = sum(row[target_col] for row in data) / len(data)
    return freq_dict, overall_avg, total_draws


# ------------------ Model Training & Prediction ------------------
def train_models_on_file(data, train_percent, val_percent, test_percent):
    global target_col
    if len(data) < 10:
        messagebox.showwarning("Not Enough Data", "Not enough data to train models.")
        return None, None
    X = np.array([[row['DN'], row['Draw Date'].toordinal()] for row in data])
    y = np.array([row[target_col] for row in data])
    print("Total samples:", len(X), "Target samples:", len(y))

    train_ratio = train_percent / 100.0
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_ratio, random_state=42)
    rem_total = 100 - train_percent
    val_ratio = val_percent / rem_total if rem_total > 0 else 0
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=val_ratio, random_state=42)

    X_date_train = X_train[:, 0].reshape(-1, 1)
    y_date_train = X_train[:, 1]
    model_date = LinearRegression().fit(X_date_train, y_date_train)

    # Train six base models.
    model_gb = make_pipeline(PolynomialFeatures(degree=2),
                             GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                       random_state=42)
                             ).fit(X_train, y_train)
    model_rf = make_pipeline(PolynomialFeatures(degree=2),
                             RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                             ).fit(X_train, y_train)
    model_et = make_pipeline(PolynomialFeatures(degree=2),
                             ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                             ).fit(X_train, y_train)
    model_svr = make_pipeline(PolynomialFeatures(degree=2),
                              StandardScaler(),
                              SVR(kernel='rbf')
                              ).fit(X_train, y_train)
    model_mlp = make_pipeline(StandardScaler(),
                              MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
                              ).fit(X_train, y_train)
    model_br = make_pipeline(PolynomialFeatures(degree=2),
                             StandardScaler(),
                             BayesianRidge()
                             ).fit(X_train, y_train)
    # Additional advanced models (if available)
    base_models = [model_gb, model_rf, model_et, model_svr, model_mlp, model_br]
    if XGBRegressor is not None:
        model_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42, verbosity=0)
        model_xgb.fit(X_train, y_train)
        base_models.append(model_xgb)
    else:
        model_xgb = None
    if LGBMRegressor is not None:
        model_lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
        model_lgb.fit(X_train, y_train)
        base_models.append(model_lgb)
    else:
        model_lgb = None

    # Build meta-model using stacking on validation set.
    preds_val = [model.predict(X_val) for model in base_models]
    meta_features_val = np.column_stack(preds_val)
    meta_model = Ridge(alpha=1.0).fit(meta_features_val, y_val)

    meta_pred_val = meta_model.predict(meta_features_val)
    val_mae = mean_absolute_error(y_val, meta_pred_val)
    print(f"Ensemble Validation MAE ({target_col}):", val_mae)

    preds_test = [model.predict(X_test) for model in base_models]
    meta_features_test = np.column_stack(preds_test)
    meta_pred_test = meta_model.predict(meta_features_test)
    test_mae = mean_absolute_error(y_test, meta_pred_test)
    print(f"Ensemble Test MAE ({target_col}):", test_mae)

    # Return date model and tuple: base models and meta-model.
    return model_date, (model_gb, model_rf, model_et, model_svr, model_mlp, model_br, model_xgb, model_lgb, meta_model)


def predict_future_sync(model_date, model_target_tuple, data, max_number):
    global target_col
    (
    model_gb, model_rf, model_et, model_svr, model_mlp, model_br, model_xgb, model_lgb, meta_model) = model_target_tuple
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
        base_preds = []
        for model in [model_gb, model_rf, model_et, model_svr, model_mlp, model_br]:
            base_preds.append(model.predict(np.array([[new_dn, pred_date_ord]]))[0])
        if model_xgb is not None:
            base_preds.append(model_xgb.predict(np.array([[new_dn, pred_date_ord]]))[0])
        if model_lgb is not None:
            base_preds.append(model_lgb.predict(np.array([[new_dn, pred_date_ord]]))[0])
        meta_features = np.array(base_preds).reshape(1, -1)
        pred_target = meta_model.predict(meta_features)[0]
        std_target = np.std(np.array([row[target_col] for row in data]))
        noise = np.random.normal(0, std_target * 0.5)
        pred_target += noise
        pred_target = int(round(pred_target))
        if max_number is not None:
            pred_target = max(1, min(pred_target, max_number))
        predictions.append((new_dn, pred_date.strftime(DISPLAY_DATE_FORMAT), pred_target))
    return predictions


def combine_predictions(all_predictions_dict):
    combined = []
    dn_dict = {}
    for split, preds in all_predictions_dict.items():
        for pred in preds:
            entry = (split,) + pred  # (split, new_dn, pred_date, pred_target)
            combined.append(entry)
            val = entry[3]
            dn_dict.setdefault(val, []).append(entry[1])
    counts = {val: len(dn_dict[val]) for val in dn_dict}
    mode_val = max(counts, key=lambda k: counts[k]) if counts else None
    avg_dn = {val: np.mean(dn_dict[val]) for val in dn_dict}
    return combined, counts, mode_val, avg_dn


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
def update_target_labels(event=None):
    global target_col
    target_col = column_combo.get()
    hist_summary_tree.heading("L1 Value", text="Historical " + target_col)
    prediction_tree.heading("Predicted Value", text="Predicted " + target_col)
    final_label.config(text="Final Predicted " + target_col + " (mode): N/A")
    top10_frame.config(text="Top-10 Predicted " + target_col + " Values")


def on_calculate_historical():
    global target_col
    target_col = column_combo.get()
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
    for l, (count, freq) in sorted([(l, freq_dict[l]) for l in range(1, max_num + 1) if freq_dict[l][0] > 0],
                                   key=lambda x: x[1][1], reverse=True):
        hist_summary_tree.insert("", "end", values=(f"{l:02d}", count, f"{freq:.2f}%"))
    overall_hist_label.config(
        text=f"Overall Historical {target_col} Average: {overall_avg:.2f} | Total Draws: {total_draws}")


def on_predict_all():
    threading.Thread(target=_on_predict_all, daemon=True).start()


def _on_predict_all():
    global target_col
    target_col = column_combo.get()
    selected_file = mode_combo.get()
    filepath = get_file_filepath(selected_file)
    max_num = get_max_number(selected_file)
    file_path_label.config(text=f"File path: {filepath}")
    data = load_file_data(filepath)
    if not data:
        messagebox.showerror("Error", "No valid data loaded from file.")
        return

    # Compute Next DN (Last DN + 1)
    last_dn = max(row['DN'] for row in data)
    next_dn = last_dn + 1
    next_dn_label.config(text=f"Next DN: {next_dn}")

    all_predictions = {}
    total_configs = len(SPLIT_CONFIGS)
    start_time = time.time()
    next_draw_preds = []
    for idx, config in enumerate(SPLIT_CONFIGS):
        split_name = config["name"]
        print(f"Processing split: {split_name}\n")
        model_date, model_target = train_models_on_file(data, config["train"], config["val"], config["test"])
        if model_date is None or model_target is None:
            messagebox.showerror("Training Error", f"Failed to train models for split {split_name}.")
            return
        preds = predict_future_sync(model_date, model_target, data, max_num)
        all_predictions[split_name] = preds
        if preds:
            next_draw_preds.append(preds[0][2])
        progress = int((idx + 1) / total_configs * 100)
        elapsed = time.time() - start_time
        estimated_total = elapsed / (idx + 1) * total_configs
        eta = estimated_total - elapsed
        root.after(0, update_progress, progress, eta)
        time.sleep(0.1)

    if next_draw_preds:
        next_draw_prediction = np.mean(next_draw_preds)
    else:
        next_draw_prediction = None
    next_draw_pred_label.config(
        text=f"Next Draw Prediction: {next_draw_prediction:.1f}" if next_draw_prediction is not None else "Next Draw Prediction: N/A")

    combined_preds, counts, mode_val, avg_dn = combine_predictions(all_predictions)
    combined_preds_sorted = sorted(combined_preds, key=lambda x: datetime.strptime(x[2], DISPLAY_DATE_FORMAT))

    # Insert predictions into the Future Prediction TreeView.
    for idx, entry in enumerate(combined_preds_sorted):
        split, new_dn, pred_date, pred_val = entry
        # Mark the first row as "Priority"
        if idx == 0:
            priority_text = "Priority: " + split
            prediction_tree.insert("", "end", values=(priority_text, new_dn, pred_date, f"{pred_val:02d}"),
                                   tags=("priority",))
        else:
            prediction_tree.insert("", "end", values=(split, new_dn, pred_date, f"{pred_val:02d}"))
    prediction_tree.tag_configure("priority", background="lightblue")

    final_label.after(0, lambda: final_label.config(
        text=f"Final Predicted {target_col} (mode): {f'{mode_val:02d}' if mode_val is not None else 'N/A'}"))

    total_preds = NUM_PREDICTIONS * total_configs
    # Adjust ranking by closeness to the global next draw prediction.
    if next_draw_prediction is not None:
        adjusted = []
        for val, count in counts.items():
            score = count / (1 + abs(avg_dn[val] - next_draw_prediction))
            adjusted.append((val, count, avg_dn[val], score))
        adjusted_sorted = sorted(adjusted, key=lambda x: x[3], reverse=True)
    else:
        adjusted_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top10 = adjusted_sorted[:10]
    for item in top10_tree.get_children():
        top10_tree.delete(item)
    for tup in top10:
        # tup = (val, count, avg_dn, score)
        val, count, avg_dn_val, score = tup
        freq = (count / total_preds * 100) if total_preds > 0 else 0
        top10_tree.insert("", "end", values=(f"{val:02d}", count, f"{freq:.2f}%", f"{avg_dn_val:.1f}"))


def update_progress(p, eta):
    progress_bar.config(value=p)
    progress_label.config(text=f"Progress: {p}%")
    eta_label.config(text=f"ETA: {eta:.1f} sec")
    root.update()


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


# ------------------ GUI Setup ------------------
root = tk.Tk()
root.title("Lottery Predictor: Multiple Splits, Final Mode & Top-10")

frame = ttk.Frame(root, padding=20)
frame.pack(fill=tk.BOTH, expand=True)

paned = ttk.PanedWindow(frame, orient="horizontal")
paned.pack(fill=tk.BOTH, expand=True)

left_frame = ttk.Frame(paned, padding=10)
paned.add(left_frame, weight=3)

right_frame = ttk.Frame(paned, padding=10)
paned.add(right_frame, weight=1)

# --- Left Panel ---
column_label = ttk.Label(left_frame, text="Select Column to Predict (L1 to L6):")
column_label.pack(pady=5)
column_combo = ttk.Combobox(left_frame, values=["L1", "L2", "L3", "L4", "L5", "L6"], state="readonly")
column_combo.pack(pady=5)
column_combo.set("L1")
column_combo.bind("<<ComboboxSelected>>", update_target_labels)

mode_label = ttk.Label(left_frame, text="Select Lottery File:")
mode_label.pack(pady=5)
mode_combo = ttk.Combobox(left_frame, values=MODE_OPTIONS, state="readonly")
mode_combo.pack(pady=5)
mode_combo.set("6_42.txt")
file_path_label = ttk.Label(left_frame, text="File path will appear here.")
file_path_label.pack(pady=5)

hist_frame = ttk.LabelFrame(left_frame, text="Historical Frequency")
hist_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
calc_hist_button = ttk.Button(hist_frame, text="Calculate Historical Frequency", command=on_calculate_historical)
calc_hist_button.pack(pady=5)
hist_summary_tree = ttk.Treeview(hist_frame, columns=("L1 Value", "Count", "Frequency (%)"), show="headings", height=8)
hist_summary_tree.heading("L1 Value", text="Historical " + target_col)
hist_summary_tree.heading("Count", text="Count")
hist_summary_tree.heading("Frequency (%)", text="Frequency (%)")
hist_summary_tree.column("L1 Value", width=100, anchor="center")
hist_summary_tree.column("Count", width=100, anchor="center")
hist_summary_tree.column("Frequency (%)", width=100, anchor="center")
hist_summary_tree.pack(pady=5, fill=tk.X)
overall_hist_label = ttk.Label(hist_frame, text="Overall Historical " + target_col + " Average: N/A")
overall_hist_label.pack(pady=5)
copy_hist_button = ttk.Button(hist_frame, text="Copy Historical Summary", command=copy_hist_summary)
copy_hist_button.pack(pady=5)

pred_frame = ttk.LabelFrame(left_frame, text="Future Draw Predictions (All Splits)")
pred_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
predict_all_button = ttk.Button(pred_frame, text="Predict All Splits", command=on_predict_all)
predict_all_button.pack(pady=5)
progress_frame = ttk.Frame(pred_frame)
progress_frame.pack(pady=5, fill=tk.X)
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
progress_bar.grid(row=0, column=0, padx=5)
progress_label = ttk.Label(progress_frame, text="Progress: 0%")
progress_label.grid(row=0, column=1, padx=5)
eta_label = ttk.Label(progress_frame, text="ETA: 0 sec")
eta_label.grid(row=0, column=2, padx=5)
next_draw_pred_label = ttk.Label(pred_frame, text="Next Draw Prediction: N/A", font=("Helvetica", 14, "bold"))
next_draw_pred_label.pack(pady=5)
prediction_tree = ttk.Treeview(pred_frame,
                               columns=("Split", "Predicted DN", "Predicted Date", "Predicted " + target_col),
                               show="headings", height=10)
prediction_tree.heading("Split", text="Split")
prediction_tree.heading("Predicted DN", text="Predicted DN")
prediction_tree.heading("Predicted Date", text="Predicted Date")
prediction_tree.heading("Predicted " + target_col, text="Predicted " + target_col)
prediction_tree.column("Split", width=100, anchor="center")
prediction_tree.column("Predicted DN", width=120, anchor="center")
prediction_tree.column("Predicted Date", width=150, anchor="center")
prediction_tree.column("Predicted " + target_col, width=120, anchor="center")
prediction_tree.pack(pady=10, fill=tk.X)
final_label = ttk.Label(pred_frame, text="Final Predicted " + target_col + " (mode): N/A",
                        font=("Helvetica", 12, "bold"))
final_label.pack(pady=5)
copy_pred_button = ttk.Button(pred_frame, text="Copy Predictions", command=copy_predictions)
copy_pred_button.pack(pady=5)

# --- Right Panel: Next DN Label, Top-10 & Console Output ---
next_dn_label = ttk.Label(right_frame, text="Next DN: N/A", font=("Helvetica", 14, "bold"))
next_dn_label.pack(pady=5)
right_paned = ttk.PanedWindow(right_frame, orient="vertical")
right_paned.pack(fill=tk.BOTH, expand=True)
top10_frame = ttk.LabelFrame(right_paned, text="Top-10 Predicted " + target_col + " Values")
right_paned.add(top10_frame, weight=1)
top10_tree = ttk.Treeview(top10_frame, columns=("Value", "Count", "Frequency (%)", "Next DN"), show="headings",
                          height=8)
top10_tree.heading("Value", text="Value")
top10_tree.heading("Count", text="Count")
top10_tree.heading("Frequency (%)", text="Frequency (%)")
top10_tree.heading("Next DN", text="Next DN")
top10_tree.column("Value", width=80, anchor="center")
top10_tree.column("Count", width=80, anchor="center")
top10_tree.column("Frequency (%)", width=80, anchor="center")
top10_tree.column("Next DN", width=120, anchor="center")
top10_tree.pack(pady=5, fill=tk.BOTH, expand=True)
copy_top10_button = ttk.Button(top10_frame, text="Copy Top-10", command=copy_top10)
copy_top10_button.pack(pady=5)
# Use default header style (not bold)

console_frame = ttk.LabelFrame(right_paned, text="Console Output")
right_paned.add(console_frame, weight=1)
console_text = ScrolledText(console_frame, height=10, bg="black", fg="white", insertbackground="white")
console_text.pack(fill=tk.BOTH, expand=True)
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
