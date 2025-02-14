import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from tkcalendar import DateEntry  # Requires: pip install tkcalendar
import subprocess
import threading

# Configuration settings
BASE_DIR = r"C:\Users\user\PycharmProjects\lottrey"

# Define file columns (the structure remains the same)
FILE_COLUMNS = ["DN", "Draw Date", "L1", "L2", "L3", "L4", "L5", "L6"]

# Date formats:
FILE_DATE_FORMAT = "%d-%b-%y"  # e.g., "01-Feb-25" (used in file saving/reading)
TK_DATE_PATTERN = "dd-mm-yy"   # e.g., "01-02-25" (used by tkcalendar DateEntry)

# Mode options (text filenames)
MODE_OPTIONS = ["6_42.txt", "6_45.txt", "6_49.txt", "6_55.txt", "6_58.txt",
                "EZ2.txt", "Swertres.txt", "4D.txt", "6D.txt"]

def get_file_filepath(filename):
    """Return the absolute file path by joining BASE_DIR and the filename."""
    return os.path.join(BASE_DIR, filename)

class LotteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCSO Lottery - ")
        self.root.geometry("1050x700")

        # For file watcher: store last modification time
        self.last_mod_time = None
        # Flag for blinking status and storage for after job id.
        self.blinking = False
        self.blink_job = None

        # Mode selection variable (for file selection)
        self.mode_var = tk.StringVar(value="6_42.txt")
        self.txt_filename = self.get_txt_filename()  # absolute path with .txt extension
        self.data = []  # List of dictionaries holding records
        self.editing_mode = False
        self.original_dn = None  # To store original DN when editing
        self.original_date = None  # To store original Draw Date when editing

        self.create_widgets()
        self.load_data()
        self.refresh_treeview()
        self.auto_fill_new_record()
        self.start_file_watch()

    def get_txt_filename(self, mode_value=None):
        if mode_value is None:
            mode_value = self.mode_var.get().strip() or "6_42.txt"
        return get_file_filepath(mode_value)

    def get_heading_text(self):
        """
        Return the heading text.
        Remove the ".txt" extension and replace underscores with slashes.
        For example, "6_42.txt" becomes "PCSO Lottery - 6/42".
        """
        mode = self.mode_var.get().strip() or "6_42.txt"
        file_no = mode.replace(".txt", "").replace("_", "/")
        return f"PCSO Lottery - {file_no}"

    def load_data(self):
        self.data = []
        if not os.path.exists(self.txt_filename):
            response = messagebox.askyesno("File Not Found",
                                           f"{self.txt_filename} was not found.\nDo you want to create a new file?")
            if response:
                self.save_data()  # Create the file with header.
            return
        try:
            with open(self.txt_filename, "r", encoding="utf-8") as file:
                lines = [line.rstrip("\n") for line in file if line.strip()]
            if not lines:
                self.data = []
                return

            self.last_mod_time = os.path.getmtime(self.txt_filename)
            first_line_fields = lines[0].split("\t")
            has_header = (len(first_line_fields) == len(FILE_COLUMNS) and first_line_fields[0] == "DN")

            records = []
            data_lines = lines[1:] if has_header else lines
            for line in data_lines:
                row = line.split("\t")
                if len(row) == len(FILE_COLUMNS) - 1:
                    row = [""] + row
                if len(row) < len(FILE_COLUMNS):
                    continue
                row_dict = {col: row[idx] for idx, col in enumerate(FILE_COLUMNS)}
                records.append(row_dict)

            try:
                records.sort(key=lambda r: datetime.strptime(r["Draw Date"].strip(), FILE_DATE_FORMAT))
            except Exception as e:
                messagebox.showerror("Date Sort Error", f"Error sorting records by date: {e}")

            for i, row in enumerate(records):
                row["DN"] = str(i + 1)

            self.data = records
            self.save_data()
            self.heading_label.config(text=self.get_heading_text())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def save_data(self):
        try:
            os.makedirs(os.path.dirname(self.txt_filename), exist_ok=True)
            with open(self.txt_filename, "w", encoding="utf-8") as file:
                file.write("\t".join(FILE_COLUMNS) + "\n")
                for row in self.data:
                    line = "\t".join([row.get(col, "") for col in FILE_COLUMNS])
                    file.write(line + "\n")
            self.last_mod_time = os.path.getmtime(self.txt_filename)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data: {e}")

    def create_widgets(self):
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill="x")
        self.heading_label = ttk.Label(header_frame,
                                       text=self.get_heading_text(),
                                       font=("Arial", 24, "bold"),
                                       foreground="blue")
        self.heading_label.pack(side="left", pady=10)
        self.status_label = tk.Label(header_frame, text="", font=("Arial", 14, "bold"))
        self.status_label.pack(side="right", padx=20)

        sub_header_frame = ttk.Frame(header_frame)
        sub_header_frame.pack(side="top", fill="x")
        mode_label = ttk.Label(sub_header_frame, text="Select File:")
        mode_label.pack(side="left", padx=(20, 5))
        self.mode_combo = ttk.Combobox(sub_header_frame, textvariable=self.mode_var,
                                       values=MODE_OPTIONS, width=10, state="readonly")
        self.mode_combo.pack(side="left")
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)
        self.last_record_button = ttk.Button(sub_header_frame, text="Last Record: N/A", command=self.go_to_last_record)
        self.last_record_button.pack(side="right")

        search_frame = ttk.Frame(self.root, padding=10)
        search_frame.pack(fill="x")
        search_label = ttk.Label(search_frame, text="Search by Date:")
        search_label.pack(side="left", padx=(20, 5))
        self.search_date_entry = DateEntry(search_frame, date_pattern=TK_DATE_PATTERN)
        self.search_date_entry.pack(side="left")
        search_button = ttk.Button(search_frame, text="Search", command=self.search_by_date)
        search_button.pack(side="left", padx=5)
        reset_button = ttk.Button(search_frame, text="Reset", command=self.reset_search)
        reset_button.pack(side="left", padx=5)

        tree_frame = ttk.Frame(self.root, padding=10)
        tree_frame.pack(fill="both", expand=True)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Treeview",
                        background="white",
                        foreground="black",
                        rowheight=25,
                        fieldbackground="white")
        style.configure("Treeview.Heading",
                        background="#4CAF50",
                        foreground="white",
                        font=("Arial", 10, "bold"))
        self.tree = ttk.Treeview(tree_frame, columns=FILE_COLUMNS, show="headings", height=10)
        self.tree.pack(side="left", fill="both", expand=True)
        for col in FILE_COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=100)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        form_frame = ttk.LabelFrame(self.root, text="Record Details", padding=10)
        form_frame.pack(fill="x", padx=10, pady=10)
        self.entries = {}
        self.entry_vars = {}
        for idx, col in enumerate(FILE_COLUMNS):
            lbl = ttk.Label(form_frame, text=col)
            lbl.grid(row=0, column=idx, padx=5, pady=5)
            var = tk.StringVar()
            var.trace("w", lambda *args, c=col: self.validate_current_inputs())
            ent = ttk.Entry(form_frame, textvariable=var, width=12)
            ent.grid(row=1, column=idx, padx=5, pady=5)
            if col in ["L1", "L2", "L3", "L4", "L5", "L6"]:
                ent.bind("<FocusOut>", lambda e, c=col: self.format_var(c))
            if col == "DN":
                ent.config(state="disabled")
            self.entries[col] = ent
            self.entry_vars[col] = var

        self.validation_error_label = ttk.Label(form_frame, text="", foreground="red")
        self.validation_error_label.grid(row=2, column=0, columnspan=len(FILE_COLUMNS), pady=(5, 0))

        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(fill="x")
        add_button = ttk.Button(button_frame, text="Add", command=self.add_record)
        add_button.pack(side="left", padx=5)
        update_button = ttk.Button(button_frame, text="Update", command=self.update_record)
        update_button.pack(side="left", padx=5)
        delete_button = ttk.Button(button_frame, text="Delete", command=self.delete_record)
        delete_button.pack(side="left", padx=5)
        clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_form)
        clear_button.pack(side="left", padx=5)
        caption_button = ttk.Button(button_frame, text="PCSO", command=self.start_pcso_scraper)
        caption_button.pack(side="left", padx=5)

    def on_mode_change(self, event):
        self.txt_filename = self.get_txt_filename()
        self.heading_label.config(text=self.get_heading_text())
        self.load_data()
        self.refresh_treeview()
        self.clear_form()

    def format_var(self, col):
        value = self.entry_vars[col].get().strip()
        if value:
            try:
                num = int(value)
                formatted = f"{num:02d}"
                if value != formatted:
                    self.entry_vars[col].set(formatted)
            except ValueError:
                pass

    def validate_current_inputs(self):
        errors = []
        for col in FILE_COLUMNS:
            if not self.entry_vars[col].get().strip():
                errors.append(f"{col} cannot be empty.")
        lottery_vals = []
        for col in ["L1", "L2", "L3", "L4", "L5", "L6"]:
            val = self.entry_vars[col].get().strip()
            if val:
                try:
                    lottery_vals.append(int(val))
                except ValueError:
                    errors.append(f"{col} must be a valid integer.")
                    break
        if len(lottery_vals) == 6:
            for i in range(len(lottery_vals) - 1):
                if lottery_vals[i] >= lottery_vals[i + 1]:
                    errors.append(f"{['L1','L2','L3','L4','L5','L6'][i]} must be less than {['L1','L2','L3','L4','L5','L6'][i+1]}.")
                    break
        current_dn = self.entry_vars["DN"].get().strip()
        current_date = self.entry_vars["Draw Date"].get().strip()
        for row in self.data:
            if self.editing_mode and self.original_dn and row.get("DN").strip() == self.original_dn:
                continue
            if current_dn and row.get("DN").strip() == current_dn:
                errors.append("Duplicate DN found.")
                break
        for row in self.data:
            if self.editing_mode and self.original_date and row.get("Draw Date").strip() == self.original_date:
                continue
            if current_date and row.get("Draw Date").strip() == current_date:
                errors.append("Duplicate Draw Date found.")
                break
        self.validation_error_label.config(text=" | ".join(errors))
        return errors

    def auto_fill_new_record(self):
        if self.data:
            try:
                max_dn = max(int(row["DN"]) for row in self.data if row.get("DN"))
            except ValueError:
                max_dn = 0
            next_dn = max_dn + 1
            self.entry_vars["DN"].set(str(next_dn))
        else:
            self.entry_vars["DN"].set("1")
        today_str = datetime.today().strftime(FILE_DATE_FORMAT)
        self.entry_vars["Draw Date"].set(today_str)
        self.editing_mode = False
        self.original_dn = None
        self.original_date = None

    def refresh_treeview(self, records=None):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if records is None:
            records = self.data
        try:
            sorted_data = sorted(records, key=lambda x: int(x["DN"]), reverse=True)
        except ValueError:
            sorted_data = sorted(records, key=lambda x: x["DN"], reverse=True)
        for row in sorted_data:
            formatted_row = row.copy()
            for key in ["L1", "L2", "L3", "L4", "L5", "L6"]:
                value = formatted_row.get(key, "").strip()
                if value:
                    try:
                        formatted_row[key] = f"{int(value):02d}"
                    except ValueError:
                        pass
            values = [formatted_row.get(col, "") for col in FILE_COLUMNS]
            self.tree.insert("", "end", values=values)
        if sorted_data:
            last_record = max(self.data, key=lambda r: datetime.strptime(r["Draw Date"].strip(), FILE_DATE_FORMAT))
            text = f"Last Record: DN: {last_record['DN']}, Draw Date: {last_record['Draw Date']}"
            self.last_record_button.config(text=text)
        else:
            self.last_record_button.config(text="No record available")
        self.update_status()

    def update_status(self):
        if self.data:
            try:
                last_record = max(self.data, key=lambda r: datetime.strptime(r["Draw Date"].strip(), FILE_DATE_FORMAT))
                last_date = datetime.strptime(last_record["Draw Date"].strip(), FILE_DATE_FORMAT).date()
                today_date = datetime.today().date()
                if last_date > today_date:
                    self.blinking = True
                    self.blink_status("Date Error", "red")
                    return
                else:
                    self.blinking = False
                    if hasattr(self, 'blink_job') and self.blink_job is not None:
                        self.root.after_cancel(self.blink_job)
                        self.blink_job = None
                diff_days = (today_date - last_date).days
                today_weekday = today_date.weekday()  # Monday=0, Tuesday=1, etc.
                if today_weekday in [1, 3, 5]:
                    if diff_days <= 1:
                        status_text = "UPDATED"
                        fg_color = "green"
                    elif diff_days <= 3:
                        status_text = "Need UPDATE"
                        fg_color = "red"
                    else:
                        status_text = "Need UPDATE"
                        fg_color = "red"
                else:
                    if diff_days < 2:
                        status_text = "UPDATED"
                        fg_color = "green"
                    elif diff_days <= 3:
                        status_text = "Need UPDATE"
                        fg_color = "red"
                    else:
                        status_text = "Need UPDATE"
                        fg_color = "red"
                self.status_label.config(text=status_text, fg=fg_color)
            except Exception as e:
                self.status_label.config(text="Status Unknown", fg="black")
        else:
            self.status_label.config(text="No Data", fg="black")

    def blink_status(self, text, color):
        if not self.blinking:
            self.status_label.config(text=text, fg=color)
            return
        current_fg = self.status_label.cget("fg")
        new_color = color if current_fg == "white" else "white"
        self.status_label.config(text=text, fg=new_color)
        self.blink_job = self.root.after(500, self.blink_status, text, color)

    def search_by_date(self):
        search_date_raw = self.search_date_entry.get()  # e.g., "01-02-25"
        try:
            search_date_obj = datetime.strptime(search_date_raw, "%d-%m-%y")
            search_date_formatted = search_date_obj.strftime(FILE_DATE_FORMAT)
        except Exception as e:
            messagebox.showerror("Search Error", "Invalid date selected.")
            return
        filtered_records = [row for row in self.data if row.get("Draw Date", "").strip() == search_date_formatted]
        if not filtered_records:
            messagebox.showinfo("Search Result", f"No records found for {search_date_formatted}.")
        self.refresh_treeview(filtered_records)

    def reset_search(self):
        self.refresh_treeview()

    def go_to_last_record(self):
        children = self.tree.get_children()
        if children:
            last_item = children[0]
            self.tree.selection_set(last_item)
            self.tree.focus(last_item)
            self.tree.see(last_item)

    def clear_form(self):
        for col in FILE_COLUMNS:
            self.entry_vars[col].set("")
        self.auto_fill_new_record()
        self.validation_error_label.config(text="")
        self.editing_mode = False

    def on_tree_select(self, event):
        selected = self.tree.selection()
        if selected:
            values = self.tree.item(selected[0])["values"]
            for idx, col in enumerate(FILE_COLUMNS):
                self.entry_vars[col].set(values[idx])
            self.entries["DN"].config(state="disabled")
            self.editing_mode = True
            self.original_dn = self.entry_vars["DN"].get().strip()
            self.original_date = self.entry_vars["Draw Date"].get().strip()
        else:
            self.editing_mode = False
        self.validate_current_inputs()

    def format_lottery_numbers(self, record):
        for key in ["L1", "L2", "L3", "L4", "L5", "L6"]:
            try:
                record[key] = f"{int(record[key]):02d}"
            except ValueError:
                pass
        return record

    def validate_lottery_order(self, record):
        try:
            numbers = [int(record[key]) for key in ["L1", "L2", "L3", "L4", "L5", "L6"]]
        except ValueError:
            return "All lottery numbers (L1-L6) must be valid integers."
        for i in range(len(numbers) - 1):
            if numbers[i] >= numbers[i + 1]:
                return f"{['L1','L2','L3','L4','L5','L6'][i]} must be less than {['L1','L2','L3','L4','L5','L6'][i+1]}."
        return ""

    def add_record(self):
        errors = self.validate_current_inputs()
        if errors:
            messagebox.showerror("Validation Error", " | ".join(errors))
            return
        new_record = {col: self.entry_vars[col].get().strip() for col in FILE_COLUMNS}
        new_record = self.format_lottery_numbers(new_record)
        if self.data:
            try:
                max_dn = max(int(row["DN"]) for row in self.data if row.get("DN"))
            except ValueError:
                max_dn = 0
            new_record["DN"] = str(max_dn + 1)
        else:
            new_record["DN"] = "1"
        for row in self.data:
            if row.get("Draw Date") == new_record["Draw Date"]:
                messagebox.showerror("Error", f"Duplicate Draw Date found: {new_record['Draw Date']}")
                return
        for row in self.data:
            if row.get("DN") == new_record["DN"]:
                messagebox.showerror("Error", f"Duplicate DN found: {new_record['DN']}")
                return
        order_error = self.validate_lottery_order(new_record)
        if order_error:
            messagebox.showerror("Error", order_error)
            return
        if not messagebox.askyesno("Confirm Add", "Do you want to add this record and save changes to the file?"):
            return
        self.data.append(new_record)
        self.save_data()
        self.refresh_treeview()
        self.clear_form()
        messagebox.showinfo("Success", "Record added and changes saved to the file.")

    def update_record(self):
        errors = self.validate_current_inputs()
        if errors:
            messagebox.showerror("Validation Error", " | ".join(errors))
            return
        selected = self.tree.selection()
        if not selected:
            messagebox.showerror("Error", "No record selected for update.")
            return
        old_values = self.tree.item(selected[0])["values"]
        new_record = {col: self.entry_vars[col].get().strip() for col in FILE_COLUMNS}
        new_record = self.format_lottery_numbers(new_record)
        order_error = self.validate_lottery_order(new_record)
        if order_error:
            messagebox.showerror("Error", order_error)
            return
        for row in self.data:
            if self.editing_mode and self.original_date and row.get("Draw Date").strip() == self.original_date:
                continue
            if row.get("Draw Date") == new_record["Draw Date"]:
                messagebox.showerror("Error", f"Duplicate Draw Date found: {new_record['Draw Date']}")
                return
        for row in self.data:
            if self.editing_mode and self.original_dn and row.get("DN").strip() == self.original_dn:
                continue
            if row.get("DN") == new_record["DN"]:
                messagebox.showerror("Error", f"Duplicate DN found: {new_record['DN']}")
                return
        if not messagebox.askyesno("Confirm Update", "Do you want to update this record and save changes to the file?"):
            return
        updated = False
        for row in self.data:
            if str(row.get("DN")).strip() == str(old_values[0]).strip():
                row.update(new_record)
                updated = True
                break
        if updated:
            self.save_data()
            self.refresh_treeview()
            self.clear_form()
            messagebox.showinfo("Success", "Record updated and changes saved to the file.")
        else:
            messagebox.showerror("Error", "Record not found for update.")

    def delete_record(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showerror("Error", "No record selected for deletion.")
            return
        if not messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete the selected record?"):
            return
        values = self.tree.item(selected[0])["values"]
        dn_to_delete = values[0]
        self.data = [row for row in self.data if str(row.get("DN")).strip() != str(dn_to_delete).strip()]
        self.save_data()
        self.refresh_treeview()
        self.clear_form()
        messagebox.showinfo("Success", "Record deleted and changes saved to the file.")

    def start_pcso_scraper(self):
        t = threading.Thread(target=self.run_pcso_scraper, daemon=True)
        t.start()

    def run_pcso_scraper(self):
        try:
            subprocess.run(["python", "PCSO_Scraper.py"], check=True)
            messagebox.showinfo("PCSO Scraper", "PCSO Scraper completed successfully!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("PCSO Scraper Error", f"PCSO Scraper failed: {e}")

    def start_file_watch(self):
        try:
            current_mod = os.path.getmtime(self.txt_filename)
        except Exception:
            current_mod = None
        if self.last_mod_time is None:
            self.last_mod_time = current_mod
        elif current_mod and current_mod != self.last_mod_time:
            self.load_data()
            self.refresh_treeview()
        self.root.after(5000, self.start_file_watch)

if __name__ == "__main__":
    root = tk.Tk()
    app = LotteryApp(root)
    root.mainloop()
