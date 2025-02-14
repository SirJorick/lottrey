import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# Base directory for file saving
BASE_DIR = r"C:\Users\user\PycharmProjects\lottrey"

# Define file columns (the structure remains the same)
FILE_COLUMNS = ["DN", "Draw Date", "L1", "L2", "L3", "L4", "L5", "L6"]
DATE_FORMAT = "%d-%b-%y"  # e.g., 06-Jan-07

# Mode options (text filenames)
MODE_OPTIONS = ["42.txt", "45.txt", "49.txt", "55.txt", "58.txt", "EZ2.txt", "Swertres.txt", "4D.txt", "6D.txt"]


def get_file_filepath(filename):
    """Return the absolute file path by joining BASE_DIR and the filename."""
    return os.path.join(BASE_DIR, filename)


class LotteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lottery Data Viewer - CRUD Operations")
        self.root.geometry("1050x700")

        # Mode selection variable (for file selection)
        self.mode_var = tk.StringVar(value="42.txt")
        self.txt_filename = self.get_txt_filename()  # absolute path with .txt extension
        self.data = []  # List of dictionaries holding records
        self.editing_mode = False
        self.original_dn = None  # To store original DN when editing
        self.original_date = None  # To store original Draw Date when editing

        self.create_widgets()
        self.load_data()
        self.refresh_treeview()
        self.auto_fill_new_record()

    def get_txt_filename(self, mode_value=None):
        """Return the file's absolute path based on the mode selection."""
        if mode_value is None:
            mode_value = self.mode_var.get().strip() or "42.txt"
        return get_file_filepath(mode_value)

    def get_heading_text(self):
        """Return the heading text, e.g., 'PSCO-42' (based on file name)."""
        mode = self.mode_var.get().strip() or "42.txt"
        file_no = mode.replace(".txt", "")
        return f"PSCO-{file_no}"

    def load_data(self):
        """Load data from the selected .txt file using tab as the delimiter.
           If the file is not found, ask the user if a new file should be created."""
        self.data = []
        if not os.path.exists(self.txt_filename):
            response = messagebox.askyesno("File Not Found",
                                           f"{self.txt_filename} was not found.\nDo you want to create a new file?")
            if response:
                self.save_data()  # Create the file with header.
            return

        try:
            with open(self.txt_filename, "r", encoding="utf-8") as file:
                lines = file.readlines()
                # Remove newline characters and ignore empty lines
                lines = [line.rstrip("\n") for line in lines if line.strip()]
                if not lines:
                    self.data = []
                    return
                # First line is header; subsequent lines are records.
                header = lines[0].split("\t")
                if header != FILE_COLUMNS:
                    messagebox.showwarning("Warning",
                                           "File header does not match expected columns. Data may be corrupted.")
                for line in lines[1:]:
                    row = line.split("\t")
                    if len(row) < len(FILE_COLUMNS):
                        continue  # skip incomplete lines
                    row_dict = {col: row[idx] for idx, col in enumerate(FILE_COLUMNS)}
                    self.data.append(row_dict)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def save_data(self):
        """Save the current data back to the .txt file using tab delimiters."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.txt_filename), exist_ok=True)
            with open(self.txt_filename, "w", encoding="utf-8") as file:
                # Write the header
                file.write("\t".join(FILE_COLUMNS) + "\n")
                # Write each record as a tab-delimited line
                for row in self.data:
                    line = "\t".join([row.get(col, "") for col in FILE_COLUMNS])
                    file.write(line + "\n")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data: {e}")

    def create_widgets(self):
        # ----------------- Header Frame -----------------
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill="x")

        self.heading_label = ttk.Label(header_frame,
                                       text=self.get_heading_text(),
                                       font=("Arial", 24, "bold"),
                                       foreground="blue")
        self.heading_label.pack(side="top", pady=10)

        # Sub-header for file selection and Last Record button
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

        # ----------------- Treeview Frame -----------------
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

        # ----------------- Record Details Frame -----------------
        form_frame = ttk.LabelFrame(self.root, text="Record Details", padding=10)
        form_frame.pack(fill="x", padx=10, pady=10)

        self.entries = {}  # Widgets for each record field
        self.entry_vars = {}  # Associated StringVars for each field
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

        # ----------------- Button Frame -----------------
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

    def on_mode_change(self, event):
        """When the mode (file) is changed, update the file path, reload data, and refresh the view."""
        self.txt_filename = self.get_txt_filename()
        self.heading_label.config(text=self.get_heading_text())
        self.load_data()
        self.refresh_treeview()
        self.clear_form()

    def format_var(self, col):
        """Format the lottery number entries to two digits."""
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
        """
        Validate that all fields are filled, numbers are valid and in ascending order,
        and no duplicate DN or Draw Date exists.
        During editing, the duplicate check ignores the current record being updated.
        """
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
                    errors.append(
                        f"{['L1', 'L2', 'L3', 'L4', 'L5', 'L6'][i]} must be less than {['L1', 'L2', 'L3', 'L4', 'L5', 'L6'][i + 1]}.")
                    break

        current_dn = self.entry_vars["DN"].get().strip()
        current_date = self.entry_vars["Draw Date"].get().strip()

        # Check for duplicate DN in self.data.
        for row in self.data:
            # When editing, ignore the current record being updated.
            if self.editing_mode and self.original_dn and row.get("DN").strip() == self.original_dn:
                continue
            if current_dn and row.get("DN").strip() == current_dn:
                errors.append("Duplicate DN found.")
                break

        # Check for duplicate Draw Date in self.data.
        for row in self.data:
            if self.editing_mode and self.original_date and row.get("Draw Date").strip() == self.original_date:
                continue
            if current_date and row.get("Draw Date").strip() == current_date:
                errors.append("Duplicate Draw Date found.")
                break

        self.validation_error_label.config(text=" | ".join(errors))
        return errors

    def auto_fill_new_record(self):
        """Automatically fill the form with the next DN and today's date."""
        if self.data:
            try:
                max_dn = max(int(row["DN"]) for row in self.data if row.get("DN"))
            except ValueError:
                max_dn = 0
            next_dn = max_dn + 1
            self.entry_vars["DN"].set(str(next_dn))
        else:
            self.entry_vars["DN"].set("1")
        today_str = datetime.today().strftime(DATE_FORMAT)
        self.entry_vars["Draw Date"].set(today_str)
        self.editing_mode = False
        self.original_dn = None
        self.original_date = None

    def refresh_treeview(self):
        """Clear and repopulate the treeview with data from self.data.
           Lottery number fields (L1–L6) are formatted as two-digit values."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        valid_data = [row for row in self.data if row.get("DN")]
        try:
            sorted_data = sorted(valid_data, key=lambda x: int(x["DN"]), reverse=True)
        except ValueError:
            sorted_data = sorted(valid_data, key=lambda x: x["DN"], reverse=True)
        for row in sorted_data:
            # Create a copy so we do not modify the original data
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
            last_record = sorted_data[0]
            text = f"Last Record: DN: {last_record['DN']}, Draw Date: {last_record['Draw Date']}"
            self.last_record_button.config(text=text)
        else:
            self.last_record_button.config(text="No record available")

    def go_to_last_record(self):
        """Select and focus the last record in the treeview."""
        children = self.tree.get_children()
        if children:
            last_item = children[0]
            self.tree.selection_set(last_item)
            self.tree.focus(last_item)
            self.tree.see(last_item)

    def clear_form(self):
        """Clear the input fields and prepare for a new record."""
        for col in FILE_COLUMNS:
            self.entry_vars[col].set("")
        self.auto_fill_new_record()
        self.validation_error_label.config(text="")
        self.editing_mode = False

    def on_tree_select(self, event):
        """When a record is selected in the treeview, populate the form for editing."""
        selected = self.tree.selection()
        if selected:
            values = self.tree.item(selected[0])["values"]
            for idx, col in enumerate(FILE_COLUMNS):
                self.entry_vars[col].set(values[idx])
            self.entries["DN"].config(state="disabled")
            self.editing_mode = True
            # Store original values to exclude them from duplicate checks during editing.
            self.original_dn = self.entry_vars["DN"].get().strip()
            self.original_date = self.entry_vars["Draw Date"].get().strip()
        else:
            self.editing_mode = False
        self.validate_current_inputs()

    def format_lottery_numbers(self, record):
        """Format lottery number values to two digits."""
        for key in ["L1", "L2", "L3", "L4", "L5", "L6"]:
            try:
                record[key] = f"{int(record[key]):02d}"
            except ValueError:
                pass
        return record

    def validate_lottery_order(self, record):
        """Ensure lottery numbers are in ascending order."""
        try:
            numbers = [int(record[key]) for key in ["L1", "L2", "L3", "L4", "L5", "L6"]]
        except ValueError:
            return "All lottery numbers (L1-L6) must be valid integers."
        for i in range(len(numbers) - 1):
            if numbers[i] >= numbers[i + 1]:
                return f"{['L1', 'L2', 'L3', 'L4', 'L5', 'L6'][i]} must be less than {['L1', 'L2', 'L3', 'L4', 'L5', 'L6'][i + 1]}."
        return ""

    def add_record(self):
        """Add a new record after validation."""
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
        if self.data:
            try:
                last_date = max(
                    datetime.strptime(row["Draw Date"], DATE_FORMAT) for row in self.data if row.get("Draw Date"))
                new_date = datetime.strptime(new_record["Draw Date"], DATE_FORMAT)
                if new_date <= last_date:
                    messagebox.showerror("Error", f"Draw Date must be after {last_date.strftime(DATE_FORMAT)}")
                    return
            except Exception:
                messagebox.showerror("Error", "Invalid Draw Date format. Use DD-MMM-YY (e.g., 06-Jan-07).")
                return
        for row in self.data:
            if row.get("Draw Date") == new_record["Draw Date"]:
                messagebox.showerror("Error", f"Duplicate Draw Date found: {new_record['Draw Date']}")
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
        """Update the selected record after validation."""
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
        """Delete the selected record and update the file."""
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


if __name__ == "__main__":
    root = tk.Tk()
    app = LotteryApp(root)
    root.mainloop()
