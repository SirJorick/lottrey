import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from webdriver_manager.chrome import ChromeDriverManager
import threading
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import webbrowser

# Set HEADLESS to False so the browser is visible (minimized) to help bypass 403 errors.
HEADLESS = False


class LottoFetcher:
    def __init__(self, master):
        self.master = master
        master.title("Lottery Results Fetcher")

        # Lottery options and mapping.
        self.lottery_options = ["6/42", "6/45", "6/49", "6/55", "6/58", "EZ2", "Swertres", "4D", "6D"]
        self.game_map = {
            "6/42": "Lotto 6/42",
            "6/45": "Mega Lotto 6/45",
            "6/49": "Super Lotto 6/49",
            "6/55": "Grand Lotto 6/55",
            "6/58": "Ultra Lotto 6/58",
            "EZ2": "EZ2",
            "Swertres": "Swertres",
            "4D": "4D Lotto",
            "6D": "6D Lotto"
        }

        # Estimated total pages for progress calculation.
        self.estimated_total_pages = 10
        self.pages_processed = 0
        self.pages_processed_last_update = 0
        self.total_bytes = 0  # Total downloaded bytes.

        # For continuous ETA update (ETA removed from display).
        self.start_time = None

        # For dynamic dot animation.
        self.downloading = False
        self.animate_counter = 0
        self.latest_progress_info = ""

        # --- UI Widgets ---
        ttk.Label(master, text="Select Lottery Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lottery_var = tk.StringVar(value=self.lottery_options[0])
        self.lottery_combo = ttk.Combobox(master, textvariable=self.lottery_var,
                                          values=self.lottery_options, state="readonly")
        self.lottery_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(master, text="From Date:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.from_date_entry = DateEntry(master, date_pattern='dd/mm/yyyy')
        self.from_date_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(master, text="Until Date:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.until_date_entry = DateEntry(master, date_pattern='dd/mm/yyyy')
        self.until_date_entry.grid(row=2, column=1, padx=5, pady=5)

        self.fetch_button = ttk.Button(master, text="Fetch Results", command=self.start_fetch_thread)
        self.fetch_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

        # Fixed progress info frame.
        self.progress_frame = ttk.Frame(master)
        self.progress_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.progress_info_label = ttk.Label(self.progress_frame, text="Progress: 0%")
        self.progress_info_label.grid(row=0, column=0, sticky="w")
        self.progress_dots_label = ttk.Label(self.progress_frame, text="")
        self.progress_dots_label.grid(row=0, column=1, sticky="w")

        ttk.Label(master, text="Status Updates:").grid(row=6, column=0, padx=5, pady=(10, 0), sticky="w")
        self.status_text = tk.Text(master, width=80, height=10)
        self.status_text.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

        # Treeview for final PCSO results.
        self.results_frame = ttk.Frame(master)
        self.results_frame.grid(row=8, column=0, columnspan=2, padx=5, pady=(10, 0), sticky="nsew")
        self.results_label = ttk.Label(self.results_frame, text="Final Results:")
        self.results_label.grid(row=0, column=0, sticky="w")
        self.draw_count_label = ttk.Label(self.results_frame, text="0 Draws Fetched")
        self.draw_count_label.grid(row=0, column=1, sticky="w", padx=10)
        self.results_tree = ttk.Treeview(master, columns=("game", "combination", "draw_date", "jackpot", "winners"),
                                         show="headings")
        self.results_tree.heading("game", text="LOTTO GAME")
        self.results_tree.heading("combination", text="COMBINATIONS")
        self.results_tree.heading("draw_date", text="DRAW DATE")
        self.results_tree.heading("jackpot", text="JACKPOT (PHP)")
        self.results_tree.heading("winners", text="WINNERS")
        self.results_tree.column("game", width=100)
        self.results_tree.column("combination", width=150)
        self.results_tree.column("draw_date", width=100)
        self.results_tree.column("jackpot", width=120)
        self.results_tree.column("winners", width=80)
        self.results_tree.grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.tree_scrollbar = ttk.Scrollbar(master, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscroll=self.tree_scrollbar.set)
        self.tree_scrollbar.grid(row=9, column=2, sticky="ns", padx=(0, 5), pady=5)

    def log_status(self, message):
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)

    def update_progress_info(self):
        # Calculate percentage.
        percentage = min(100, (self.pages_processed / self.estimated_total_pages) * 100)
        # Calculate average page size.
        avg_page_size = self.total_bytes / self.pages_processed if self.pages_processed > 0 else 0
        estimated_total_bytes = self.estimated_total_pages * avg_page_size
        remaining_bytes = estimated_total_bytes - self.total_bytes
        estimated_total_MB = estimated_total_bytes / (1024 * 1024)
        remaining_MB = remaining_bytes / (1024 * 1024)
        self.latest_progress_info = (f"Estimated Total: {estimated_total_MB:.2f} MB, Remaining: {remaining_MB:.2f} MB, "
                                     f"{percentage:.0f}%")
        self.progress_info_label.config(text=f"Progress: {self.latest_progress_info}")

    def animate_progress(self):
        if self.downloading:
            dots = "." * (self.animate_counter % 7)
            self.animate_counter += 1
            self.progress_dots_label.config(text=dots)
            self.master.after(500, self.animate_progress)

    def start_fetch_thread(self):
        self.pages_processed = 0
        self.total_bytes = 0
        self.pages_processed_last_update = 0
        self.progress_info_label.config(text="Progress: 0%")
        self.progress_dots_label.config(text="")
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.draw_count_label.config(text="0 Draws Fetched")
        self.status_text.delete("1.0", tk.END)
        self.downloading = True
        self.animate_counter = 0
        self.start_time = time.time()
        self.master.after(500, self.animate_progress)
        thread = threading.Thread(target=self.fetch_results)
        thread.daemon = True
        thread.start()

    def fetch_results(self):
        lottery_type = self.lottery_var.get().strip()  # e.g., "6/42"
        from_date_str = self.from_date_entry.get().strip()  # dd/mm/yyyy
        until_date_str = self.until_date_entry.get().strip()

        if not from_date_str or not until_date_str:
            self.log_status("Input Error: Please provide both From and Until dates.")
            return

        try:
            from_date_obj = datetime.strptime(from_date_str, "%d/%m/%Y")
            until_date_obj = datetime.strptime(until_date_str, "%d/%m/%Y")
        except Exception as e:
            self.log_status(f"Date format error: {e}")
            return

        # Adjust starting date: subtract 1 day.
        effective_from_date = from_date_obj - relativedelta(days=1)
        self.log_status(f"Effective starting date for search: {effective_from_date.strftime('%d/%m/%Y')}")

        try:
            options = Options()
            if HEADLESS:
                options.headless = True
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.5195.102 Safari/537.36")
            options.add_argument("referer=https://www.pcso.gov.ph/")
            options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 2})

            # Uncomment to use a proxy if necessary.
            # options.add_argument("--proxy-server=http://your-proxy-server:port")

            service = ChromeDriverManager().install()
            driver = webdriver.Chrome(service=ChromeService(executable_path=service), options=options)

            # Hide the PCSO webpage.
            driver.minimize_window()

            stealth(driver,
                    languages=["en-US", "en"],
                    vendor="Google Inc.",
                    platform="Win32",
                    webgl_vendor="Intel Inc.",
                    renderer="Intel Iris OpenGL Engine",
                    fix_hairline=True)

            wait = WebDriverWait(driver, 30)

            self.log_status("Loading PCSO Lotto Results page...")
            driver.get("https://www.pcso.gov.ph/SearchLottoResult.aspx")
            time.sleep(2)
            self.pages_processed += 1
            self.total_bytes += len(driver.page_source.encode("utf-8"))
            self.master.after(0, self.update_progress_info)

            self.log_status("Setting date range via drop-downs...")
            try:
                start_month_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlStartMonth']")
                Select(start_month_dd).select_by_visible_text(effective_from_date.strftime("%B"))
            except Exception:
                self.log_status("Start month dropdown not found.")
            try:
                start_day_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlStartDay']")
                Select(start_day_dd).select_by_visible_text(str(effective_from_date.day))
            except Exception:
                self.log_status("Start day dropdown not found; skipping day selection.")
            try:
                start_year_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlStartYear']")
                Select(start_year_dd).select_by_visible_text(str(effective_from_date.year))
            except Exception:
                self.log_status("Start year dropdown not found.")
            try:
                end_month_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlEndMonth']")
                Select(end_month_dd).select_by_visible_text(until_date_obj.strftime("%B"))
            except Exception:
                self.log_status("End month dropdown not found.")
            try:
                end_day_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlEndDay']")
                Select(end_day_dd).select_by_visible_text(str(until_date_obj.day))
            except Exception:
                self.log_status("End day dropdown not found; skipping day selection.")
            try:
                end_year_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlEndYear']")
                Select(end_year_dd).select_by_visible_text(str(until_date_obj.year))
            except Exception:
                self.log_status("End year dropdown not found.")
            self.log_status("Drop-downs set.")
            self.pages_processed += 1
            self.total_bytes += len(driver.page_source.encode("utf-8"))
            self.master.after(0, self.update_progress_info)

            self.log_status("Triggering search...")
            search_button = driver.find_element(By.CSS_SELECTOR, "input[id*='btnSearch'], button[id*='btnSearch']")
            search_button.click()
            time.sleep(3)
            self.pages_processed += 1
            self.total_bytes += len(driver.page_source.encode("utf-8"))
            self.master.after(0, self.update_progress_info)

            self.log_status("Waiting for results...")
            results_table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table[id*='GridView']")))
            self.pages_processed += 1
            self.total_bytes += len(driver.page_source.encode("utf-8"))
            self.master.after(0, self.update_progress_info)

            self.log_status("Scraping results...")
            draws = []
            expected_game = self.game_map.get(lottery_type, "").strip().lower()
            date_formats = ["%m/%d/%Y", "%d/%m/%Y", "%m/%d/%Y %I:%M %p", "%d/%m/%Y %I:%M %p"]
            while True:
                rows = results_table.find_elements(By.TAG_NAME, "tr")
                for row in rows[1:]:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 5:
                        game_name = cells[0].text.strip().lower()
                        combination = cells[1].text.strip()
                        draw_date = cells[2].text.strip()
                        jackpot = cells[3].text.strip()
                        winners = cells[4].text.strip()
                        draw_date_obj = None
                        for fmt in date_formats:
                            try:
                                draw_date_obj = datetime.strptime(draw_date, fmt)
                                break
                            except Exception:
                                continue
                        if draw_date_obj is None:
                            self.log_status(f"Skipping row; unable to parse date: {draw_date}")
                            continue
                        if not (effective_from_date <= draw_date_obj <= until_date_obj):
                            continue
                        if expected_game and expected_game not in game_name:
                            continue
                        parts = re.split(r'[-\s]+', combination)
                        try:
                            sorted_parts = sorted(parts, key=lambda x: int(x))
                            sorted_combination = "-".join(sorted_parts)
                        except Exception:
                            sorted_combination = combination
                        draws.append((cells[0].text.strip(), sorted_combination, draw_date, jackpot, winners))
                self.pages_processed += 1
                self.total_bytes += len(driver.page_source.encode("utf-8"))
                self.master.after(0, self.update_progress_info)
                try:
                    next_buttons = driver.find_elements(By.LINK_TEXT, "Next")
                    if next_buttons:
                        next_button = next_buttons[0]
                        if not next_button.get_attribute("href"):
                            self.log_status("No more pages found.")
                            break
                        else:
                            self.log_status("Navigating to the next page...")
                            driver.execute_script("arguments[0].click();", next_button)
                            time.sleep(3)
                            results_table = wait.until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "table[id*='GridView']")))
                    else:
                        self.log_status("Next button not found; assuming last page.")
                        break
                except Exception:
                    self.log_status("Error navigating to the next page; ending pagination.")
                    break

            # Force final progress to 100%
            self.pages_processed = self.estimated_total_pages
            self.master.after(0, self.update_progress_info)

            self.log_status("Final processing...")
            if len(draws) == 0:
                self.master.title("Lottery Results Fetcher - 0 Draws Fetched")
                draws = [("No results found for the selected lottery type in the chosen date range", "", "", "", "")]
            else:
                num_draws = len(draws)
                self.master.title(f"Lottery Results Fetcher - {num_draws} Draws Fetched")
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            for row in draws:
                self.results_tree.insert("", tk.END, values=row)
            self.draw_count_label.config(text=f"{len(draws)} Draws Fetched")

            # Mark download as complete.
            self.progress_info_label.config(text="Download Complete...")
            self.progress_dots_label.config(text="")
            self.log_status("Fetching from PCSO completed successfully.")
            self.downloading = False
        except Exception as e:
            self.log_status(f"Error fetching results: {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = LottoFetcher(root)
    root.mainloop()
