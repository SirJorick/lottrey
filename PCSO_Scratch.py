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

# Set HEADLESS to False to help bypass 403 errors.
HEADLESS = False


# HyperlinkManager helps create clickable hyperlinks in the Text widget.
class HyperlinkManager:
    def __init__(self, text):
        self.text = text
        self.text.tag_config("hyper", foreground="blue", underline=1)
        self.text.tag_bind("hyper", "<Enter>", lambda e: self.text.config(cursor="hand2"))
        self.text.tag_bind("hyper", "<Leave>", lambda e: self.text.config(cursor=""))
        self.links = {}

    def add(self, action):
        tag = "hyper-%d" % len(self.links)
        self.links[tag] = action
        return ("hyper", tag)

    def click(self, event):
        for tag in self.text.tag_names("current"):
            if tag.startswith("hyper-"):
                self.links[tag]()
                return "break"


class LottoFetcher:
    def __init__(self, master):
        self.master = master
        master.title("Lottery Results Fetcher")

        # Updated lottery options (display values)
        self.lottery_options = ["6/42", "6/45", "6/49", "6/55", "6/58", "EZ2", "Swertres", "4D", "6D"]

        # Mapping from combobox value to expected text in the PCSO "LOTTO GAME" column.
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

        # Option: Use Bing News instead of PCSO website.
        self.search_entire = tk.BooleanVar(value=False)
        self.search_entire_cb = ttk.Checkbutton(master, text="Bing News", variable=self.search_entire)
        self.search_entire_cb.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.fetch_button = ttk.Button(master, text="Fetch Results", command=self.start_fetch_thread)
        self.fetch_button.grid(row=4, column=0, columnspan=2, padx=5, pady=10)

        # Progress bar and label.
        self.progress_bar = ttk.Progressbar(master, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        self.progress_label = ttk.Label(master, text="Progress: 0% - ETA: N/A")
        self.progress_label.grid(row=6, column=0, columnspan=2)

        # Status text box.
        ttk.Label(master, text="Status Updates:").grid(row=7, column=0, padx=5, pady=(10, 0), sticky="w")
        self.status_text = tk.Text(master, width=80, height=10)
        self.status_text.grid(row=8, column=0, columnspan=2, padx=5, pady=5)

        # Final results text box.
        ttk.Label(master, text="Final Results:").grid(row=9, column=0, padx=5, pady=(10, 0), sticky="w")
        self.results_text = tk.Text(master, width=80, height=10)
        self.results_text.grid(row=10, column=0, columnspan=2, padx=5, pady=5)
        self.results_text.tag_config("hyper", foreground="blue", underline=1)
        self.results_text.bind("<Button-1>", lambda e: self.hyperlink_click(e))
        self.hyperlink_manager = HyperlinkManager(self.results_text)

        self.total_steps = 8

    def hyperlink_click(self, event):
        return self.hyperlink_manager.click(event)

    def log_status(self, message):
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)

    def update_progress(self, step, start_time):
        percent = int((step / self.total_steps) * 100)
        elapsed = time.time() - start_time
        remaining_steps = self.total_steps - step
        avg_time = elapsed / step if step > 0 else 0
        eta_seconds = int(avg_time * remaining_steps) if step > 0 else 0
        minutes, seconds = divmod(eta_seconds, 60)
        eta_str = f"{minutes}m {seconds}s" if eta_seconds > 0 else "N/A"
        self.master.after(0, lambda: self.progress_bar.configure(value=percent))
        self.master.after(0, lambda: self.progress_label.configure(text=f"Progress: {percent}% - ETA: {eta_str}"))

    def start_fetch_thread(self):
        thread = threading.Thread(target=self.fetch_results)
        thread.daemon = True
        thread.start()

    def fetch_results(self):
        lottery_type = self.lottery_var.get().strip()  # e.g., "6/42"
        from_date_str = self.from_date_entry.get().strip()  # dd/mm/yyyy
        until_date_str = self.until_date_entry.get().strip()

        if not from_date_str or not until_date_str:
            self.master.after(0,
                              lambda: messagebox.showerror("Input Error", "Please provide both From and Until dates."))
            return

        try:
            from_date_obj = datetime.strptime(from_date_str, "%d/%m/%Y")
            until_date_obj = datetime.strptime(until_date_str, "%d/%m/%Y")
        except Exception as e:
            err = f"Date format error: {e}"
            self.master.after(0, lambda: messagebox.showerror("Date Error", err))
            return

        # Adjust starting date: subtract 1 day from the user-input from_date.
        effective_from_date = from_date_obj - relativedelta(days=1)
        self.log_status(f"Effective starting date for search: {effective_from_date.strftime('%d/%m/%Y')}")

        start_time = time.time()
        current_step = 0
        self.master.after(0, lambda: self.progress_bar.configure(value=0))
        self.master.after(0, lambda: self.progress_label.configure(text="Progress: 0% - ETA: N/A"))
        self.master.after(0, lambda: self.status_text.delete("1.0", tk.END))
        self.master.after(0, lambda: self.results_text.delete("1.0", tk.END))

        try:
            options = Options()
            if HEADLESS:
                options.headless = True
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-extensions")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) " \
                                 "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.5195.102 Safari/537.36")
            options.add_argument("referer=https://www.pcso.gov.ph/")
            options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 2})

            service = ChromeService(executable_path=ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)

            stealth(driver,
                    languages=["en-US", "en"],
                    vendor="Google Inc.",
                    platform="Win32",
                    webgl_vendor="Intel Inc.",
                    renderer="Intel Iris OpenGL Engine",
                    fix_hairline=True)

            wait = WebDriverWait(driver, 30)
            current_step += 1
            self.update_progress(current_step, start_time)

            if self.search_entire.get():
                self.log_status("Bing News branch enabled. Performing Bing search...")
                query = (
                    f"{lottery_type} lottery news {effective_from_date.strftime('%d/%m/%Y')} to {until_date_obj.strftime('%d/%m/%Y')}")
                driver.get("https://www.bing.com")
                time.sleep(2)
                search_box = wait.until(EC.presence_of_element_located((By.ID, "sb_form_q")))
                search_box.clear()
                search_box.send_keys(query)
                search_box.submit()
                time.sleep(3)
                results = driver.find_elements(By.CSS_SELECTOR, "li.b_algo")
                self.results_text.delete("1.0", tk.END)
                for result in results:
                    try:
                        title_elem = result.find_element(By.TAG_NAME, "h2")
                        link_elem = title_elem.find_element(By.TAG_NAME, "a")
                        title = title_elem.text.strip()
                        link = link_elem.get_attribute("href")
                        if title:
                            self.results_text.insert(tk.END, title,
                                                     self.hyperlink_manager.add(lambda url=link: webbrowser.open(url)))
                            self.results_text.insert(tk.END, "\n\n")
                    except Exception:
                        continue
                self.log_status("Bing News search completed.")
                num_headlines = len(
                    [line for line in self.results_text.get("1.0", tk.END).splitlines() if line.strip()])
                self.master.title(f"Lottery Results Fetcher - {num_headlines} Headlines Fetched")
                driver.quit()
                return

            self.log_status("Step 2: Loading PCSO Lotto Results page...")
            driver.get("https://www.pcso.gov.ph/SearchLottoResult.aspx")
            time.sleep(2)
            current_step += 1
            self.update_progress(current_step, start_time)

            self.log_status("Step 3: Lottery dropdown not available; skipping lottery type selection.")
            current_step += 1
            self.update_progress(current_step, start_time)

            self.log_status("Step 4: Setting date range via drop-downs...")
            try:
                start_month_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlStartMonth']")
                Select(start_month_dd).select_by_visible_text(effective_from_date.strftime("%B"))
                self.log_status("Start month selected.")
            except Exception:
                self.log_status("Start month dropdown not found.")
            try:
                start_day_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlStartDay']")
                Select(start_day_dd).select_by_visible_text(str(effective_from_date.day))
                self.log_status("Start day selected.")
            except Exception:
                self.log_status("Start day dropdown not found; skipping day selection.")
            try:
                start_year_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlStartYear']")
                Select(start_year_dd).select_by_visible_text(str(effective_from_date.year))
                self.log_status("Start year selected.")
            except Exception:
                self.log_status("Start year dropdown not found.")
            try:
                end_month_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlEndMonth']")
                Select(end_month_dd).select_by_visible_text(until_date_obj.strftime("%B"))
                self.log_status("End month selected.")
            except Exception:
                self.log_status("End month dropdown not found.")
            try:
                end_day_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlEndDay']")
                Select(end_day_dd).select_by_visible_text(str(until_date_obj.day))
                self.log_status("End day selected.")
            except Exception:
                self.log_status("End day dropdown not found; skipping day selection.")
            try:
                end_year_dd = driver.find_element(By.CSS_SELECTOR, "select[id*='ddlEndYear']")
                Select(end_year_dd).select_by_visible_text(str(until_date_obj.year))
                self.log_status("End year selected.")
            except Exception:
                self.log_status("End year dropdown not found.")
            self.log_status("Drop-downs for date range attempted.")
            current_step += 1
            self.update_progress(current_step, start_time)

            self.log_status("Step 5: Triggering search...")
            search_button = driver.find_element(By.CSS_SELECTOR, "input[id*='btnSearch'], button[id*='btnSearch']")
            search_button.click()
            current_step += 1
            self.update_progress(current_step, start_time)

            self.log_status("Step 6: Waiting for results...")
            try:
                results_table = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table[id*='GridView']"))
                )
            except Exception as e:
                snippet = driver.page_source[:1000]
                raise Exception("Results table not found. Page snippet: " + snippet)
            current_step += 1
            self.update_progress(current_step, start_time)

            self.log_status("Step 7: Scraping results and filtering by date and lottery type...")
            extracted_results = "LOTTO GAME\tCOMBINATIONS\tDRAW DATE\tJACKPOT (PHP)\tWINNERS\n"
            expected_game = self.game_map.get(lottery_type, "").strip().lower()
            date_formats = ["%m/%d/%Y", "%d/%m/%Y", "%m/%d/%Y %I:%M %p", "%d/%m/%Y %I:%M %p"]
            page = 1
            while True:
                self.log_status(f"Scraping page {page}...")
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
                        extracted_results += f"{cells[0].text.strip()}\t{combination}\t{draw_date}\t{jackpot}\t{winners}\n"
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
                                EC.presence_of_element_located((By.CSS_SELECTOR, "table[id*='GridView']"))
                            )
                            page += 1
                    else:
                        self.log_status("Next button not found; assuming this is the last page.")
                        break
                except Exception:
                    self.log_status("Error navigating to the next page; ending pagination.")
                    break

            current_step += 1
            self.update_progress(current_step, start_time)

            self.log_status("Step 8: Final processing...")
            current_step += 1
            self.update_progress(current_step, start_time)

            if extracted_results.strip() == "LOTTO GAME\tCOMBINATIONS\tDRAW DATE\tJACKPOT (PHP)\tWINNERS":
                extracted_results = "No results found for the selected lottery type in the chosen date range."
            else:
                num_draws = len([line for line in extracted_results.splitlines() if line.strip()]) - 1
                self.master.title(f"Lottery Results Fetcher - {num_draws} Draws Fetched")

            self.master.after(0, lambda: self.results_text.insert(tk.END, extracted_results))
            self.log_status("Fetching from PCSO completed successfully.")
        except Exception as e:
            error_msg = f"Error fetching results: {e}"
            self.master.after(0, lambda: messagebox.showerror("Fetch Error", error_msg))
            self.log_status(error_msg)
        finally:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = LottoFetcher(root)
    root.mainloop()
