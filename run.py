"""
run.py — Master launcher for the VRP QAOA project.

Modern GUI-based launcher with instance and vehicle selection.
Dynamically discovers scripts in scripts/ folder.
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
INSTANCES_DIR = os.path.join(ROOT, "Instances")

# Platform-aware fonts
if sys.platform == "darwin":
    FONT_UI   = "Helvetica Neue"
    FONT_MONO = "Menlo"
elif sys.platform == "win32":
    FONT_UI   = "Segoe UI"
    FONT_MONO = "Cascadia Code"
else:
    FONT_UI   = "DejaVu Sans"
    FONT_MONO = "DejaVu Sans Mono"

C = {
    "bg":           "#f0f4f8",
    "card":         "#ffffff",
    "card_border":  "#d1d9e6",
    "header":       "#1e1b4b",
    "header_text":  "#ffffff",
    "header_muted": "#c7d2fe",
    "header_stripe":"#4f46e5",
    "text":         "#111827",
    "text_muted":   "#6b7280",
    "accent":       "#4f46e5",
    "accent_dark":  "#3730a3",
    "accent_fade":  "#eef2ff",
    "red":          "#dc2626",
    "red_dark":     "#b91c1c",
    "green":        "#059669",
    "secondary":    "#f1f5f9",
    "secondary_h":  "#e2e8f0",
    "sep":          "#e5e7eb",
    "term_bg":      "#0d1117",
    "term_bar":     "#161b22",
    "term_text":    "#c9d1d9",
    "term_green":   "#56d364",
    "term_red":     "#f85149",
    "term_yellow":  "#e3b341",
    "term_blue":    "#79c0ff",
    "term_dim":     "#484f58",
    "term_border":  "#21262d",
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def discover_scripts():
    scripts = []
    if os.path.isdir(SCRIPTS_DIR):
        for file in sorted(os.listdir(SCRIPTS_DIR)):
            if file.endswith(".py") and not file.startswith("__"):
                scripts.append({"name": file[:-3], "file": os.path.join("scripts", file)})
    return scripts


def discover_instances():
    instances = []
    if os.path.isdir(INSTANCES_DIR):
        for root, dirs, files in os.walk(INSTANCES_DIR):
            for file in sorted(files):
                if file.endswith(".vrp"):
                    name = file[:-4]
                    value = name.replace("RioClaroPostToy_", "") if name.startswith("RioClaroPostToy_") else name
                    instances.append({"name": name, "value": value})
    return instances


def sep(parent, vertical=False, **kwargs):
    """Create a separator line. Can override bg with kwargs."""
    bg = kwargs.pop("bg", C["sep"])  # Extract bg if provided, otherwise use default
    if vertical:
        return tk.Frame(parent, width=1, bg=bg, **kwargs)
    return tk.Frame(parent, height=1, bg=bg, **kwargs)


# ─── Custom Styled Button ──────────────────────────────────────────────────────

class Btn(tk.Frame):
    _THEMES = {
        "primary":   dict(bg=C["accent"],    fg="#fff",          hov=C["accent_dark"],  dis_bg="#a5b4fc", dis_fg="#e0e7ff"),
        "danger":    dict(bg=C["red"],        fg="#fff",          hov=C["red_dark"],     dis_bg="#fca5a5", dis_fg="#fff1f2"),
        "secondary": dict(bg=C["secondary"], fg=C["text"],       hov=C["secondary_h"], dis_bg="#f9fafb", dis_fg="#9ca3af"),
        "ghost":     dict(bg=C["bg"],        fg=C["text_muted"], hov=C["secondary"],   dis_bg=C["bg"],   dis_fg="#d1d5db"),
    }

    def __init__(self, parent, text, command=None, variant="secondary", **kwargs):
        try:
            parent_bg = parent.cget("bg")
        except Exception:
            parent_bg = C["bg"]
        super().__init__(parent, bg=parent_bg)
        t = self._t = self._THEMES[variant]
        self._cmd  = command
        self._on   = True
        self._bold = variant == "primary"

        self._lbl = tk.Label(
            self, text=text, bg=t["bg"], fg=t["fg"],
            font=(FONT_UI, 10, "bold" if self._bold else "normal"),
            padx=18, pady=9, cursor="hand2",
        )
        self._lbl.pack(fill=tk.BOTH, expand=True)

        for w in (self, self._lbl):
            w.bind("<Enter>",    lambda e: self._hover(True))
            w.bind("<Leave>",    lambda e: self._hover(False))
            w.bind("<Button-1>", lambda e: self._click())

    def _hover(self, on):
        if self._on:
            self._lbl.config(bg=self._t["hov"] if on else self._t["bg"])

    def _click(self):
        if self._on and self._cmd:
            self._cmd()

    def config(self, state=None, text=None, **kw):
        if state in (tk.DISABLED, "disabled"):
            self._on = False
            self._lbl.config(bg=self._t["dis_bg"], fg=self._t["dis_fg"], cursor="arrow")
        elif state in (tk.NORMAL, "normal"):
            self._on = True
            self._lbl.config(bg=self._t["bg"], fg=self._t["fg"], cursor="hand2")
        if text is not None:
            self._lbl.config(text=text)

    configure = config


# ─── Main App ──────────────────────────────────────────────────────────────────

class VRPLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VRP QAOA Launcher")
        self.root.geometry("980x740")
        self.root.minsize(820, 600)
        self.root.configure(bg=C["bg"])

        if sys.platform == "win32":
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

        self.scripts  = discover_scripts()
        self.instances = discover_instances()
        self.current_process = None
        self.running = False

        self._setup_ttk()
        self._build_ui()

    # ── TTK Theme ──────────────────────────────────────────────────────────────

    def _setup_ttk(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox",
            fieldbackground=C["card"], background=C["card"],
            foreground=C["text"], selectbackground=C["accent"],
            selectforeground="#fff", arrowcolor=C["accent"],
            bordercolor=C["card_border"], lightcolor=C["card"],
            darkcolor=C["card_border"], padding=(8, 6),
        )
        s.configure("TSpinbox",
            fieldbackground=C["card"], background=C["card"],
            foreground=C["text"], arrowcolor=C["accent"],
            bordercolor=C["card_border"], lightcolor=C["card"],
            darkcolor=C["card_border"], padding=(8, 6),
        )
        s.configure("Vertical.TScrollbar",
            background=C["term_bar"], troughcolor=C["term_bg"],
            bordercolor=C["term_bg"], arrowcolor=C["term_dim"],
            gripcount=0,
        )
        s.map("Vertical.TScrollbar",
            background=[("active", C["term_dim"]), ("pressed", C["term_dim"])],
        )

    # ── UI Layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_statusbar()   # pack bottom first
        self._build_header()
        self._build_body()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["header"], height=76)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        # Left colour stripe
        tk.Frame(hdr, width=5, bg=C["header_stripe"]).pack(side=tk.LEFT, fill=tk.Y)

        inner = tk.Frame(hdr, bg=C["header"])
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=22, pady=0)

        tk.Label(
            inner, text="VRP QAOA Launcher",
            font=(FONT_UI, 18, "bold"),
            bg=C["header"], fg=C["header_text"],
        ).pack(anchor=tk.W, pady=(16, 2))

        tk.Label(
            inner,
            text=f"{len(self.scripts)} scripts  ·  {len(self.instances)} instances detected",
            font=(FONT_UI, 10),
            bg=C["header"], fg=C["header_muted"],
        ).pack(anchor=tk.W)

    def _build_body(self):
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=26, pady=20)

        self._build_config_card(body)
        self._build_button_row(body)
        self._build_terminal(body)

    # ── Config card ───────────────────────────────────────────────────────────

    def _build_config_card(self, parent):
        card = tk.Frame(parent, bg=C["card"],
                        highlightthickness=1, highlightbackground=C["card_border"])
        card.pack(fill=tk.X, pady=(0, 16))

        # Card header strip
        strip = tk.Frame(card, bg=C["accent_fade"])
        strip.pack(fill=tk.X)
        tk.Label(
            strip, text="Configuration",
            font=(FONT_UI, 11, "bold"),
            bg=C["accent_fade"], fg=C["accent"],
            padx=18, pady=9,
        ).pack(side=tk.LEFT)

        sep(card).pack(fill=tk.X)

        grid = tk.Frame(card, bg=C["card"])
        grid.pack(fill=tk.X, padx=18, pady=14)
        grid.columnconfigure(1, weight=1)

        def field_label(row, text):
            tk.Label(
                grid, text=text,
                font=(FONT_UI, 10),
                bg=C["card"], fg=C["text_muted"],
                anchor=tk.W, width=14,
            ).grid(row=row * 2, column=0, sticky=tk.W, padx=(0, 16), pady=(6, 6))

        # Script
        field_label(0, "Script")
        self.script_var = tk.StringVar()
        self.script_combo = ttk.Combobox(
            grid, textvariable=self.script_var,
            values=[s["name"] for s in self.scripts],
            state="readonly", font=(FONT_UI, 10),
        )
        self.script_combo.grid(row=0, column=1, sticky=tk.EW, pady=(6, 6))
        if self.scripts:
            self.script_combo.current(0)

        sep(grid).grid(row=1, column=0, columnspan=2, sticky=tk.EW)

        # Instance
        field_label(1, "Instance")
        self.instance_var = tk.StringVar()
        self.instance_combo = ttk.Combobox(
            grid, textvariable=self.instance_var,
            values=[i["name"] for i in self.instances],
            state="readonly", font=(FONT_UI, 10),
        )
        self.instance_combo.grid(row=2, column=1, sticky=tk.EW, pady=(6, 6))
        if self.instances:
            self.instance_combo.current(0)

        sep(grid).grid(row=3, column=0, columnspan=2, sticky=tk.EW)

        # Vehicles
        field_label(2, "Vehicles (k)")
        spinbox_wrap = tk.Frame(grid, bg=C["card"])
        spinbox_wrap.grid(row=4, column=1, sticky=tk.W, pady=(6, 6))
        self.k_var = tk.StringVar(value="7")
        self.k_spinbox = ttk.Spinbox(
            spinbox_wrap, from_=1, to=100, textvariable=self.k_var,
            width=9, font=(FONT_UI, 10),
        )
        self.k_spinbox.pack(side=tk.LEFT)

    # ── Button row ────────────────────────────────────────────────────────────

    def _build_button_row(self, parent):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill=tk.X, pady=(0, 16))

        self.run_btn = Btn(row, "▶  Run Script",    command=self._run_script,      variant="primary")
        self.run_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.run_all_btn = Btn(row, "⏩  Run All",   command=self._run_all_scripts, variant="primary")
        self.run_all_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = Btn(row, "⏹  Stop",         command=self._stop_script,     variant="danger")
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_btn.config(state=tk.DISABLED)

        sep(row, vertical=True).pack(side=tk.LEFT, fill=tk.Y, padx=(4, 12), pady=4)

        Btn(row, "Clear Output", command=self._clear_output, variant="secondary").pack(side=tk.LEFT, padx=(0, 8))
        Btn(row, "Exit",         command=self.root.quit,     variant="ghost").pack(side=tk.RIGHT)

    # ── Terminal output ───────────────────────────────────────────────────────

    def _build_terminal(self, parent):
        # Label row
        lrow = tk.Frame(parent, bg=C["bg"])
        lrow.pack(fill=tk.X, pady=(0, 8))
        tk.Label(lrow, text="Output", font=(FONT_UI, 11, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(side=tk.LEFT)

        # Outer terminal frame
        term = tk.Frame(parent, bg=C["term_bg"],
                        highlightthickness=1, highlightbackground=C["term_border"])
        term.pack(fill=tk.BOTH, expand=True)

        # macOS-style title bar with traffic-light dots
        tbar = tk.Frame(term, bg=C["term_bar"], height=32)
        tbar.pack(fill=tk.X)
        tbar.pack_propagate(False)
        for dot_col in ("#ff5f57", "#febc2e", "#28c840"):
            tk.Label(tbar, text="●", fg=dot_col, bg=C["term_bar"],
                     font=(FONT_UI, 12)).pack(side=tk.LEFT, padx=(9, 0))
        tk.Label(tbar, text="output terminal", font=(FONT_UI, 9),
                 bg=C["term_bar"], fg=C["term_dim"]).pack(side=tk.LEFT, padx=12)

        sep(term, bg=C["term_border"]).pack(fill=tk.X)

        # Text widget + scrollbar
        text_frame = tk.Frame(term, bg=C["term_bg"])
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(
            text_frame,
            bg=C["term_bg"], fg=C["term_text"],
            font=(FONT_MONO, 11),
            relief=tk.FLAT, bd=0,
            padx=16, pady=12,
            insertbackground=C["term_text"],
            selectbackground="#2d333b",
            selectforeground=C["term_text"],
            state=tk.DISABLED,
            wrap=tk.WORD,
            cursor="arrow",
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scroll = ttk.Scrollbar(text_frame, command=self.output_text.yview,
                               style="Vertical.TScrollbar")
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scroll.set)

        self.output_text.tag_config("success", foreground=C["term_green"])
        self.output_text.tag_config("error",   foreground=C["term_red"])
        self.output_text.tag_config("warning", foreground=C["term_yellow"])
        self.output_text.tag_config("info",    foreground=C["term_blue"])
        self.output_text.tag_config("dim",     foreground=C["term_dim"])

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=C["card"], height=28,
                       highlightthickness=1, highlightbackground=C["sep"])
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)

        self._status_dot  = tk.Label(bar, text="●", font=(FONT_UI, 11),
                                     bg=C["card"], fg=C["green"])
        self._status_dot.pack(side=tk.LEFT, padx=(14, 4))

        self._status_text = tk.Label(bar, text="Ready", font=(FONT_UI, 9),
                                     bg=C["card"], fg=C["text_muted"])
        self._status_text.pack(side=tk.LEFT)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _log(self, message, tag=None):
        self.output_text.config(state=tk.NORMAL)
        if tag:
            self.output_text.insert(tk.END, message + "\n", tag)
        else:
            self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.root.update()

    def _set_running(self, on):
        self.running = on
        if on:
            self.run_btn.config(state=tk.DISABLED)
            self.run_all_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self._status_dot.config(fg=C["accent"])
            self._status_text.config(text="Running…")
        else:
            self.run_btn.config(state=tk.NORMAL)
            self.run_all_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self._status_dot.config(fg=C["green"])
            self._status_text.config(text="Ready")
        self.root.update()

    def _clear_output(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)

    def _build_args(self):
        args = []
        inst = self.instance_var.get()
        if inst:
            entry = next((i for i in self.instances if i["name"] == inst), None)
            if entry:
                args += ["--instance", entry["value"]]
        k = self.k_var.get()
        if k:
            args += ["--k", k]
        return args

    # ── Run logic ─────────────────────────────────────────────────────────────

    def _run_script(self):
        name = self.script_var.get()
        if not name:
            messagebox.showerror("Error", "Please select a script.")
            return
        entry = next((s for s in self.scripts if s["name"] == name), None)
        if not entry:
            return
        path = os.path.join(ROOT, entry["file"])
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Script not found:\n{path}")
            return
        threading.Thread(
            target=self._run_in_thread, args=(path, self._build_args()), daemon=True
        ).start()

    def _run_all_scripts(self):
        if not self.scripts:
            messagebox.showerror("Error", "No scripts found in scripts/ folder.")
            return
        threading.Thread(
            target=self._run_all_in_thread, args=(self._build_args(),), daemon=True
        ).start()

    def _run_in_thread(self, path, args):
        try:
            self._set_running(True)
            self._log(f"\n  {'─' * 62}", "dim")
            self._log(f"  ▶  {os.path.basename(path)}", "info")
            if args:
                self._log(f"     {' '.join(args)}", "dim")
            self._log(f"  {'─' * 62}\n", "dim")

            cmd = [sys.executable, path] + args
            self.current_process = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            for line in self.current_process.stdout:
                self._log(line.rstrip())
            self.current_process.wait()
            code = self.current_process.returncode

            self._log(f"\n  {'─' * 62}", "dim")
            if code == 0:
                self._log("  ✓  Completed successfully.", "success")
            else:
                self._log(f"  ✗  Exited with code {code}.", "error")
            self._log(f"  {'─' * 62}\n", "dim")

        except Exception as e:
            self._log(f"\n  ✗  {e}", "error")
        finally:
            self.current_process = None
            self._set_running(False)

    def _run_all_in_thread(self, args):
        try:
            self._set_running(True)
            n = len(self.scripts)
            self._log(f"\n  {'═' * 62}", "info")
            self._log(f"  ⏩  Running all {n} scripts", "info")
            self._log(f"  {'═' * 62}\n", "info")

            for idx, script in enumerate(self.scripts, 1):
                if hasattr(self, "_stop_all"):
                    self._log("\n  ⊘  Aborted.", "warning")
                    break

                path = os.path.join(ROOT, script["file"])
                self._log(f"\n  [{idx}/{n}]  {script['name']}", "info")
                if args:
                    self._log(f"       {' '.join(args)}", "dim")
                self._log(f"  {'─' * 62}\n", "dim")

                cmd = [sys.executable, path] + args
                self.current_process = subprocess.Popen(
                    cmd, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                for line in self.current_process.stdout:
                    self._log(line.rstrip())
                self.current_process.wait()
                code = self.current_process.returncode
                self.current_process = None

                if code == 0:
                    self._log(f"\n  ✓  {script['name']} done.", "success")
                else:
                    self._log(f"\n  ✗  {script['name']} exited {code}.", "error")

            self._log(f"\n  {'═' * 62}", "info")
            self._log("  ✓  All scripts completed.", "success")
            self._log(f"  {'═' * 62}\n", "info")

        except Exception as e:
            self._log(f"\n  ✗  {e}", "error")
        finally:
            self.current_process = None
            self._set_running(False)
            if hasattr(self, "_stop_all"):
                delattr(self, "_stop_all")

    def _stop_script(self):
        self._stop_all = True
        if self.current_process:
            try:
                self.current_process.terminate()
                self._log("\n  ⏹  Terminating process…", "warning")
                self.current_process.wait(timeout=5)
                self._log("  ✓  Stopped.", "success")
            except Exception as e:
                self._log(f"  ✗  {e}", "error")
                try:
                    self.current_process.kill()
                except Exception:
                    pass


def main():
    root = tk.Tk()
    app = VRPLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
