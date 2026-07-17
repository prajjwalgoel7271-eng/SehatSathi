import subprocess
import sys
import os
import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import threading

# ─── Globals ─────────────────────────────────────────────────────────────────
SELECTED_CAMERA_INDEX = 0
AVAILABLE_CAMERAS = []


# ─── Camera Scanning ─────────────────────────────────────────────────────────

def scan_cameras(max_index=6):
    """Scan for available camera devices by probing indices 0..max_index-1."""
    cameras = []
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow on Windows
            if cap is not None and cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append(i)
                cap.release()
        except Exception:
            pass
    return cameras


# ─── Disclaimer Screen ──────────────────────────────────────────────────────

def show_disclaimer(on_accept):
    """
    Full-screen medical disclaimer that must be acknowledged before
    the launcher loads.  Calls *on_accept()* when the user clicks
    'I Understand — Continue'.
    """
    win = tk.Toplevel()
    win.title("SehatSaathi — Medical Disclaimer")
    win.configure(bg="#0D0D12")
    win.geometry("760x620")
    win.resizable(False, False)

    # Prevent closing without acknowledgment
    win.protocol("WM_DELETE_WINDOW", lambda: None)

    # ── outer container
    outer = tk.Frame(win, bg="#0D0D12")
    outer.pack(expand=True, fill=tk.BOTH, padx=40, pady=30)

    # ── icon + title
    tk.Label(
        outer, text="⚕️", font=("Segoe UI Emoji", 40),
        bg="#0D0D12", fg="#FF6B6B"
    ).pack(pady=(0, 5))

    tk.Label(
        outer, text="Medical Disclaimer",
        font=("Segoe UI", 24, "bold"), bg="#0D0D12", fg="#FFFFFF"
    ).pack()

    # ── separator accent
    sep = tk.Frame(outer, bg="#FF6B6B", height=2)
    sep.pack(fill=tk.X, pady=(12, 18), padx=60)

    # ── disclaimer body
    body_frame = tk.Frame(outer, bg="#16161E", highlightbackground="#2A2A3A",
                          highlightthickness=1)
    body_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 18))

    disclaimer_text = (
        "Sehat Sathi is a screening tool, not a diagnostic device. "
        "It uses computer vision and audio analysis to estimate risk "
        "indicators for Parkinson's, Anemia, and Tuberculosis — it does "
        "not replace a clinical diagnosis, blood test, or examination by "
        "a licensed doctor.\n\n"
        "Results shown are probabilistic risk estimates, not medical facts. "
        "Ensure good lighting and a quiet environment for best accuracy.\n\n"
        "If this tool flags any risk — mild, moderate, or severe — please "
        "consult a qualified medical professional for proper testing and "
        "diagnosis.\n\n"
        "By continuing, you acknowledge this app is for "
        "educational/screening purposes only."
    )

    body_label = tk.Label(
        body_frame, text=disclaimer_text,
        font=("Segoe UI", 12), bg="#16161E", fg="#C0C0D0",
        wraplength=620, justify="left", padx=25, pady=20
    )
    body_label.pack(expand=True, fill=tk.BOTH)

    # ── accept button
    def accept():
        win.destroy()
        on_accept()

    btn = tk.Button(
        outer, text="   I Understand — Continue   ",
        font=("Segoe UI", 13, "bold"),
        bg="#2ECC71", fg="#FFFFFF", activebackground="#27AE60",
        activeforeground="#FFFFFF", relief=tk.FLAT, bd=0,
        cursor="hand2", command=accept
    )
    btn.pack(pady=(0, 5), ipady=10)

    # hover effect
    btn.bind("<Enter>", lambda e: btn.config(bg="#27AE60"))
    btn.bind("<Leave>", lambda e: btn.config(bg="#2ECC71"))

    # Center
    win.update_idletasks()
    w = win.winfo_width()
    h = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (w // 2)
    y = (win.winfo_screenheight() // 2) - (h // 2)
    win.geometry(f"+{x}+{y}")
    win.grab_set()


# ─── Module Launcher ─────────────────────────────────────────────────────────

def launch_module(filename):
    """Launch a sub-module, passing the selected camera index via env var."""
    if not os.path.exists(filename):
        messagebox.showerror(
            "Module Not Available",
            f"The module '{filename}' was not found.\n\n"
            "Please make sure it is in the same directory as main.py."
        )
    else:
        env = os.environ.copy()
        env["SEHATSAATHI_CAMERA_INDEX"] = str(SELECTED_CAMERA_INDEX)
        subprocess.Popen([sys.executable, filename], env=env)


# ─── Card Widget ─────────────────────────────────────────────────────────────

def create_card(parent, icon, title, description, filename, accent_color):
    """A dark card with icon, title, description, and a launch button."""

    card = tk.Frame(
        parent, bg="#181822",
        highlightbackground="#2A2A3A", highlightthickness=1,
        padx=20, pady=22
    )

    # ── hover glow on card border
    def on_enter(e):
        card.config(highlightbackground=accent_color, highlightthickness=2)

    def on_leave(e):
        card.config(highlightbackground="#2A2A3A", highlightthickness=1)

    card.bind("<Enter>", on_enter)
    card.bind("<Leave>", on_leave)

    card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=12, pady=12)

    # icon
    icon_lbl = tk.Label(
        card, text=icon, font=("Segoe UI Emoji", 44),
        bg="#181822", fg=accent_color
    )
    icon_lbl.pack(pady=(5, 4))

    # title
    title_lbl = tk.Label(
        card, text=title, font=("Segoe UI", 15, "bold"),
        bg="#181822", fg="#F0F0F5"
    )
    title_lbl.pack(pady=(0, 8))

    # description
    desc_lbl = tk.Label(
        card, text=description, font=("Segoe UI", 10),
        bg="#181822", fg="#888899", wraplength=200, justify="center"
    )
    desc_lbl.pack(expand=True, fill=tk.BOTH)

    # launch button
    btn = tk.Button(
        card, text="  Launch  ", font=("Segoe UI", 11, "bold"),
        bg="#22222E", fg=accent_color, activebackground=accent_color,
        activeforeground="#0D0D12", relief=tk.FLAT, bd=0,
        command=lambda: launch_module(filename), cursor="hand2"
    )

    def btn_enter(e):
        btn.config(bg=accent_color, fg="#0D0D12")
        on_enter(None)

    def btn_leave(e):
        btn.config(bg="#22222E", fg=accent_color)
        on_leave(None)

    btn.bind("<Enter>", btn_enter)
    btn.bind("<Leave>", btn_leave)

    # propagate card hover from inner labels
    for widget in (icon_lbl, title_lbl, desc_lbl):
        widget.bind("<Enter>", lambda e: on_enter(None))
        widget.bind("<Leave>", lambda e: on_leave(None))

    btn.pack(pady=(12, 5), ipadx=22, ipady=8)


# ─── Main Launcher ──────────────────────────────────────────────────────────

def build_launcher(root):
    """Build the main launcher UI (called after disclaimer is accepted)."""
    global AVAILABLE_CAMERAS, SELECTED_CAMERA_INDEX

    root.deiconify()  # un-hide the root window

    # ── header
    header = tk.Frame(root, bg="#0D0D12", pady=18)
    header.pack(fill=tk.X)

    tk.Label(
        header, text="SehatSaathi",
        font=("Segoe UI", 34, "bold"), bg="#0D0D12", fg="#FFFFFF"
    ).pack()

    tk.Label(
        header, text="AI-Powered Early Disease Screening",
        font=("Segoe UI", 13), bg="#0D0D12", fg="#6E6E80"
    ).pack(pady=(4, 0))

    # accent separator
    tk.Frame(header, bg="#BB86FC", height=2).pack(fill=tk.X, padx=180, pady=(14, 0))

    # ── camera selector bar
    cam_bar = tk.Frame(root, bg="#12121A", pady=10)
    cam_bar.pack(fill=tk.X, padx=30)

    tk.Label(
        cam_bar, text="📷  Camera:",
        font=("Segoe UI", 11), bg="#12121A", fg="#AAAABC"
    ).pack(side=tk.LEFT, padx=(15, 8))

    cam_var = tk.StringVar()

    cam_combo = ttk.Combobox(
        cam_bar, textvariable=cam_var,
        state="readonly", width=22,
        font=("Segoe UI", 10)
    )
    cam_combo.pack(side=tk.LEFT, padx=(0, 10))

    # Style the combobox for dark theme readability
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "TCombobox",
        fieldbackground="#22222E", background="#22222E",
        foreground="#E0E0F0", bordercolor="#3A3A4A",
        arrowcolor="#AAAABC", selectbackground="#3A3A4A",
        selectforeground="#E0E0F0"
    )

    def populate_camera_list():
        global AVAILABLE_CAMERAS, SELECTED_CAMERA_INDEX
        cam_combo.set("Scanning...")
        root.update()

        def _scan():
            global AVAILABLE_CAMERAS, SELECTED_CAMERA_INDEX
            AVAILABLE_CAMERAS = scan_cameras()
            if AVAILABLE_CAMERAS:
                labels = [f"Camera {i}" for i in AVAILABLE_CAMERAS]
                cam_combo["values"] = labels
                cam_combo.current(0)
                SELECTED_CAMERA_INDEX = AVAILABLE_CAMERAS[0]
            else:
                cam_combo["values"] = ["No camera found"]
                cam_combo.set("No camera found")
                SELECTED_CAMERA_INDEX = 0

        thread = threading.Thread(target=_scan, daemon=True)
        thread.start()
        # poll until done
        def poll():
            if thread.is_alive():
                root.after(200, poll)
            else:
                if AVAILABLE_CAMERAS:
                    labels = [f"Camera {i}" for i in AVAILABLE_CAMERAS]
                    cam_combo["values"] = labels
                    cam_combo.current(0)
                    SELECTED_CAMERA_INDEX = AVAILABLE_CAMERAS[0]
                else:
                    cam_combo["values"] = ["No camera found"]
                    cam_combo.set("No camera found")
        root.after(200, poll)

    def on_camera_selected(event):
        global SELECTED_CAMERA_INDEX
        idx = cam_combo.current()
        if 0 <= idx < len(AVAILABLE_CAMERAS):
            SELECTED_CAMERA_INDEX = AVAILABLE_CAMERAS[idx]

    cam_combo.bind("<<ComboboxSelected>>", on_camera_selected)

    refresh_btn = tk.Button(
        cam_bar, text="🔄 Refresh", font=("Segoe UI", 10),
        bg="#22222E", fg="#AAAABC", activebackground="#3A3A4A",
        activeforeground="#E0E0F0", relief=tk.FLAT, bd=0,
        cursor="hand2", command=populate_camera_list
    )
    refresh_btn.pack(side=tk.LEFT, padx=4, ipadx=8, ipady=3)
    refresh_btn.bind("<Enter>", lambda e: refresh_btn.config(bg="#3A3A4A"))
    refresh_btn.bind("<Leave>", lambda e: refresh_btn.config(bg="#22222E"))

    # initial scan
    populate_camera_list()

    # ── disease cards
    cards_container = tk.Frame(root, bg="#0D0D12")
    cards_container.pack(expand=True, fill=tk.BOTH, padx=28, pady=(8, 10))

    create_card(
        cards_container,
        icon="🧠",
        title="Parkinson's Detection",
        description="Multi-modal neurological screening: spiral, motor, voice & reaction tests.",
        filename="Parkinson1.py",
        accent_color="#BB86FC"
    )

    create_card(
        cards_container,
        icon="🩸",
        title="Anemia Detection",
        description="Eye conjunctiva & palm pallor analysis via computer vision.",
        filename="anemia2.py",
        accent_color="#FF5252"
    )

    create_card(
        cards_container,
        icon="🫁",
        title="TB Screening",
        description="Cough audio analysis using MFCC, spectral features & AI.",
        filename="TB Checker.py",
        accent_color="#00BCD4"
    )

    # ── footer disclaimer
    footer = tk.Frame(root, bg="#0D0D12", pady=8)
    footer.pack(fill=tk.X, side=tk.BOTTOM)

    tk.Label(
        footer,
        text="⚕️  Screening tool only — not a medical diagnosis. Consult a doctor for any flagged risk.",
        font=("Segoe UI", 9), bg="#0D0D12", fg="#555566"
    ).pack()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("SehatSaathi — AI Disease Screening")
    root.geometry("960x600")
    root.minsize(860, 540)
    root.configure(bg="#0D0D12")
    root.withdraw()  # hide until disclaimer is accepted

    # Show disclaimer first, then build the launcher on acceptance
    show_disclaimer(lambda: build_launcher(root))

    root.mainloop()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
