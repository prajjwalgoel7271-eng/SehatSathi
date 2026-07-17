import sys
sys.modules['tensorflow'] = None

import cv2
import numpy as np
import mediapipe as mp
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import parselmouth
from parselmouth.praat import call
import tkinter as tk
from tkinter import Canvas, messagebox, ttk
from PIL import Image, ImageTk
import time, random, os, json, glob
from scipy.signal import find_peaks
from sklearn.neighbors import KNeighborsClassifier

# Global scores
motor_conf = 0.0
voice_conf = 0.0
spiral_conf = 0.0
reaction_conf = 0.0

# Raw clinical metrics
raw_motor_decrement = 0.0
raw_motor_speed = 0.0
raw_voice_jitter = 0.0
raw_voice_shimmer = 0.0
raw_voice_hnr = 0.0
raw_spiral_dev = 0.0
raw_reaction_lat = 0.0

# Camera index from launcher (env var) or default 0
CAMERA_INDEX = int(os.environ.get("SEHATSAATHI_CAMERA_INDEX", "0"))

# Configurable Clinical Thresholds Configuration
TH_JITTER_PCT = 1.04         # voice jitter (%)
TH_SHIMMER_PCT = 3.81        # voice shimmer (%)
TH_HNR_DB = 20.0             # Harmonics-to-Noise Ratio (dB)
TH_TAP_DECREMENT_PCT = 30.0  # finger tapping amplitude decrement (%)
TH_SPIRAL_TREMOR_MIN_HZ = 4.0
TH_SPIRAL_TREMOR_MAX_HZ = 6.0
TH_SPIRAL_DEV_MAX_PX = 45.0
TH_REACTION_MEAN_MS = 650.0   # adjusted to 650ms for mouse movement
TH_REACTION_STD_MS = 150.0    # adjusted to 150ms for mouse movement

# Scoring Weights (slightly weighted towards voice and motor)
WT_MOTOR = 0.30
WT_VOICE = 0.30
WT_SPIRAL = 0.20
WT_REACTION = 0.20


class EnhancedNeuroTesterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Neurological Health Assessment")
        self.root.geometry("1000x740")
        self.root.configure(bg="#0D0D12")
        
        # Center the window
        self.root.update_idletasks()
        w = 1000
        h = 740
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # Set up ttk progress bar style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "Purple.Horizontal.TProgressbar",
            troughcolor="#16161E",
            background="#BB86FC",
            bordercolor="#2A2A3A",
            lightcolor="#BB86FC",
            darkcolor="#BB86FC"
        )

        self.setup_main_interface()

    def setup_main_interface(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#0D0D12")
        header_frame.pack(fill='x', padx=30, pady=(12, 6))
        
        title = tk.Label(
            header_frame,
            text="[Neuro] Neurological Health Assessment",
            font=("Segoe UI", 24, "bold"),
            bg="#0D0D12",
            fg="#FFFFFF"
        )
        title.pack()
        
        subtitle = tk.Label(
            header_frame,
            text="Multi-Modal Neurological Screening System",
            font=("Segoe UI", 11),
            bg="#0D0D12",
            fg="#6E6E80"
        )
        subtitle.pack(pady=(2, 0))

        # Purple separator accent
        sep = tk.Frame(header_frame, bg="#BB86FC", height=2)
        sep.pack(fill='x', padx=180, pady=(8, 0))
        
        # Progress section
        progress_frame = tk.Frame(self.root, bg="#16161E", highlightbackground="#2A2A3A", highlightthickness=1)
        progress_frame.pack(fill='x', padx=30, pady=6)
        
        tk.Label(
            progress_frame,
            text="Test Progress:",
            font=("Segoe UI", 11, "bold"),
            bg="#16161E",
            fg="#AAAABC"
        ).pack(anchor='w', padx=20, pady=(8, 2))
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            style="Purple.Horizontal.TProgressbar",
            length=900,
            mode="determinate"
        )
        self.progress_bar.pack(padx=20, pady=(2, 6), fill='x')
        self.progress_bar['value'] = 0
        
        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to start testing",
            font=("Segoe UI", 10),
            bg="#16161E",
            fg="#888899"
        )
        self.progress_label.pack(anchor='w', padx=20, pady=(0, 8))
        
        # Test buttons frame
        buttons_frame = tk.Frame(self.root, bg="#0D0D12")
        buttons_frame.pack(fill='both', expand=True, padx=30, pady=6)
        
        tests = [
            ("[Motor] Motor Test", "Test finger dexterity & coordination", self.run_motor_test, "#FF6B6B"),
            ("[Voice] Voice Test", "Analyze speech patterns & stability", self.run_voice_test, "#3498db"),
            ("[Spiral] Spiral Test", "Drawing coordination assessment", self.run_spiral_test, "#9b59b6"),
            ("[Reaction] Reaction Test", "Interactive latency & movement analysis", self.run_reaction_test, "#2ecc71")
        ]
        
        for i, (name, desc, command, accent_color) in enumerate(tests):
            card = tk.Frame(
                buttons_frame, bg="#181822",
                highlightbackground="#2A2A3A", highlightthickness=1,
                padx=15, pady=8
            )
            card.pack(fill='x', pady=4)
            
            # Hover highlight border effect
            def make_hover(c_widget, color):
                def on_enter(e):
                    c_widget.config(highlightbackground=color, highlightthickness=1)
                def on_leave(e):
                    c_widget.config(highlightbackground="#2A2A3A", highlightthickness=1)
                c_widget.bind("<Enter>", on_enter)
                c_widget.bind("<Leave>", on_leave)
                
            make_hover(card, accent_color)

            # Button
            btn = tk.Button(
                card, text=name, font=("Segoe UI", 11, "bold"),
                bg="#22222E", fg=accent_color, activebackground=accent_color,
                activeforeground="#0D0D12", relief=tk.FLAT, bd=0,
                width=18, height=2, cursor="hand2", command=command
            )
            btn.pack(side='left', padx=10)
            
            def make_btn_hover(b_widget, act_color, parent_card):
                def btn_enter(e):
                    b_widget.config(bg=act_color, fg="#0D0D12")
                    parent_card.config(highlightbackground=act_color, highlightthickness=1)
                def btn_leave(e):
                    b_widget.config(bg="#22222E", fg=act_color)
                    parent_card.config(highlightbackground="#2A2A3A", highlightthickness=1)
                b_widget.bind("<Enter>", btn_enter)
                b_widget.bind("<Leave>", btn_leave)
                
            make_btn_hover(btn, accent_color, card)
            
            desc_lbl = tk.Label(
                card, text=desc, font=("Segoe UI", 10),
                bg="#181822", fg="#888899"
            )
            desc_lbl.pack(side='left', padx=20)
            
            # Propagate hover to card
            desc_lbl.bind("<Enter>", lambda e, c=card, col=accent_color: c.config(highlightbackground=col))
            desc_lbl.bind("<Leave>", lambda e, c=card: c.config(highlightbackground="#2A2A3A"))

        # Results section (height reduced to 3 lines)
        results_frame = tk.Frame(self.root, bg="#16161E", highlightbackground="#2A2A3A", highlightthickness=1)
        results_frame.pack(fill='x', padx=30, pady=6)
        
        tk.Label(
            results_frame,
            text="Test Results:",
            font=("Segoe UI", 11, "bold"),
            bg="#16161E",
            fg="#AAAABC"
        ).pack(anchor='w', padx=20, pady=(6, 2))
        
        self.results_text = tk.Text(
            results_frame,
            height=3,
            bg="#12121A",
            fg="#E0E0F0",
            font=("Consolas", 10),
            highlightthickness=1,
            highlightbackground="#2A2A3A",
            insertbackground="white",
            padx=10,
            pady=3
        )
        self.results_text.pack(fill='both', expand=True, padx=20, pady=(2, 8))
        
        # Final score button frame (compact layout)
        final_frame = tk.Frame(self.root, bg="#0D0D12")
        final_frame.pack(fill='x', padx=30, pady=6)
        
        self.final_btn = tk.Button(
            final_frame,
            text="Calculate Health Index",
            font=("Segoe UI", 11, "bold"),
            bg="#2ECC71", fg="#FFFFFF", activebackground="#27AE60",
            activeforeground="#FFFFFF", relief=tk.FLAT, bd=0,
            cursor="hand2", command=self.show_final_score,
            width=26
        )
        self.final_btn.pack(side=tk.LEFT, padx=10, pady=2, ipady=6, expand=True)
        self.final_btn.bind("<Enter>", lambda e: self.final_btn.config(bg="#27AE60"))
        self.final_btn.bind("<Leave>", lambda e: self.final_btn.config(bg="#2ECC71"))
        
        self.readings_btn = tk.Button(
            final_frame,
            text="View Detailed Readings",
            font=("Segoe UI", 11, "bold"),
            bg="#9B59B6", fg="#FFFFFF", activebackground="#8E44AD",
            activeforeground="#FFFFFF", relief=tk.FLAT, bd=0,
            cursor="hand2", command=self.show_detailed_readings,
            width=26
        )
        self.readings_btn.pack(side=tk.LEFT, padx=10, pady=2, ipady=6, expand=True)
        self.readings_btn.bind("<Enter>", lambda e: self.readings_btn.config(bg="#8E44AD"))
        self.readings_btn.bind("<Leave>", lambda e: self.readings_btn.config(bg="#9B59B6"))
        
        self.final_score_label = tk.Label(
            self.root,
            text="Complete all tests to see your Neurological Health Index",
            font=("Segoe UI", 11, "bold"),
            bg="#0D0D12",
            fg="orange"
        )
        self.final_score_label.pack(pady=4)

    def update_progress(self, value, text):
        self.progress_bar['value'] = value * 100
        self.progress_label.configure(text=text)
        self.root.update()
    
    def add_result(self, test_name, score):
        self.results_text.insert("end", f"   • {test_name}: {score:.2f}%\n")
        self.root.update()
    
    def run_motor_test(self):
        self.update_progress(0, "Starting Motor Test...")
        messagebox.showinfo("Motor Test", "Move thumb and index finger apart and together rhythmically.\nTest runs for 20 seconds.")
        motor_test_enhanced(self)
        self.update_progress(0.25, "Motor Test completed")
        self.add_result("Motor Test", motor_conf)
    
    def run_voice_test(self):
        self.update_progress(0.25, "Starting Voice Test...")
        messagebox.showinfo("Voice Test", 'Say a steady "AAAAA" or any vowel sound.\nRecording for 5 seconds.')
        voice_test_enhanced(self)
        self.update_progress(0.50, "Voice Test completed")
        self.add_result("Voice Test", voice_conf)
    
    def run_spiral_test(self):
        self.update_progress(0.50, "Starting Spiral Test...")
        spiral_test_enhanced(self)
        self.update_progress(0.75, "Spiral Test completed")
        self.add_result("Spiral Test", spiral_conf)
    
    def run_reaction_test(self):
        self.update_progress(0.75, "Starting Reaction Test...")
        messagebox.showinfo("Reaction Test", "Position cursor on canvas. Move cursor to touch the red target circles as they spawn.\nTest runs for 10 trials.")
        reaction_test_enhanced(self)
        self.update_progress(1.0, "All tests completed!")
        self.add_result("Reaction Test", reaction_conf)
    
    def show_final_score(self):
        if motor_conf == 0.0 or voice_conf == 0.0 or spiral_conf == 0.0 or reaction_conf == 0.0:
            messagebox.showwarning("Incomplete Tests", 
                                 "Please complete all four tests before calculating the final score!")
            return
            
        # Slightly weighted score calculation
        avg = (motor_conf * WT_MOTOR) + (voice_conf * WT_VOICE) + (spiral_conf * WT_SPIRAL) + (reaction_conf * WT_REACTION)
        
        if avg >= 75:
            color = "#2ecc71"
            status = "EXCELLENT"
        elif avg >= 70:
            color = "#27ae60"
            status = "GOOD"
        elif avg >= 65:
            color = "#f39c12"
            status = "FAIR"
        else:
            color = "#e74c3c"
            status = "Needs Attention"
        
        self.final_score_label.configure(
            text=f"NEUROLOGICAL HEALTH INDEX (WEIGHTED): {avg:.2f}% - {status}",
            fg=color
        )
        
        result = f"""
=========================================
      NEUROLOGICAL HEALTH INDEX
=========================================

Motor Test (30% WT):  {motor_conf:6.2f}%
Voice Test (30% WT):  {voice_conf:6.2f}%  
Spiral Test (20% WT): {spiral_conf:6.2f}%
Reaction Test (20% WT):{reaction_conf:6.2f}%

-----------------------------------------
OVERALL SCORE:     {avg:6.2f}%
Health Status:     {status}
=========================================
"""
        messagebox.showinfo("Final Neurological Health Index", result)

    def show_detailed_readings(self):
        if (raw_motor_speed == 0.0 and raw_voice_jitter == 0.0 and 
            raw_spiral_dev == 0.0 and raw_reaction_lat == 0.0):
            messagebox.showwarning("No Readings", 
                                 "Please complete tests first to collect raw clinical readings!")
            return
            
        readings = f"""
=========================================
        DETAILED CLINICAL READINGS
=========================================

Motor Finger Tapping:
  - Amplitude Decrement: {raw_motor_decrement:.2f}% (Threshold: {TH_TAP_DECREMENT_PCT}%)
  - Tapping Speed:      {raw_motor_speed:.2f} taps/sec

Voice Stability:
  - Jitter (local):     {raw_voice_jitter:.3f}% (Threshold: {TH_JITTER_PCT}%)
  - Shimmer (local):    {raw_voice_shimmer:.3f}% (Threshold: {TH_SHIMMER_PCT}%)
  - HNR:                {raw_voice_hnr:.2f} dB (Threshold: {TH_HNR_DB} dB)

Spiral Tracing:
  - Mean Deviation:     {raw_spiral_dev:.2f} pixels

Reaction Latency:
  - Mean Latency:       {raw_reaction_lat:.2f} ms (Threshold: {TH_REACTION_MEAN_MS + 2*TH_REACTION_STD_MS:.1f} ms)
=========================================
"""
        messagebox.showinfo("Detailed Clinical Readings", readings)


# Test functions (keeping original logic, just enhanced visuals)
mp_hands = mp.solutions.hands

def motor_test_enhanced(gui):
    global motor_conf, raw_motor_decrement, raw_motor_speed
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "No camera detected. Please check your camera connection.")
            motor_conf = 0.0
            return
    except Exception as e:
        messagebox.showerror("Camera Error", f"Could not access camera: {e}")
        motor_conf = 0.0
        return

    hands = mp_hands.Hands(max_num_hands=1)
    distances, timestamps = [], []
    start = time.time()
    duration = 20

    try:
        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            elapsed = time.time() - start
            remaining = duration - elapsed

            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            bar_width = int((elapsed / duration) * (w - 40))
            cv2.rectangle(frame, (20, 20), (w - 20, 50), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 20), (20 + bar_width, 50), (0, 255, 150), -1)
            cv2.rectangle(frame, (20, 20), (w - 20, 50), (0, 200, 255), 2)

            cv2.putText(frame, f"Time: {remaining:.1f}s", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    thumb = handLms.landmark[4]
                    index = handLms.landmark[8]
                    x1, y1 = int(thumb.x * w), int(thumb.y * h)
                    x2, y2 = int(index.x * w), int(index.y * h)

                    cv2.circle(frame, (x1, y1), 15, (255, 0, 0), -1)
                    cv2.circle(frame, (x2, y2), 15, (0, 255, 0), -1)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                    dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
                    distances.append(dist)
                    timestamps.append(elapsed)

                    cv2.putText(frame, f"Distance: {int(dist)}px", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Motor Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(distances) < 20:
        motor_conf = 0.0
        return

    # Clinical analysis: amplitude decrement, speed, regularity
    dist_arr = np.array(distances)
    time_arr = np.array(timestamps)

    # Detect tap peaks (local maxima = fingers apart)
    peaks, props = find_peaks(dist_arr, distance=8, prominence=15)

    if len(peaks) >= 6:
        peak_amplitudes = dist_arr[peaks]
        first5 = np.mean(peak_amplitudes[:5])
        last5 = np.mean(peak_amplitudes[-5:])
        decrement = ((first5 - last5) / first5) * 100 if first5 > 0 else 0

        # Tapping speed (taps per second)
        if len(peaks) >= 2:
            tap_intervals = np.diff(time_arr[peaks])
            tapping_speed = 1.0 / np.mean(tap_intervals) if np.mean(tap_intervals) > 0 else 0
        else:
            tapping_speed = 0

        # Regularity
        regularity_cv = np.std(tap_intervals) / np.mean(tap_intervals) if len(tap_intervals) > 0 and np.mean(tap_intervals) > 0 else 1.0

        # Save raw values
        raw_motor_decrement = decrement
        raw_motor_speed = tapping_speed

        # Scoring
        decrement_penalty = min(40, max(0, decrement - 10) * 1.3)
        speed_penalty = max(0, (2.0 - tapping_speed) * 10) if tapping_speed < 2.0 else 0
        regularity_penalty = min(20, regularity_cv * 30)

        motor_conf = max(0.0, min(100.0, 100.0 - decrement_penalty - speed_penalty - regularity_penalty))

        # Plot
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(time_arr, dist_arr, 'cyan', linewidth=2)
        ax1.plot(time_arr[peaks], dist_arr[peaks], 'rv', markersize=10, label='Tap peaks')
        ax1.set_title(f"Finger Tapping — Decrement: {decrement:.1f}% | Speed: {tapping_speed:.1f} taps/s", fontsize=14)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Distance (px)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        fft = np.fft.fft(dist_arr)
        freqs = np.fft.fftfreq(len(dist_arr), d=np.mean(np.diff(time_arr)))
        ax2.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(fft)//2], 'yellow', linewidth=2)
        ax2.set_title("Frequency Analysis", fontsize=14)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        deviation = np.std(distances)
        motor_conf = max(0.0, 100.0 - deviation / 5)
        raw_motor_decrement = 0.0
        raw_motor_speed = 0.0


def voice_test_enhanced(gui):
    global voice_conf, raw_voice_jitter, raw_voice_shimmer, raw_voice_hnr
    fs = 22050
    duration = 5

    for i in range(3, 0, -1):
        gui.progress_label.configure(text=f"Recording starts in {i}...")
        gui.root.update()
        time.sleep(1)

    gui.progress_label.configure(text='Say a steady "AAAAA" or any vowel sound...')
    gui.root.update()

    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
    except Exception as e:
        messagebox.showerror("Microphone Error", f"Could not access microphone:\n{e}")
        voice_conf = 0.0
        return

    y = audio.flatten()

    # Parselmouth / Praat clinical analysis
    snd = parselmouth.Sound(y, sampling_frequency=fs)
    pitch = call(snd, "To Pitch", 0.0, 75, 500)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    # Jitter
    try:
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    except Exception:
        jitter = 0.0

    # Shimmer
    try:
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    except Exception:
        shimmer = 0.0

    # HNR
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
    except Exception:
        hnr = 25.0

    # Save raw values
    raw_voice_jitter = jitter
    raw_voice_shimmer = shimmer
    raw_voice_hnr = hnr

    # Scoring based on clinical thresholds
    score = 100.0
    if jitter > TH_JITTER_PCT:
        score -= min(30, (jitter - TH_JITTER_PCT) * 15)
    if shimmer > TH_SHIMMER_PCT:
        score -= min(30, (shimmer - TH_SHIMMER_PCT) * 8)
    if hnr < TH_HNR_DB:
        score -= min(30, (TH_HNR_DB - hnr) * 3)

    voice_conf = max(0.0, min(100.0, score))

    # Librosa pitch tracking for visualization
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=fs)
    times = librosa.times_like(f0)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(times, f0, 'lime', linewidth=2)
    axes[0].set_title("Voice Pitch Over Time", fontsize=14)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].grid(True, alpha=0.3)

    time_audio = np.linspace(0, duration, len(y))
    axes[1].plot(time_audio, y, 'cyan', alpha=0.7)
    axes[1].set_title("Audio Waveform", fontsize=14)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    # Clinical metrics bar chart
    colors_bar = [
        '#2ecc71' if jitter <= TH_JITTER_PCT else '#e74c3c',
        '#2ecc71' if shimmer <= TH_SHIMMER_PCT else '#e74c3c',
        '#2ecc71' if hnr >= TH_HNR_DB else '#e74c3c'
    ]
    metrics = [jitter, shimmer, hnr]
    labels = [
        f'Jitter\n{jitter:.2f}%\n(<= {TH_JITTER_PCT}%)', 
        f'Shimmer\n{shimmer:.2f}%\n(<= {TH_SHIMMER_PCT}%)', 
        f'HNR\n{hnr:.1f}dB\n(>= {TH_HNR_DB}dB)'
    ]
    axes[2].bar(labels, metrics, color=colors_bar, edgecolor='white', linewidth=0.5)
    axes[2].set_title("Clinical Voice Metrics (Green=Normal, Red=Abnormal)", fontsize=14)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def resample_points(points, num_points=100):
    """Resample/normalize a path of 2D coordinates into exactly num_points."""
    pts = np.array(points, dtype=float)[:, :2]
    
    # Remove duplicates
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    keep = np.where(dists > 1e-5)[0]
    clean_pts = [pts[0]]
    for idx in keep:
        clean_pts.append(pts[idx + 1])
    clean_pts = np.array(clean_pts)
    
    if len(clean_pts) < 2:
        if len(clean_pts) == 1:
            return np.repeat(clean_pts, num_points, axis=0)
        return np.zeros((num_points, 2))
        
    diffs = np.diff(clean_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.zeros(len(clean_pts))
    cum_dists[1:] = np.cumsum(dists)
    
    total_dist = cum_dists[-1]
    if total_dist < 1e-5:
        return np.repeat(clean_pts[:1], num_points, axis=0)
        
    target_dists = np.linspace(0, total_dist, num_points)
    new_x = np.interp(target_dists, cum_dists, clean_pts[:, 0])
    new_y = np.interp(target_dists, cum_dists, clean_pts[:, 1])
    return np.column_stack((new_x, new_y))


def _load_spiral_dataset():
    """Load all spiral JSON files and extract features for classification."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "spiral_dataset")

    features, labels = [], []
    for pattern, label in [("healthy_*.json", 0), ("parkinson_*.json", 1)]:
        for fpath in glob.glob(os.path.join(dataset_dir, pattern)):
            try:
                with open(fpath, 'r') as f:
                    points = json.load(f)
                if len(points) < 10:
                    continue
                feat = _extract_spiral_features(points)
                if feat is not None:
                    features.append(feat)
                    labels.append(label)
            except Exception:
                continue
    return np.array(features) if features else None, np.array(labels) if labels else None


def _extract_spiral_features(points, mean_dev=None, std_dev=None):
    """Extract normalized clinical features from resampled and scaled points."""
    try:
        # 1. Resample to 100 points
        pts = resample_points(points, num_points=100)

        # 2. Center of mass normalization
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        pts_centered = pts - [cx, cy]

        # 3. Scale normalization by max radius
        radii = np.sqrt(pts_centered[:, 0]**2 + pts_centered[:, 1]**2)
        max_r = np.max(radii)
        
        # FILTER OUT STATIC MOCK POINTS (prevent zero-feature vector duplicates)
        if max_r < 5.0:
            return None

        pts_norm = pts_centered / max_r
        radii_norm = radii / max_r

        # 4. Ideal spiral radii (normalized)
        if mean_dev is None or std_dev is None:
            ideal_norm = np.linspace(radii_norm[0], radii_norm[-1], len(radii_norm))
            deviation_norm = radii_norm - ideal_norm
            mean_dev = np.mean(np.abs(deviation_norm))
            std_dev = np.std(deviation_norm)

        # 5. Tremor frequency via FFT on normalized deviation signal
        # 100 points, assume average drawing duration of ~3.3 seconds (fs = 30 Hz)
        # Bins 13 to 20 correspond to the 4-6 Hz band.
        fft_vals = np.abs(np.fft.rfft(deviation_norm - np.mean(deviation_norm)))
        tremor_power = np.sum(fft_vals[13:21]) / (np.sum(fft_vals) + 1e-8)

        # 6. Curvature variance
        dx = np.diff(pts_norm[:, 0])
        dy = np.diff(pts_norm[:, 1])
        if len(dx) > 2:
            angles = np.unwrap(np.arctan2(dy, dx))
            curvature = np.diff(angles)
            curvature_var = np.var(curvature)
        else:
            curvature_var = 0.0

        # 7. Total path length
        segments = np.sqrt(dx**2 + dy**2)
        total_length = np.sum(segments)

        return [mean_dev, std_dev, tremor_power, curvature_var, total_length]
    except Exception:
        return None


def generate_reference_spiral(cx=230, cy=210, num_turns=4, max_r=180):
    points = []
    theta_max = num_turns * 2 * np.pi
    b = max_r / theta_max
    steps = 400
    for i in range(steps):
        theta = (i / (steps - 1)) * theta_max
        r = b * theta
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        points.append((x, y))
    return points


def spiral_test_enhanced(gui):
    global spiral_conf, raw_spiral_dev

    # Load dataset
    X_data, y_data = _load_spiral_dataset()

    # Create drawing window
    draw_win = tk.Toplevel(gui.root)
    draw_win.title("Spiral Drawing Test")
    draw_win.geometry("500x600")
    draw_win.resizable(False, False)
    draw_win.configure(bg="#0D0D12")

    # Center the window relative to parent
    draw_win.update_idletasks()
    w = 500
    h = 600
    x = gui.root.winfo_x() + (gui.root.winfo_width() // 2) - (w // 2)
    y = gui.root.winfo_y() + (gui.root.winfo_height() // 2) - (h // 2)
    draw_win.geometry(f"{w}x{h}+{x}+{y}")

    tk.Label(
        draw_win, text="🌀 Draw a spiral below",
        font=("Segoe UI", 18, "bold"),
        bg="#0D0D12", fg="#FFFFFF"
    ).pack(pady=(15, 4))
    
    tk.Label(
        draw_win, text="Hold mouse button and draw over the guide spiral from center outward",
        font=("Segoe UI", 11), bg="#0D0D12", fg="#888899"
    ).pack(pady=(0, 10))

    canvas = Canvas(draw_win, width=460, height=420, bg="#1a1a2e",
                    highlightthickness=1, highlightbackground="#3a3a5a")
    canvas.pack(padx=20, pady=(0, 12))

    ref_points = generate_reference_spiral()

    def draw_ref_spiral():
        for i in range(len(ref_points) - 1):
            x0, y0 = ref_points[i]
            x1, y1 = ref_points[i+1]
            canvas.create_line(x0, y0, x1, y1, fill="#444455", width=2, tags="ref_spiral")

    # Draw ref spiral initially
    draw_ref_spiral()

    drawn_points = []
    drawing = [False]

    def on_press(event):
        drawing[0] = True
        drawn_points.clear()
        canvas.delete("all")
        draw_ref_spiral()
        drawn_points.append([event.x, event.y, time.time()])

    def on_drag(event):
        if drawing[0]:
            drawn_points.append([event.x, event.y, time.time()])
            if len(drawn_points) >= 2:
                x0, y0, _ = drawn_points[-2]
                canvas.create_line(x0, y0, event.x, event.y,
                                   fill="#BB86FC", width=3, smooth=True, tags="user_line")

    def on_release(event):
        drawing[0] = False

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    def submit():
        global spiral_conf, raw_spiral_dev
        if len(drawn_points) < 20:
            messagebox.showwarning("Too Short", "Please draw a longer spiral.")
            return

        ref_arr = np.array(ref_points)
        user_arr = np.array([[p[0], p[1]] for p in drawn_points])
        
        deviations = []
        for up in user_arr:
            dists = np.linalg.norm(ref_arr - up, axis=1)
            deviations.append(np.min(dists))
        mean_dev = np.mean(deviations) if deviations else 0.0
        std_dev = np.std(deviations) if deviations else 0.0

        # Save raw deviation
        raw_spiral_dev = mean_dev

        feat = _extract_spiral_features(drawn_points, mean_dev, std_dev)
        if feat is None:
            # Heuristic calculation if feature extraction skips
            dev_score = max(0.0, min(100.0, 100.0 - (mean_dev / TH_SPIRAL_DEV_MAX_PX) * 50.0))
            spiral_conf = dev_score
            draw_win.destroy()
            return

        # Classify comparing against normalized dataset
        if X_data is not None and len(X_data) >= 3:
            k = min(5, len(X_data))
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_data, y_data)
            proba = knn.predict_proba([feat])[0]
            healthy_prob = proba[0] if len(proba) > 1 else (1.0 if y_data[0] == 0 else 0.0)
            
            # Combine classification with direct deviation heuristic for maximum accuracy and feedback
            dev_score = max(0.0, min(100.0, 100.0 - (mean_dev / TH_SPIRAL_DEV_MAX_PX) * 50.0))
            spiral_conf = 0.5 * (healthy_prob * 100.0) + 0.5 * dev_score
        else:
            # Fallback heuristic
            tremor = feat[2]
            score = 100.0
            if tremor > 0.15:
                score -= 30.0
            if mean_dev > TH_SPIRAL_DEV_MAX_PX:
                score -= min(50.0, (mean_dev - TH_SPIRAL_DEV_MAX_PX) * 2.0)
            spiral_conf = max(0.0, min(100.0, score))

        spiral_conf = max(0.0, min(100.0, spiral_conf))
        
        # Save points with schema
        label_str = "healthy" if spiral_conf >= 50.0 else "parkinson"
        filename = f"{label_str}_{int(time.time())}.json"
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spiral_dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        with open(os.path.join(dataset_dir, filename), 'w') as f:
            json.dump(drawn_points, f)

        draw_win.destroy()

    btn = tk.Button(
        draw_win, text="Submit Drawing", font=("Segoe UI", 12, "bold"),
        bg="#9b59b6", fg="#FFFFFF", activebackground="#7d3c98",
        activeforeground="#FFFFFF", relief=tk.FLAT, bd=0,
        cursor="hand2", command=submit, height=2, width=20
    )
    btn.pack(pady=10)
    
    btn.bind("<Enter>", lambda e: btn.config(bg="#7d3c98"))
    btn.bind("<Leave>", lambda e: btn.config(bg="#9b59b6"))

    draw_win.grab_set()
    gui.root.wait_window(draw_win)


def reaction_test_enhanced(gui):
    """Interactive reaction test rendering live webcam feed and spawning touch-targets."""
    global reaction_conf, raw_reaction_lat

    win = tk.Toplevel(gui.root)
    win.title("Reaction Test")
    win.geometry("680x600")
    win.resizable(False, False)
    win.configure(bg="#0D0D12")

    # Center relative to parent
    win.update_idletasks()
    w = 680
    h = 600
    x = gui.root.winfo_x() + (gui.root.winfo_width() // 2) - (w // 2)
    y = gui.root.winfo_y() + (gui.root.winfo_height() // 2) - (h // 2)
    win.geometry(f"{w}x{h}+{x}+{y}")

    tk.Label(
        win, text="[Reaction] Move cursor to touch the targets rapidly!",
        font=("Segoe UI", 16, "bold"),
        bg="#0D0D12", fg="#FFFFFF"
    ).pack(pady=(12, 4))

    canvas = Canvas(win, width=640, height=480, bg="#1a1a2e",
                    highlightthickness=1, highlightbackground="#3a3a5a")
    canvas.pack(padx=20, pady=(0, 15))

    # Try opening webcam index selected by user
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    except Exception as e:
        messagebox.showerror("Camera Error", f"Failed to open camera: {e}")
        reaction_conf = 0.0
        win.destroy()
        return

    NUM_TRIALS = 10
    latencies = []
    mouse_paths = []
    current_trial = [0]
    target_pos = [320, 240]
    target_radius = 25
    trial_start = [0.0]
    current_path = []
    waiting = [True]        # Waiting for countdown delay
    target_active = [False] # Is target currently spawned
    
    status_text = ["Trial 1/10 - Get Ready..."]

    def next_trial():
        target_active[0] = False
        if current_trial[0] >= NUM_TRIALS:
            finish()
            return

        delay = random.uniform(0.7, 2.2)
        waiting[0] = True
        status_text[0] = f"Trial {current_trial[0]+1}/{NUM_TRIALS}\nGet ready..."
        win.after(int(delay * 1000), show_target)

    def show_target():
        if not win.winfo_exists():
            return
        # Spawning coordinates well within 640x480 canvas bounds
        target_pos[0] = random.randint(80, 560)
        target_pos[1] = random.randint(80, 400)
        trial_start[0] = time.time()
        current_path.clear()
        waiting[0] = False
        target_active[0] = True
        status_text[0] = "TOUCH TARGET!"

    def on_motion(event):
        if not waiting[0] and target_active[0]:
            current_path.append((event.x, event.y, time.time()))
            
            # Distance from cursor to target center
            dx = event.x - target_pos[0]
            dy = event.y - target_pos[1]
            if (dx**2 + dy**2) <= (target_radius + 20)**2:  # slightly larger tolerance for easy touch
                target_active[0] = False
                latency = (time.time() - trial_start[0]) * 1000  # ms
                latencies.append(latency)
                mouse_paths.append(list(current_path))
                current_trial[0] += 1
                next_trial()

    canvas.bind("<Motion>", on_motion)

    # Frame update loop
    def update_frame():
        if not win.winfo_exists():
            return
        
        ret, frame = cap.read()
        if ret:
            # mirror effect
            frame = cv2.flip(frame, 1)
            # convert colors
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)
            
            canvas.img_tk = img_tk
            canvas.create_image(0, 0, image=img_tk, anchor="nw", tags="video")

        # Draw target if active
        if target_active[0]:
            x, y = target_pos[0], target_pos[1]
            canvas.create_oval(x - target_radius, y - target_radius,
                               x + target_radius, y + target_radius,
                               fill="#e74c3c", outline="#ff6b6b", width=3, tags="target")
            # Draw center dot
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="white", outline="white", tags="target_dot")

        # Draw status text overlay
        canvas.create_rectangle(10, 10, 220, 50, fill="#0D0D12", outline="#2A2A3A", tags="status_bg")
        canvas.create_text(115, 30, text=status_text[0].replace('\n', ' '), fill="#FFFFFF",
                           font=("Segoe UI", 11, "bold"), tags="status")

        win.after(16, update_frame)

    # Cleanup webcam on window close
    def on_close():
        cap.release()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)

    # Start update loop and trials
    update_frame()
    next_trial()

    def finish():
        global reaction_conf, raw_reaction_lat
        cap.release()
        
        if not latencies:
            reaction_conf = 0.0
            win.destroy()
            return

        mean_lat = np.mean(latencies)
        raw_reaction_lat = mean_lat
        
        # Calculate jerk penalty (smoothness)
        jerk_counts = []
        for path in mouse_paths:
            if len(path) < 4:
                jerk_counts.append(0)
                continue
            xs = [p[0] for p in path]
            dx = np.diff(xs)
            sign_changes = np.sum(np.abs(np.diff(np.sign(dx))) > 0)
            jerk_counts.append(sign_changes)
        avg_jerks = np.mean(jerk_counts) if jerk_counts else 0

        # Score calculations
        # abnormal threshold is mean + 2*std
        abnormal_threshold = TH_REACTION_MEAN_MS + 2.0 * TH_REACTION_STD_MS
        span = abnormal_threshold - TH_REACTION_MEAN_MS
        
        latency_score = 100.0 - ((mean_lat - TH_REACTION_MEAN_MS) / span) * 100.0
        latency_score = max(0.0, min(100.0, latency_score))
        
        jerk_penalty = min(30.0, avg_jerks * 3.0)
        reaction_conf = max(0.0, min(100.0, latency_score - jerk_penalty))

        win.destroy()

    win.grab_set()
    gui.root.wait_window(win)


def run_neuro_tests():
    app = EnhancedNeuroTesterGUI()
    app.root.mainloop()


if __name__ == "__main__":
    run_neuro_tests()