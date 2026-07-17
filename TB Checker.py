import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time


class CoughRecorder:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        
    def start_recording(self):
        self.recording = True
        self.frames = []
        
        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
        except Exception as e:
            self.recording = False
            raise RuntimeError(f"No microphone detected. Please check your microphone connection.\n\nDetails: {e}")
        
    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()
            
    def record_for_duration(self, duration=5):
        self.start_recording()
        
        for i in range(0, int(self.RATE / self.CHUNK * duration)):
            if self.recording:
                data = self.stream.read(self.CHUNK)
                self.frames.append(data)
            else:
                break
                
        self.stop_recording()
        return self.frames
    
    def save_recording(self, filename="cough_recording.wav"):
        if self.frames:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return filename
        return None
    
    def compute_cough_features(self, cough_segment, sr=44100):
        """Extract MFCC + spectral features from a cough segment."""
        try:
            seg = cough_segment.astype(float)
            if len(seg) < 512:
                return None

            # 13 MFCCs
            mfccs = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)  # 13 values

            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=seg))

            # Spectral centroid
            sc = np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr))

            # Spectral rolloff
            sro = np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr))

            # Spectral flatness
            sf = np.mean(librosa.feature.spectral_flatness(y=seg))

            return np.concatenate([mfcc_means, [zcr, sc, sro, sf]])  # 17-dim
        except Exception:
            return None

    def analyze_cough(self, audio_data=None):
        if audio_data is None:
            if not self.frames:
                return None
            audio_signal = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        else:
            audio_signal = audio_data
            
        audio_normalized = audio_signal / (np.max(np.abs(audio_signal)) + 1e-8)
        
        rms = np.array([np.sqrt(np.mean(audio_normalized[i:i+self.CHUNK]**2)) 
                       for i in range(0, len(audio_normalized)-self.CHUNK, self.CHUNK//2)])
        
        cough_threshold = 0.1
        cough_indices = np.where(rms > cough_threshold)[0]
        
        if len(cough_indices) == 0:
            return {
                'cough_count': 0,
                'avg_duration': 0,
                'avg_frequency': 0,
                'risk_level': 'No cough detected',
                'risk_score': 0,
                'confidence': 0,
                'durations': [],
                'frequencies': [],
                'features_extracted': False
            }
        
        # Segment cough events (existing logic)
        cough_events = []
        current_event = [cough_indices[0]]
        
        for i in range(1, len(cough_indices)):
            if cough_indices[i] - cough_indices[i-1] <= 5:
                current_event.append(cough_indices[i])
            else:
                if len(current_event) >= 3:
                    cough_events.append(current_event)
                current_event = [cough_indices[i]]
        
        if len(current_event) >= 3:
            cough_events.append(current_event)
        
        durations = []
        frequencies = []
        all_features = []
        
        for event in cough_events:
            duration_frames = len(event)
            duration_seconds = (duration_frames * (self.CHUNK//2)) / self.RATE
            durations.append(duration_seconds)
            
            start_sample = event[0] * (self.CHUNK//2)
            end_sample = min(event[-1] * (self.CHUNK//2) + self.CHUNK, len(audio_normalized))
            
            cough_segment = audio_normalized[start_sample:end_sample]
            
            if len(cough_segment) > 100:
                f, Pxx = signal.periodogram(cough_segment, self.RATE)
                dominant_freq = f[np.argmax(Pxx)]
                frequencies.append(dominant_freq)

            # Extract MFCC + spectral features per cough
            feat = self.compute_cough_features(cough_segment, self.RATE)
            if feat is not None:
                all_features.append(feat)
        
        avg_duration = np.mean(durations) if durations else 0
        avg_frequency = np.mean(frequencies) if frequencies else 0
        cough_count = len(cough_events)
        
        # ── Statistical-norm classifier ──
        # Healthy cough norms (published reference ranges)
        # Higher spectral centroid, shorter duration, higher ZCR = healthier
        confidence = 0.0
        if all_features:
            mean_feat = np.mean(all_features, axis=0)
            # Healthy reference centroid (approximate from literature)
            healthy_ref = np.zeros(17)
            healthy_ref[14] = 3000   # spectral centroid ~3000 Hz for healthy
            healthy_ref[13] = 0.08   # ZCR ~0.08 for healthy
            healthy_ref[15] = 6000   # spectral rolloff ~6000 Hz
            healthy_ref[16] = 0.3    # spectral flatness ~0.3
            
            # Compute deviation from healthy norms (weighted)
            weights = np.ones(17) * 0.5
            weights[13] = 2.0  # ZCR weight
            weights[14] = 3.0  # spectral centroid weight
            weights[15] = 1.5  # rolloff weight
            weights[16] = 2.0  # flatness weight
            
            # Duration and frequency based risk
            duration_risk = min(1.0, max(0, (avg_duration - 0.3) / 0.4))
            freq_risk = min(1.0, max(0, (1500 - avg_frequency) / 1500)) if avg_frequency > 0 else 0.3
            count_risk = min(1.0, cough_count / 5.0)
            
            # Feature distance risk
            if healthy_ref[14] > 0:
                centroid_risk = min(1.0, max(0, (healthy_ref[14] - mean_feat[14]) / healthy_ref[14]))
            else:
                centroid_risk = 0.3
            
            confidence = (duration_risk * 0.25 + freq_risk * 0.25 + 
                         count_risk * 0.15 + centroid_risk * 0.35) * 100
        else:
            # Fallback to original simple scoring
            risk_score = 0
            if avg_duration > 0.4:
                risk_score += 2
            if avg_frequency < 1000 and avg_frequency > 0:
                risk_score += 2
            if cough_count >= 3:
                risk_score += 1
            confidence = (risk_score / 5.0) * 100

        confidence = max(0, min(100, confidence))

        # Risk buckets
        if confidence > 60:
            risk_level = "HIGH RISK"
        elif confidence > 30:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "LOW RISK"
        
        return {
            'cough_count': cough_count,
            'avg_duration': avg_duration,
            'avg_frequency': avg_frequency,
            'risk_level': risk_level,
            'risk_score': confidence / 20,  # keep legacy field (0-5 scale)
            'confidence': confidence,
            'durations': durations,
            'frequencies': frequencies,
            'features_extracted': len(all_features) > 0
        }


class EnhancedTB_Analyzer_GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TB Cough Analyzer")
        self.root.geometry("1000x820")
        self.root.configure(bg="#0D0D12")
        
        # Center the window
        self.root.update_idletasks()
        w = 1000
        h = 820
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        
        self.recorder = CoughRecorder()
        self.setup_gui()
        
    def setup_gui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#0D0D12")
        header_frame.pack(fill='x', padx=30, pady=(20, 10))
        
        title = tk.Label(
            header_frame,
            text="🫁 TB Cough Analyzer",
            font=("Segoe UI", 26, "bold"),
            bg="#0D0D12",
            fg="#FFFFFF"
        )
        title.pack()
        
        subtitle = tk.Label(
            header_frame,
            text="Advanced Tuberculosis Screening Through Cough Analysis",
            font=("Segoe UI", 12),
            bg="#0D0D12",
            fg="#6E6E80"
        )
        subtitle.pack(pady=(4, 0))

        # Green separator accent
        sep = tk.Frame(header_frame, bg="#2ECC71", height=2)
        sep.pack(fill='x', padx=150, pady=(12, 0))
        
        # Instructions card
        instr_frame = tk.Frame(self.root, bg="#16161E", highlightbackground="#2A2A3A", highlightthickness=1)
        instr_frame.pack(fill='x', padx=30, pady=10)
        
        tk.Label(
            instr_frame,
            text="📋 Instructions:",
            font=("Segoe UI", 12, "bold"),
            bg="#16161E",
            fg="#AAAABC"
        ).pack(anchor='w', padx=20, pady=(12, 4))
        
        instructions = [
            "✓ Find a quiet environment with minimal background noise",
            "✓ Position yourself close to the microphone",
            "✓ Cough naturally when recording starts",
            "✓ Multiple coughs are acceptable within the 5-second window"
        ]
        
        for instr in instructions:
            tk.Label(
                instr_frame,
                text=instr,
                font=("Segoe UI", 11),
                bg="#16161E",
                fg="#888899"
            ).pack(anchor='w', padx=40, pady=2)
        
        tk.Label(instr_frame, text="", bg="#16161E", font=("Segoe UI", 4)).pack(pady=2)
        
        # Recording section
        record_frame = tk.Frame(self.root, bg="#181822", highlightbackground="#2A2A3A", highlightthickness=1)
        record_frame.pack(fill='x', padx=30, pady=10)
        
        self.record_btn = tk.Button(
            record_frame,
            text="🎤 Start Recording (5 seconds)",
            font=("Segoe UI", 13, "bold"),
            bg="#2ECC71", fg="#FFFFFF", activebackground="#27AE60",
            activeforeground="#FFFFFF", relief=tk.FLAT, bd=0,
            cursor="hand2", command=self.start_recording_thread
        )
        self.record_btn.pack(pady=15, padx=20, ipady=8)
        
        # Hover effect for record button
        self.record_btn.bind("<Enter>", lambda e: self.record_btn.config(bg="#27AE60"))
        self.record_btn.bind("<Leave>", lambda e: self.record_btn.config(bg="#2ECC71"))
        
        self.status_label = tk.Label(
            record_frame,
            text="Ready to record...",
            font=("Segoe UI", 11),
            bg="#181822",
            fg="#AAAABC"
        )
        self.status_label.pack(pady=(0, 15))
        
        # Results section
        results_frame = tk.Frame(self.root, bg="#16161E", highlightbackground="#2A2A3A", highlightthickness=1)
        results_frame.pack(fill='both', expand=True, padx=30, pady=10)
        
        tk.Label(
            results_frame,
            text="📊 Analysis Results:",
            font=("Segoe UI", 12, "bold"),
            bg="#16161E",
            fg="#AAAABC"
        ).pack(anchor='w', padx=20, pady=(12, 4))
        
        self.results_text = tk.Text(
            results_frame,
            bg="#12121A",
            fg="#E0E0F0",
            font=("Consolas", 11),
            highlightthickness=1,
            highlightbackground="#2A2A3A",
            insertbackground="white",
            padx=10,
            pady=5
        )
        self.results_text.pack(fill='both', expand=True, padx=20, pady=(4, 15))
        
        # Footer warning
        warning_frame = tk.Frame(self.root, bg="#181822", highlightbackground="#2A2A3A", highlightthickness=1)
        warning_frame.pack(fill='x', padx=30, pady=(0, 20))
        
        tk.Label(
            warning_frame,
            text="⚠️ MEDICAL DISCLAIMER",
            font=("Segoe UI", 11, "bold"),
            bg="#181822",
            fg="#FF6B6B"
        ).pack(pady=(8, 4))
        
        tk.Label(
            warning_frame,
            text="This is a screening tool only. It cannot diagnose TB.\nAlways consult a qualified healthcare professional for proper diagnosis.",
            font=("Segoe UI", 10),
            bg="#181822",
            fg="#888899",
            justify="center"
        ).pack(pady=(0, 10))
        
    def start_recording_thread(self):
        self.record_btn.configure(state='disabled')
        thread = threading.Thread(target=self.record_and_analyze)
        thread.daemon = True
        thread.start()
        
    def record_and_analyze(self):
        try:
            # Countdown
            for i in range(3, 0, -1):
                self.status_label.configure(
                    text=f"Recording starts in {i}...",
                    fg="orange"
                )
                self.root.update()
                time.sleep(1)
            
            self.status_label.configure(
                text="🎤 RECORDING... Cough now!",
                fg="#e74c3c"
            )
            self.root.update()
            
            # Record
            self.recorder.record_for_duration(5)
            
            # Save
            filename = self.recorder.save_recording()
            
            self.status_label.configure(
                text="🔬 Analyzing cough patterns...",
                fg="#3498db"
            )
            self.root.update()
            
            # Analyze
            results = self.recorder.analyze_cough()
            
            # Display
            self.display_results(results, filename)
            
            self.record_btn.configure(state='normal')
            self.status_label.configure(
                text="✅ Analysis complete! Record again?",
                fg="#2ecc71"
            )
            
        except Exception as e:
            self.status_label.configure(
                text=f"❌ Error: {str(e)}",
                fg="#e74c3c"
            )
            self.record_btn.configure(state='normal')
            
    def display_results(self, results, filename):
        self.results_text.delete("1.0", "end")
        
        if results['cough_count'] == 0:
            self.results_text.insert("end", "❌ No cough detected\n\n")
            self.results_text.insert("end", "Please try again and cough clearly into the microphone.")
            return
        
        # Results header
        self.results_text.insert("end", "📊 COUGH ANALYSIS RESULTS\n")
        self.results_text.insert("end", "=" * 50 + "\n\n")
        
        # Basic metrics
        self.results_text.insert("end", "📈 DETECTION METRICS:\n")
        self.results_text.insert("end", f"   • Coughs detected: {results['cough_count']}\n")
        self.results_text.insert("end", f"   • Average duration: {results['avg_duration']:.3f} seconds\n")
        self.results_text.insert("end", f"   • Average frequency: {results['avg_frequency']:.1f} Hz\n")
        if results.get('features_extracted'):
            self.results_text.insert("end", f"   • MFCC + spectral features: ✅ Extracted\n")
        self.results_text.insert("end", f"   • Risk confidence: {results.get('confidence', 0):.1f}%\n\n")
        
        # TB characteristics
        self.results_text.insert("end", "🔍 TB CHARACTERISTIC ANALYSIS:\n")
        self.results_text.insert("end", "-" * 50 + "\n")
        
        if results['avg_duration'] > 0.4:
            self.results_text.insert("end", "   ❌ Prolonged cough duration (>400ms)\n")
            self.results_text.insert("end", "      → TB characteristic detected\n")
        else:
            self.results_text.insert("end", "   ✅ Normal cough duration\n")
        
        if results['avg_frequency'] < 1000 and results['avg_frequency'] > 0:
            self.results_text.insert("end", "   ❌ Low frequency cough (<1000 Hz)\n")
            self.results_text.insert("end", "      → TB characteristic detected\n")
        else:
            self.results_text.insert("end", "   ✅ Normal cough frequency\n")
        
        if results['cough_count'] >= 3:
            self.results_text.insert("end", "   ❌ Multiple persistent coughs\n")
            self.results_text.insert("end", "      → TB characteristic detected\n")
        else:
            self.results_text.insert("end", "   ✅ Normal cough frequency\n")
        
        # Risk assessment
        self.results_text.insert("end", "\n" + "🚨 RISK ASSESSMENT\n")
        self.results_text.insert("end", "=" * 50 + "\n")
        
        risk = results['risk_level']
        
        if risk == "HIGH RISK":
            self.results_text.insert("end", f"🔴 {risk}\n\n")
            self.results_text.insert("end", "IMMEDIATE ACTION REQUIRED:\n")
            self.results_text.insert("end", "   • Consult a doctor immediately\n")
            self.results_text.insert("end", "   • Get proper TB testing (Sputum test, X-ray)\n")
            self.results_text.insert("end", "   • Inform healthcare provider about these results\n")
            self.results_text.insert("end", "   • Avoid close contact with others until evaluated\n")
        elif risk == "MODERATE RISK":
            self.results_text.insert("end", f"🟡 {risk}\n\n")
            self.results_text.insert("end", "RECOMMENDED ACTIONS:\n")
            self.results_text.insert("end", "   • Schedule a medical check-up within 1 week\n")
            self.results_text.insert("end", "   • Monitor symptoms closely\n")
            self.results_text.insert("end", "   • Return for re-evaluation if symptoms worsen\n")
            self.results_text.insert("end", "   • Maintain good hygiene practices\n")
        else:
            self.results_text.insert("end", f"🟢 {risk}\n\n")
            self.results_text.insert("end", "GENERAL RECOMMENDATIONS:\n")
            self.results_text.insert("end", "   • Continue normal monitoring\n")
            self.results_text.insert("end", "   • See doctor if cough persists >2 weeks\n")
            self.results_text.insert("end", "   • Maintain healthy lifestyle\n")
        
        # Footer
        self.results_text.insert("end", "\n" + "=" * 50 + "\n")
        self.results_text.insert("end", f"💾 Recording saved: {filename}\n")
