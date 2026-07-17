import sys
sys.modules['tensorflow'] = None

import cv2
import numpy as np
from datetime import datetime
import json
import os
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# Camera index from launcher (env var) or default 0
CAMERA_INDEX = int(os.environ.get("SEHATSAATHI_CAMERA_INDEX", "0"))

class EnhancedAnemiaScanner:
    """Enhanced Anemia Scanner with modern OpenCV UI"""
    
    def __init__(self):
        self.cap = None
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.thresholds = {
            'palm': {
                'healthy_red_ratio': 0.35,
                'anemic_red_ratio': 0.28,
                'saturation_min': 20,
                'redness_index_min': 0.08,
            },
            'nail': {
                'healthy_pink_score': 25,
                'anemic_pink_score': 10,
                'whiteness_threshold': 210,
                'redness_index_min': 0.06,
            },
            'conjunctiva': {
                'healthy_red_ratio': 0.38,
                'anemic_red_ratio': 0.30,
                'red_channel_min': 160,
                'pallor_threshold': 0.40,
            }
        }
        
        self.results_log = []
        self.PALM_CENTER_LANDMARKS = [0, 5, 9, 13, 17]
        self.FINGERTIP_LANDMARKS = [4, 8, 12, 16, 20]
        self.LOWER_EYELID_LANDMARKS = [145, 146, 147, 148, 149, 150, 151, 152, 
                                      374, 375, 376, 377, 378, 379, 380, 381]
        # Sclera landmarks for white-balance reference
        self.SCLERA_LANDMARKS = [33, 133, 362, 263]
        
        # UI Colors
        self.colors = {
            'bg': (20, 20, 30),
            'panel': (35, 35, 50),
            'primary': (0, 200, 255),
            'success': (0, 255, 150),
            'warning': (255, 180, 0),
            'danger': (255, 60, 80),
            'text': (255, 255, 255),
            'text_dim': (180, 180, 200)
        }
    
    def draw_modern_ui_background(self, frame, h, w):
        """Draw modern gradient background"""
        overlay = frame.copy()
        
        # Gradient effect
        for i in range(h):
            alpha = i / h
            color = tuple(int(self.colors['bg'][j] * (1 - alpha * 0.3)) for j in range(3))
            cv2.line(overlay, (0, i), (w, i), color, 1)
        
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame
    
    def draw_modern_panel(self, frame, x, y, width, height, title="", color=None):
        """Draw a modern panel with rounded corners"""
        if color is None:
            color = self.colors['panel']
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['primary'], 2)
        
        # Title bar
        if title:
            title_h = 40
            cv2.rectangle(frame, (x, y), (x + width, y + title_h), self.colors['primary'], -1)
            cv2.putText(frame, title, (x + 15, y + 27), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['bg'], 2)
    
    def draw_progress_bar(self, frame, x, y, width, progress, label=""):
        """Draw modern progress bar"""
        height = 25
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 50), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.colors['primary'], 1)
        
        # Progress
        fill_width = int(width * progress)
        if fill_width > 0:
            cv2.rectangle(frame, (x + 2, y + 2), 
                         (x + fill_width - 2, y + height - 2), 
                         self.colors['success'], -1)
        
        # Label
        if label:
            text = f"{label} {int(progress * 100)}%"
            cv2.putText(frame, text, (x + 10, y + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def extract_palm_region(self, frame, hand_landmarks, image_shape):
        """Extract palm region (unchanged logic)"""
        try:
            h, w, _ = image_shape
            palm_points = []
            for idx in self.PALM_CENTER_LANDMARKS:
                landmark = hand_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                palm_points.append((x, y))
            
            xs = [p[0] for p in palm_points]
            ys = [p[1] for p in palm_points]
            
            x_min, x_max = max(0, min(xs)), min(w, max(xs))
            y_min, y_max = max(0, min(ys)), min(h, max(ys))
            
            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            if x_max <= x_min or y_max <= y_min:
                return None, None
                
            roi = frame[y_min:y_max, x_min:x_max]
            
            if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                return None, None
            
            return roi, (x_min, y_min, x_max, y_max)
        except Exception as e:
            return None, None
    
    def extract_nail_regions(self, frame, hand_landmarks, image_shape):
        """Extract nail regions (unchanged logic)"""
        try:
            h, w, _ = image_shape
            nail_rois = []
            nail_coords = []
            
            for tip_idx in self.FINGERTIP_LANDMARKS:
                landmark = hand_landmarks.landmark[tip_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                nail_size = 15
                x1 = max(0, x - nail_size)
                y1 = max(0, y - nail_size)
                x2 = min(w, x + nail_size)
                y2 = min(h, y + nail_size)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                nail_roi = frame[y1:y2, x1:x2]
                
                if nail_roi.size > 0 and nail_roi.shape[0] > 5 and nail_roi.shape[1] > 5:
                    nail_rois.append(nail_roi)
                    nail_coords.append((x1, y1, x2, y2))
            
            if len(nail_rois) < 2:
                return None, None
            
            combined_roi = np.vstack([cv2.resize(roi, (30, 30)) for roi in nail_rois[:3]])
            return combined_roi, nail_coords
        except Exception as e:
            return None, None
    
    def extract_conjunctiva_region(self, frame, face_landmarks, image_shape):
        """Extract conjunctiva region (unchanged logic)"""
        try:
            h, w, _ = image_shape
            
            eyelid_points = []
            for idx in self.LOWER_EYELID_LANDMARKS:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    eyelid_points.append((x, y))
            
            if len(eyelid_points) < 6:
                return None, None
            
            xs = [p[0] for p in eyelid_points]
            ys = [p[1] for p in eyelid_points]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            margin = 3
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            if x_max <= x_min or y_max <= y_min:
                return None, None
                
            roi = frame[y_min:y_max, x_min:x_max]
            
            if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                return None, None
            
            return roi, (x_min, y_min, x_max, y_max)
        except Exception as e:
            return None, None

    def extract_sclera_region(self, frame, face_landmarks, image_shape):
        """Extract sclera region for white-balance reference."""
        try:
            h, w, _ = image_shape
            points = []
            for idx in self.SCLERA_LANDMARKS:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    points.append((int(lm.x * w), int(lm.y * h)))
            if len(points) < 2:
                return None
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = max(0, min(xs) - 2), min(w, max(xs) + 2)
            y_min, y_max = max(0, min(ys) - 2), min(h, max(ys) + 2)
            if x_max <= x_min or y_max <= y_min:
                return None
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
                return None
            return roi
        except Exception:
            return None
    
    def advanced_color_analysis(self, roi):
        """Color analysis (unchanged)"""
        try:
            if roi.size == 0:
                return self.get_default_analysis()
            
            b, g, r = cv2.split(roi)
            r_mean, r_std = np.mean(r), np.std(r)
            g_mean, g_std = np.mean(g), np.std(g)
            b_mean, b_std = np.mean(b), np.std(b)
            
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s_mean = np.mean(s)
            v_mean = np.mean(v)
            
            total = r_mean + g_mean + b_mean + 1e-6
            red_ratio = r_mean / total
            redness_index = (r_mean - g_mean) / (r_mean + g_mean + 1e-6)
            pallor_index = (255 - r_mean) / 255.0
            pink_score = r_mean - g_mean
            color_variance = np.mean([r_std, g_std, b_std])
            
            return {
                'rgb': (r_mean, g_mean, b_mean),
                'saturation': s_mean,
                'brightness': v_mean,
                'red_ratio': red_ratio,
                'redness_index': redness_index,
                'pallor_index': pallor_index,
                'pink_score': pink_score,
                'color_variance': color_variance,
                'valid_analysis': True
            }
        except Exception as e:
            return self.get_default_analysis()
    
    def get_default_analysis(self):
        return {
            'rgb': (128, 128, 128),
            'saturation': 50,
            'brightness': 128,
            'red_ratio': 0.33,
            'redness_index': 0.0,
            'pallor_index': 0.5,
            'pink_score': 0,
            'color_variance': 0,
            'valid_analysis': False
        }
    
    def assess_palm_with_ml(self, analysis):
        """Palm assessment (unchanged)"""
        try:
            if not analysis.get('valid_analysis', True):
                return 0, ["Invalid analysis"], 0.1
                
            score = 0
            warnings = []
            
            if analysis['red_ratio'] < self.thresholds['palm']['anemic_red_ratio']:
                score += 1.5
                warnings.append("Mild palm pallor")
            elif analysis['red_ratio'] < self.thresholds['palm']['healthy_red_ratio']:
                score += 0.5
                warnings.append("Slight color reduction")
            
            if analysis['saturation'] < self.thresholds['palm']['saturation_min']:
                score += 0.5
                warnings.append("Moderate saturation")
            
            confidence = max(0.5, 1.0 - (analysis['color_variance'] / 100.0))
            
            return min(score, 5), warnings, confidence
        except Exception as e:
            return 0, ["Assessment error"], 0.1
    
    def assess_nail_with_ml(self, analysis):
        """Nail assessment (unchanged)"""
        try:
            if not analysis.get('valid_analysis', True):
                return 0, ["Invalid analysis"], 0.1
                
            score = 0
            warnings = []
            
            if analysis['pink_score'] < self.thresholds['nail']['anemic_pink_score']:
                score += 1.5
                warnings.append("Pale nail beds")
            elif analysis['pink_score'] < self.thresholds['nail']['healthy_pink_score']:
                score += 0.5
                warnings.append("Slight nail pallor")
            
            confidence = 0.7 if analysis['color_variance'] < 30 else 0.5
            
            return min(score, 5), warnings, confidence
        except Exception as e:
            return 0, ["Assessment error"], 0.1
    
    def assess_conjunctiva_with_ml(self, analysis, sclera_roi=None):
        """Conjunctiva assessment using clinical pallor ratio with sclera white-balance."""
        try:
            if not analysis.get('valid_analysis', True):
                return 0, ["Invalid analysis"], 0.1

            r_mean, g_mean, b_mean = analysis['rgb']

            # Compute pallor ratio: R / (G + B)
            pallor_ratio = r_mean / (g_mean + b_mean + 1e-6)

            # White-balance normalize against sclera if available
            if sclera_roi is not None and sclera_roi.size > 0:
                sb, sg, sr = cv2.split(sclera_roi)
                sclera_r = np.mean(sr)
                sclera_gb = np.mean(sg) + np.mean(sb)
                sclera_ratio = sclera_r / (sclera_gb + 1e-6)
                if sclera_ratio > 0:
                    pallor_ratio = pallor_ratio / sclera_ratio  # normalized

            # Severity bands
            score = 0
            warnings = []

            if pallor_ratio > 0.85:
                score = 0
                warnings.append("Conjunctiva appears healthy")
            elif pallor_ratio > 0.70:
                score = 1.5
                warnings.append("Mild conjunctival pallor detected")
            elif pallor_ratio > 0.55:
                score = 3.0
                warnings.append("Moderate conjunctival pallor detected")
            else:
                score = 4.5
                warnings.append("Severe conjunctival pallor detected")

            confidence = 0.8
            return min(score, 5), warnings, confidence
        except Exception as e:
            return 0, ["Assessment error"], 0.1
    
    def combined_ml_assessment(self, results):
        """Combined assessment (unchanged)"""
        try:
            total_score = 0
            all_warnings = []
            scan_count = 0
            
            for scan_type, data in results.items():
                if data is not None:
                    score = data['risk_score']
                    total_score += score
                    all_warnings.extend(data['warnings'])
                    scan_count += 1
            
            if scan_count == 0:
                return None
            
            avg_score = total_score / scan_count
            
            if avg_score >= 3.5:
                risk_level = "SEVERE"
                recommendation = "Consult a doctor immediately for blood work"
                color = self.colors['danger']
            elif avg_score >= 2.0:
                risk_level = "MODERATE"
                recommendation = "Consider a medical check-up soon"
                color = self.colors['warning']
            elif avg_score >= 1.0:
                risk_level = "MILD"
                recommendation = "Monitor nutrition and iron intake"
                color = self.colors['primary']
            else:
                risk_level = "HEALTHY" 
                recommendation = "Indicators appear normal"
                color = self.colors['success']
            
            assessment = {
                'risk_level': risk_level,
                'avg_score': round(avg_score, 2),
                'weighted_score': round(avg_score, 2),
                'confidence': 0.7,
                'scans_completed': scan_count,
                'warnings': list(set(all_warnings)),
                'recommendation': recommendation,
                'color': color,
                'results': results
            }
            
            return assessment
            
        except Exception as e:
            return None

    def visualize_combined(self, frame, assessment):
        """Enhanced combined results visualization"""
        h, w = frame.shape[:2]
        
        # Create modern results panel
        panel = np.zeros((600, w, 3), dtype=np.uint8)
        panel[:] = self.colors['bg']
        
        # Main title panel
        self.draw_modern_panel(panel, 20, 20, w - 40, 80, "ANEMIA SCREENING RESULTS")
        
        # Risk level with color coding
        risk = assessment['risk_level']
        color = assessment['color']
        
        cv2.putText(panel, f"RISK LEVEL: {risk}", (50, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Score visualization
        score_y = 200
        cv2.putText(panel, "ASSESSMENT SCORES:", (50, score_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['primary'], 2)
        
        score_y += 40
        cv2.putText(panel, f"Average Score: {assessment['avg_score']:.2f} / 5.0", 
                   (70, score_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        score_y += 30
        cv2.putText(panel, f"Scans Completed: {assessment['scans_completed']} / 3", 
                   (70, score_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_dim'], 1)
        
        # Progress bar for score
        score_y += 40
        self.draw_progress_bar(panel, 70, score_y, w - 140, 
                             assessment['avg_score'] / 5.0, "Health Score")
        
        # Key findings
        findings_y = score_y + 60
        cv2.putText(panel, "KEY FINDINGS:", (50, findings_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['primary'], 2)
        
        findings_y += 35
        if assessment['warnings']:
            for i, warning in enumerate(assessment['warnings'][:4]):
                cv2.putText(panel, f"• {warning}", (70, findings_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                findings_y += 30
        else:
            cv2.putText(panel, "• No significant concerns detected", (70, findings_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['success'], 1)
        
        # Recommendation box
        rec_y = findings_y + 40
        self.draw_modern_panel(panel, 40, rec_y, w - 80, 100, "RECOMMENDATION")
        
        rec_text_y = rec_y + 60
        cv2.putText(panel, assessment['recommendation'], (60, rec_text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        result = np.vstack([frame, panel])
        return result

    def run_scanner(self):
        """Main scanning loop wrapped in a beautiful dark Tkinter GUI matching main.py"""
        self.root = tk.Tk()
        self.root.title("Anemia Conjunctiva Scanner")
        self.root.geometry("1150x700")
        self.root.configure(bg="#0D0D12")
        self.root.resizable(False, False)
        
        # Center the window
        self.root.update_idletasks()
        w = 1150
        h = 700
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        
        # State variables
        self.results = {'palm': None, 'nail': None, 'conjunctiva': None}
        self.scan_sequence = ['palm', 'nail', 'conjunctiva']
        self.current_index = 0
        
        # Open camera
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "No camera detected. Please check camera connection.")
                return
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not access camera: {e}")
            return
            
        self.setup_gui()
        
        # Bind keys
        self.root.bind("<space>", lambda e: self.capture_current_scan())
        self.root.bind("<r>", lambda e: self.reset_scanner())
        self.root.bind("<c>", lambda e: self.show_combined_results())
        
        # Start camera update loop
        self.update_camera()
        self.root.mainloop()
        
        # Release resources when done
        self.cap.release()
        cv2.destroyAllWindows()

    def setup_gui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#0D0D12")
        header_frame.pack(fill='x', padx=30, pady=(20, 10))
        
        title = tk.Label(
            header_frame,
            text="🩸 Anemia Conjunctiva Scanner",
            font=("Segoe UI", 26, "bold"),
            bg="#0D0D12",
            fg="#FFFFFF"
        )
        title.pack()
        
        subtitle = tk.Label(
            header_frame,
            text="Multispectral Pallor Assessment Screen",
            font=("Segoe UI", 12),
            bg="#0D0D12",
            fg="#6E6E80"
        )
        subtitle.pack(pady=(4, 0))

        # Red separator accent
        sep = tk.Frame(header_frame, bg="#FF6B6B", height=2)
        sep.pack(fill='x', padx=150, pady=(12, 0))

        # Main split container
        main_container = tk.Frame(self.root, bg="#0D0D12")
        main_container.pack(fill='both', expand=True, padx=30, pady=10)

        # Left panel: Camera stream
        cam_frame = tk.Frame(main_container, bg="#16161E", highlightbackground="#2A2A3A", highlightthickness=1)
        cam_frame.pack(side='left', padx=(0, 15), pady=10)
        
        self.cam_label = tk.Label(cam_frame, bg="#12121A", width=640, height=480)
        self.cam_label.pack(padx=10, pady=10)

        # Right panel: Controls and details
        right_panel = tk.Frame(main_container, bg="#0D0D12")
        right_panel.pack(side='right', fill='both', expand=True, pady=10)

        # Step instructions panel
        self.instr_frame = tk.Frame(right_panel, bg="#16161E", highlightbackground="#2A2A3A", highlightthickness=1, padx=15, pady=15)
        self.instr_frame.pack(fill='x', pady=(0, 10))

        self.step_title_lbl = tk.Label(
            self.instr_frame,
            text="Current Step: Palm Scan",
            font=("Segoe UI", 13, "bold"),
            bg="#16161E",
            fg="#FF6B6B"
        )
        self.step_title_lbl.pack(anchor='w')

        self.step_desc_lbl = tk.Label(
            self.instr_frame,
            text="Position your palm flat in the camera frame. Try to match the center of the screen.",
            font=("Segoe UI", 11),
            bg="#16161E",
            fg="#888899",
            wraplength=350,
            justify="left"
        )
        self.step_desc_lbl.pack(anchor='w', pady=(8, 0))

        # Real-time detection status indicator
        self.detection_status_var = tk.StringVar(value="⏳ SCANNING FOR TARGET...")
        self.detection_status_lbl = tk.Label(
            right_panel,
            textvariable=self.detection_status_var,
            font=("Segoe UI", 12, "bold"),
            bg="#0D0D12",
            fg="#FF6B6B"
        )
        self.detection_status_lbl.pack(anchor='w', pady=5)

        # Checklist panel
        checklist_frame = tk.Frame(right_panel, bg="#181822", highlightbackground="#2A2A3A", highlightthickness=1, padx=15, pady=15)
        checklist_frame.pack(fill='x', pady=10)

        self.check_palm_lbl = tk.Label(checklist_frame, text="○ Palm Scan", font=("Segoe UI", 11), bg="#181822", fg="#888899")
        self.check_palm_lbl.pack(anchor='w', pady=2)
        
        self.check_nail_lbl = tk.Label(checklist_frame, text="○ Nail Scan", font=("Segoe UI", 11), bg="#181822", fg="#888899")
        self.check_nail_lbl.pack(anchor='w', pady=2)

        self.check_conjunctiva_lbl = tk.Label(checklist_frame, text="○ Conjunctiva Scan", font=("Segoe UI", 11), bg="#181822", fg="#888899")
        self.check_conjunctiva_lbl.pack(anchor='w', pady=2)

        # Buttons frame
        btn_frame = tk.Frame(right_panel, bg="#0D0D12")
        btn_frame.pack(fill='x', pady=10)

        self.cap_btn = tk.Button(
            btn_frame,
            text="📸 Capture Scan",
            font=("Segoe UI", 11, "bold"),
            bg="#FF6B6B", fg="#FFFFFF", activebackground="#E74C3C",
            activeforeground="#FFFFFF", relief=tk.FLAT, bd=0,
            cursor="hand2", command=self.capture_current_scan, width=15
        )
        self.cap_btn.pack(side='left', padx=(0, 10), ipady=6)
        self.cap_btn.bind("<Enter>", lambda e: self.cap_btn.config(bg="#E74C3C"))
        self.cap_btn.bind("<Leave>", lambda e: self.cap_btn.config(bg="#FF6B6B"))

        self.reset_btn = tk.Button(
            btn_frame,
            text="🔄 Reset",
            font=("Segoe UI", 11, "bold"),
            bg="#22222E", fg="#FF6B6B", activebackground="#FF6B6B",
            activeforeground="#0D0D12", relief=tk.FLAT, bd=0,
            cursor="hand2", command=self.reset_scanner, width=8
        )
        self.reset_btn.pack(side='left', padx=10, ipady=6)
        self.reset_btn.bind("<Enter>", lambda e: self.reset_btn.config(bg="#FF6B6B", fg="#0D0D12"))
        self.reset_btn.bind("<Leave>", lambda e: self.reset_btn.config(bg="#22222E", fg="#FF6B6B"))

        self.results_btn = tk.Button(
            btn_frame,
            text="📊 Calculate Risk",
            font=("Segoe UI", 11, "bold"),
            bg="#2ECC71", fg="#FFFFFF", activebackground="#27AE60",
            activeforeground="#FFFFFF", relief=tk.FLAT, bd=0,
            cursor="hand2", command=self.show_combined_results, width=15
        )
        self.results_btn.pack(side='left', padx=10, ipady=6)
        self.results_btn.bind("<Enter>", lambda e: self.results_btn.config(bg="#27AE60"))
        self.results_btn.bind("<Leave>", lambda e: self.results_btn.config(bg="#2ECC71"))

        # Results terminal
        self.results_text = tk.Text(
            right_panel,
            height=6,
            bg="#12121A",
            fg="#E0E0F0",
            font=("Consolas", 10),
            highlightthickness=1,
            highlightbackground="#2A2A3A",
            insertbackground="white",
            padx=10,
            pady=5
        )
        self.results_text.pack(fill='both', expand=True, pady=10)

    def update_step_instructions(self):
        mode = self.scan_sequence[self.current_index]
        self.step_title_lbl.config(text=f"Current Step: {mode.capitalize()} Scan")
        if mode == 'palm':
            self.step_desc_lbl.config(text="Position your palm flat in the camera frame. Try to match the center of the screen.")
        elif mode == 'nail':
            self.step_desc_lbl.config(text="Position your hand so that the fingernails are visible in the frame.")
        elif mode == 'conjunctiva':
            self.step_desc_lbl.config(text="Look directly into the camera and pull down your lower eyelid slightly.")

    def update_checklist(self):
        if self.results['palm'] is not None:
            self.check_palm_lbl.config(text="✓ Palm Scan", fg="#2ECC71")
        else:
            self.check_palm_lbl.config(text="○ Palm Scan", fg="#888899")

        if self.results['nail'] is not None:
            self.check_nail_lbl.config(text="✓ Nail Scan", fg="#2ECC71")
        else:
            self.check_nail_lbl.config(text="○ Nail Scan", fg="#888899")

        if self.results['conjunctiva'] is not None:
            self.check_conjunctiva_lbl.config(text="✓ Conjunctiva Scan", fg="#2ECC71")
        else:
            self.check_conjunctiva_lbl.config(text="○ Conjunctiva Scan", fg="#888899")

    def update_results_text(self, text):
        self.results_text.insert("end", text)
        self.results_text.see("end")

    def update_camera(self):
        if not self.root.winfo_exists():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(15, self.update_camera)
            return
            
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mode = self.scan_sequence[self.current_index]
        display = frame.copy()
        
        self.roi = None
        self.coords = None
        self.sclera_roi = None
        
        # Mode-specific detection
        if mode == 'palm':
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(display, hand_landmarks, 
                                           self.mp_hands.HAND_CONNECTIONS)
                self.roi, self.coords = self.extract_palm_region(frame, hand_landmarks, frame.shape)
        
        elif mode == 'nail':
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(display, hand_landmarks, 
                                           self.mp_hands.HAND_CONNECTIONS)
                self.roi, self.coords = self.extract_nail_regions(frame, hand_landmarks, frame.shape)
        
        elif mode == 'conjunctiva':
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                for idx in [145, 155, 159, 386, 374, 380]:
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(display, (x, y), 3, (0, 255, 150), -1)
                self.roi, self.coords = self.extract_conjunctiva_region(frame, face_landmarks, frame.shape)
                self.sclera_roi = self.extract_sclera_region(frame, face_landmarks, frame.shape)
                
        # Draw ROI boundaries
        if self.roi is not None:
            if isinstance(self.coords, list):
                for c in self.coords:
                    cv2.rectangle(display, (c[0], c[1]), (c[2], c[3]), (0, 255, 150), 2)
            else:
                cv2.rectangle(display, (self.coords[0], self.coords[1]), (self.coords[2], self.coords[3]), (0, 255, 150), 2)
                
            self.detection_status_var.set("✓ TARGET DETECTED")
            self.detection_status_lbl.config(fg="#2ECC71")
        else:
            self.detection_status_var.set("⏳ POSITION TARGET IN FRAME")
            self.detection_status_lbl.config(fg="#FF6B6B")
            
        # Convert BGR display frame to RGB for Tkinter
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        # Resize to fit the left label
        img = Image.fromarray(display_rgb)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.cam_label.img_tk = img_tk
        self.cam_label.config(image=img_tk)
        
        self.root.after(15, self.update_camera)

    def capture_current_scan(self):
        if self.roi is None:
            messagebox.showwarning("Not Detected", "Please position the target correctly in the camera view first.")
            return
            
        mode = self.scan_sequence[self.current_index]
        try:
            analysis = self.advanced_color_analysis(self.roi)
            
            if mode == 'palm':
                score, warnings, confidence = self.assess_palm_with_ml(analysis)
            elif mode == 'nail':
                score, warnings, confidence = self.assess_nail_with_ml(analysis)
            elif mode == 'conjunctiva':
                score, warnings, confidence = self.assess_conjunctiva_with_ml(analysis, self.sclera_roi)
            
            self.results[mode] = {
                'analysis': analysis,
                'risk_score': score,
                'warnings': warnings,
                'confidence': confidence,
            }
            
            self.update_results_text(f"✅ Captured {mode.capitalize()}: Score = {score:.1f}/5\n")
            
            # Update checklist status
            self.update_checklist()
            
            if self.current_index < len(self.scan_sequence) - 1:
                self.current_index += 1
                self.update_step_instructions()
            else:
                self.update_results_text("\n🎉 All scans completed! Click 'Calculate Risk' to view combined results.\n")
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze scan: {e}")

    def reset_scanner(self):
        self.results = {'palm': None, 'nail': None, 'conjunctiva': None}
        self.current_index = 0
        self.results_text.delete("1.0", "end")
        self.update_checklist()
        self.update_step_instructions()
        self.update_results_text("🔄 Scanner reset. Position palm to start.\n")

    def show_combined_results(self):
        assessment = self.combined_ml_assessment(self.results)
        if not assessment:
            messagebox.showwarning("No Data", "Please complete all scans first!")
            return
            
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "📊 ANEMIA SCREENING RESULTS\n")
        self.results_text.insert("end", "="*40 + "\n\n")
        self.results_text.insert("end", f"Risk Level: {assessment['risk_level']}\n")
        self.results_text.insert("end", f"Average Score: {assessment['avg_score']:.2f} / 5.0\n")
        self.results_text.insert("end", f"Scans Completed: {assessment['scans_completed']} / 3\n\n")
        
        self.results_text.insert("end", "🔍 KEY FINDINGS:\n")
        if assessment['warnings']:
            for w in assessment['warnings']:
                self.results_text.insert("end", f"   • {w}\n")
        else:
            self.results_text.insert("end", "   • No significant pallor detected.\n")
            
        self.results_text.insert("end", "\n📋 RECOMMENDATION:\n")
        self.results_text.insert("end", f"   {assessment['recommendation']}\n")
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            viz = self.visualize_combined(frame, assessment)
            
            # Display viz in a tkinter TopLevel window
            viz_win = tk.Toplevel(self.root)
            viz_win.title("Analysis Visualization")
            viz_win.configure(bg="#0D0D12")
            
            # Center relative to parent
            viz_win.update_idletasks()
            w = viz.shape[1]
            h = viz.shape[0]
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (w // 2)
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (h // 2)
            viz_win.geometry(f"{w}x{h}+{x}+{y}")
            
            # Convert BGR to RGB
            viz_rgb = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(viz_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            lbl = tk.Label(viz_win, image=img_tk, bg="#0D0D12")
            lbl.image = img_tk
            lbl.pack(padx=20, pady=20)
            viz_win.grab_set()


def run_anemia_scanner():
    """Entry point for anemia scanner"""
    try:
        scanner = EnhancedAnemiaScanner()
        scanner.run_scanner()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_anemia_scanner()