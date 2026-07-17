"""
Parkinson's detection logic extracted from Parkinson1.py.
All thresholds, formulas, and scoring preserved EXACTLY as-is.
Only change: Tkinter/OpenCV GUI removed; functions accept data and return dicts.
"""
import numpy as np
import os, json, glob, io, base64
from scipy.signal import find_peaks
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
import librosa

# ── Thresholds (EXACT copy from Parkinson1.py) ──────────────────────────────
TH_JITTER_PCT = 1.04
TH_SHIMMER_PCT = 3.81
TH_HNR_DB = 20.0
TH_TAP_DECREMENT_PCT = 30.0
TH_SPIRAL_TREMOR_MIN_HZ = 4.0
TH_SPIRAL_TREMOR_MAX_HZ = 6.0
TH_SPIRAL_DEV_MAX_PX = 45.0
TH_REACTION_MEAN_MS = 650.0
TH_REACTION_STD_MS = 150.0

WT_MOTOR = 0.30
WT_VOICE = 0.30
WT_SPIRAL = 0.20
WT_REACTION = 0.20


# ── Spiral helpers ───────────────────────────────────────────────────────────

def resample_points(points, num_points=100):
    """Resample/normalize a path of 2D coordinates into exactly num_points."""
    pts = np.array(points, dtype=float)[:, :2]
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


def _extract_spiral_features(points, mean_dev=None, std_dev=None):
    """Extract normalized clinical features from resampled and scaled points."""
    try:
        pts = resample_points(points, num_points=100)
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        pts_centered = pts - [cx, cy]
        radii = np.sqrt(pts_centered[:, 0]**2 + pts_centered[:, 1]**2)
        max_r = np.max(radii)
        if max_r < 5.0:
            return None
        pts_norm = pts_centered / max_r
        radii_norm = radii / max_r
        if mean_dev is None or std_dev is None:
            ideal_norm = np.linspace(radii_norm[0], radii_norm[-1], len(radii_norm))
            deviation_norm = radii_norm - ideal_norm
            mean_dev = np.mean(np.abs(deviation_norm))
            std_dev = np.std(deviation_norm)
        fft_vals = np.abs(np.fft.rfft(deviation_norm - np.mean(deviation_norm)))
        tremor_power = np.sum(fft_vals[13:21]) / (np.sum(fft_vals) + 1e-8)
        dx = np.diff(pts_norm[:, 0])
        dy = np.diff(pts_norm[:, 1])
        if len(dx) > 2:
            angles = np.unwrap(np.arctan2(dy, dx))
            curvature = np.diff(angles)
            curvature_var = np.var(curvature)
        else:
            curvature_var = 0.0
        segments = np.sqrt(dx**2 + dy**2)
        total_length = np.sum(segments)
        return [mean_dev, std_dev, tremor_power, curvature_var, total_length]
    except Exception:
        return None


def _load_spiral_dataset():
    """Load all spiral JSON files and extract features for classification."""
    # Go up one level from detection/ to find spiral_dataset/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "spiral_dataset")
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
        points.append([round(x, 2), round(y, 2)])
    return points


# ── Plot helper ──────────────────────────────────────────────────────────────

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#0D0D12', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Motor analysis ───────────────────────────────────────────────────────────

def analyze_motor_data(distances, timestamps):
    """Analyze finger-tapping data. Returns dict with score, raw metrics, plot, rhythm stats."""
    if len(distances) < 20:
        return {'score': 0.0, 'decrement': 0, 'speed': 0, 'plot': None,
                'rhythm_status': 'N/A', 'rhythm_warning': None,
                'error': 'Not enough data points (need >= 20)'}

    dist_arr = np.array(distances, dtype=float)
    time_arr = np.array(timestamps, dtype=float)
    peaks, _ = find_peaks(dist_arr, distance=8, prominence=15)

    if len(peaks) >= 6:
        peak_amplitudes = dist_arr[peaks]
        first5 = np.mean(peak_amplitudes[:5])
        last5 = np.mean(peak_amplitudes[-5:])
        decrement = ((first5 - last5) / first5) * 100 if first5 > 0 else 0
        if len(peaks) >= 2:
            tap_intervals = np.diff(time_arr[peaks])
            tapping_speed = 1.0 / np.mean(tap_intervals) if np.mean(tap_intervals) > 0 else 0
        else:
            tapping_speed = 0
            tap_intervals = []
        
        regularity_cv = (np.std(tap_intervals) / np.mean(tap_intervals)
                         if len(tap_intervals) > 0 and np.mean(tap_intervals) > 0 else 1.0)

        decrement_penalty = min(40, max(0, decrement - 10) * 1.3)
        # speed_penalty is removed so we measure rhythm stability rather than speed
        regularity_penalty = min(40, regularity_cv * 80)
        score = max(0.0, min(100.0, 100.0 - decrement_penalty - regularity_penalty))

        rhythm_status = "Stable / Regular"
        rhythm_warning = None
        if regularity_cv > 0.35:
            rhythm_status = "Irregular / Arrhythmic"
            rhythm_warning = "Irregular tapping rhythm detected. This may indicate coordination difficulty or motor instability."

        # Generate plot
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(time_arr, dist_arr, 'cyan', linewidth=2)
        ax1.plot(time_arr[peaks], dist_arr[peaks], 'rv', markersize=10, label='Tap peaks')
        ax1.set_title(f"Finger Tapping - Decrement: {decrement:.1f}% | Rhythm CV: {regularity_cv:.2f}")
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Distance (px)")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        fft = np.fft.fft(dist_arr)
        freqs = np.fft.fftfreq(len(dist_arr), d=np.mean(np.diff(time_arr)))
        ax2.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(fft)//2], 'yellow', linewidth=2)
        ax2.set_title("Frequency Analysis")
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_b64 = _fig_to_base64(fig)

        return {'score': round(score, 2), 'decrement': round(decrement, 2),
                'speed': round(tapping_speed, 2), 'plot': plot_b64,
                'rhythm_status': rhythm_status, 'rhythm_warning': rhythm_warning}
    else:
        deviation = np.std(distances)
        score = max(0.0, 100.0 - deviation / 5)
        return {'score': round(score, 2), 'decrement': 0, 'speed': 0, 'plot': None,
                'rhythm_status': 'Insufficient peaks to assess',
                'rhythm_warning': 'Please ensure your hand is fully visible and you tap clearly.'}


# ── Voice analysis ───────────────────────────────────────────────────────────

def analyze_voice_audio(audio_path_or_bytes, sample_rate=22050):
    """Analyze voice audio file. Returns dict with score, raw metrics, plot."""
    try:
        if isinstance(audio_path_or_bytes, str):
            y, sr = librosa.load(audio_path_or_bytes, sr=sample_rate)
        else:
            y, sr = librosa.load(audio_path_or_bytes, sr=sample_rate)
    except Exception as e:
        return {'score': 0.0, 'error': f'Could not load audio: {e}'}

    fs = sr
    snd = parselmouth.Sound(y, sampling_frequency=fs)
    pitch = call(snd, "To Pitch", 0.0, 75, 500)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    try:
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    except Exception:
        jitter = 0.0
    try:
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    except Exception:
        shimmer = 0.0
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
    except Exception:
        hnr = 25.0

    score = 100.0
    if jitter > TH_JITTER_PCT:
        score -= min(30, (jitter - TH_JITTER_PCT) * 15)
    if shimmer > TH_SHIMMER_PCT:
        score -= min(30, (shimmer - TH_SHIMMER_PCT) * 8)
    if hnr < TH_HNR_DB:
        score -= min(30, (TH_HNR_DB - hnr) * 3)
    score = max(0.0, min(100.0, score))

    # Plot
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=fs)
    times = librosa.times_like(f0)
    duration = len(y) / fs

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(times, f0, 'lime', linewidth=2)
    axes[0].set_title("Voice Pitch Over Time"); axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Frequency (Hz)")
    axes[0].grid(True, alpha=0.3)

    time_audio = np.linspace(0, duration, len(y))
    axes[1].plot(time_audio, y, 'cyan', alpha=0.7)
    axes[1].set_title("Audio Waveform"); axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    colors_bar = [
        '#2ecc71' if jitter <= TH_JITTER_PCT else '#e74c3c',
        '#2ecc71' if shimmer <= TH_SHIMMER_PCT else '#e74c3c',
        '#2ecc71' if hnr >= TH_HNR_DB else '#e74c3c'
    ]
    metrics = [jitter, shimmer, hnr]
    labels = [f'Jitter\n{jitter:.2f}%', f'Shimmer\n{shimmer:.2f}%', f'HNR\n{hnr:.1f}dB']
    axes[2].bar(labels, metrics, color=colors_bar, edgecolor='white', linewidth=0.5)
    axes[2].set_title("Clinical Voice Metrics"); axes[2].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_b64 = _fig_to_base64(fig)

    return {'score': round(score, 2), 'jitter': round(jitter, 3),
            'shimmer': round(shimmer, 3), 'hnr': round(hnr, 2), 'plot': plot_b64}


# ── Spiral analysis ──────────────────────────────────────────────────────────

def compute_tremor_index(points):
    """Calculate scale-invariant high frequency radius residual and raw curvature variance."""
    pts = np.array(points, dtype=float)[:, :2]
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    pts_centered = pts - [cx, cy]
    
    # Polar coordinates
    radii = np.sqrt(pts_centered[:, 0]**2 + pts_centered[:, 1]**2)
    angles = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
    
    # Sort by unwrapped angles to trace the spiral outwards
    unwrapped_angles = np.unwrap(angles)
    sort_idx = np.argsort(unwrapped_angles)
    sorted_radii = radii[sort_idx]
    
    # Moving average filter to extract smooth trend
    window_size = 15
    if len(sorted_radii) < window_size * 2:
        return 0.0, 0.0
    
    padded = np.pad(sorted_radii, (window_size//2, window_size//2), mode='edge')
    smooth_radii = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    
    residual = sorted_radii - smooth_radii
    max_r = np.max(radii)
    normalized_residual = residual / (max_r + 1e-8)
    
    tremor_power = np.std(normalized_residual) * 100 # percentage scale
    
    # Curvature variance on the raw points (excluding very close duplicate coordinates)
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    keep = dists > 1.0
    if np.sum(keep) > 5:
        clean_pts = pts[1:][keep]
        dx = np.diff(clean_pts[:, 0])
        dy = np.diff(clean_pts[:, 1])
        raw_angles = np.unwrap(np.arctan2(dy, dx))
        raw_curv = np.diff(raw_angles)
        raw_curv_var = np.var(raw_curv)
    else:
        raw_curv_var = 0.0
        
    return tremor_power, raw_curv_var


def analyze_spiral_data(drawn_points):
    """Analyze spiral drawing data. Returns dict with score, deviation, tremor index, and status."""
    if len(drawn_points) < 20:
        return {'score': 0.0, 'deviation': 0, 'tremor_index': 0.0, 'curvature_var': 0.0,
                'smoothness_status': 'Drawing too short', 'error': 'Drawing too short'}

    X_data, y_data = _load_spiral_dataset()
    ref_points = generate_reference_spiral()

    ref_arr = np.array(ref_points)
    user_arr = np.array([[p[0], p[1]] for p in drawn_points])

    deviations = []
    for up in user_arr:
        dists = np.linalg.norm(ref_arr - up, axis=1)
        deviations.append(np.min(dists))
    mean_dev = np.mean(deviations) if deviations else 0.0
    std_dev = np.std(deviations) if deviations else 0.0

    feat = _extract_spiral_features(drawn_points, mean_dev, std_dev)
    if feat is None:
        dev_score = max(0.0, min(100.0, 100.0 - (mean_dev / TH_SPIRAL_DEV_MAX_PX) * 50.0))
        score = dev_score
    elif X_data is not None and len(X_data) >= 3:
        k = min(5, len(X_data))
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_data, y_data)
        proba = knn.predict_proba([feat])[0]
        healthy_prob = proba[0] if len(proba) > 1 else (1.0 if y_data[0] == 0 else 0.0)
        dev_score = max(0.0, min(100.0, 100.0 - (mean_dev / TH_SPIRAL_DEV_MAX_PX) * 50.0))
        score = 0.5 * (healthy_prob * 100.0) + 0.5 * dev_score
    else:
        tremor = feat[2]
        score = 100.0
        if tremor > 0.15:
            score -= 30.0
        if mean_dev > TH_SPIRAL_DEV_MAX_PX:
            score -= min(50.0, (mean_dev - TH_SPIRAL_DEV_MAX_PX) * 2.0)

    # Calculate polar radius high frequency residual and raw curvature variance
    tremor_index, raw_curv_var = compute_tremor_index(drawn_points)
    
    # Calculate a Jaggedness Penalty
    # We penalize any high-frequency radius residual > 0.8% and curvature variance > 0.4.
    tremor_penalty = 0.0
    if tremor_index > 0.8:
        tremor_penalty += min(45.0, (tremor_index - 0.8) * 45.0)
    if raw_curv_var > 0.4:
        tremor_penalty += min(45.0, (raw_curv_var - 0.4) * 35.0)
        
    score = score - tremor_penalty
    score = max(0.0, min(100.0, score))
    
    smoothness_status = "Smooth / Stable"
    if tremor_penalty > 35.0:
        smoothness_status = "Severely Jagged / Tremor"
    elif tremor_penalty > 0.0:
        smoothness_status = "Mild Jitter / Tremor"

    return {
        'score': round(score, 2),
        'deviation': round(mean_dev, 2),
        'tremor_index': round(tremor_index, 3),
        'curvature_var': round(raw_curv_var, 3),
        'smoothness_status': smoothness_status
    }


# ── Reaction analysis ────────────────────────────────────────────────────────

def analyze_reaction_data(latencies, mouse_paths=None):
    """Analyze reaction test data. Returns dict with score, mean latency."""
    if not latencies:
        return {'score': 0.0, 'mean_latency': 0, 'error': 'No latencies recorded'}

    mean_lat = float(np.mean(latencies))

    jerk_counts = []
    if mouse_paths:
        for path in mouse_paths:
            if len(path) < 4:
                jerk_counts.append(0)
                continue
            xs = [p[0] for p in path]
            dx = np.diff(xs)
            sign_changes = int(np.sum(np.abs(np.diff(np.sign(dx))) > 0))
            jerk_counts.append(sign_changes)
    avg_jerks = float(np.mean(jerk_counts)) if jerk_counts else 0

    abnormal_threshold = TH_REACTION_MEAN_MS + 2.0 * TH_REACTION_STD_MS
    span = abnormal_threshold - TH_REACTION_MEAN_MS
    latency_score = 100.0 - ((mean_lat - TH_REACTION_MEAN_MS) / span) * 100.0
    latency_score = max(0.0, min(100.0, latency_score))
    jerk_penalty = min(30.0, avg_jerks * 3.0)
    score = max(0.0, min(100.0, latency_score - jerk_penalty))

    return {'score': round(score, 2), 'mean_latency': round(mean_lat, 2)}


# ── Health index ─────────────────────────────────────────────────────────────

def calculate_health_index(motor, voice, spiral, reaction):
    """Calculate weighted neurological health index."""
    avg = motor * WT_MOTOR + voice * WT_VOICE + spiral * WT_SPIRAL + reaction * WT_REACTION
    if avg >= 75:
        status = "EXCELLENT"
    elif avg >= 70:
        status = "GOOD"
    elif avg >= 65:
        status = "FAIR"
    else:
        status = "Needs Attention"
    return {'score': round(avg, 2), 'status': status,
            'motor': round(motor, 2), 'voice': round(voice, 2),
            'spiral': round(spiral, 2), 'reaction': round(reaction, 2)}
