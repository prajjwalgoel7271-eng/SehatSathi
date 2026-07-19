"""
TB cough analysis logic extracted from TB Checker.py.
All thresholds, formulas, and scoring preserved EXACTLY as-is.
"""
import numpy as np
from scipy import signal
import librosa
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CHUNK = 1024
RATE = 44100


def compute_cough_features(cough_segment, sr=44100):
    """Extract MFCC + spectral features from a cough segment. (unchanged)"""
    try:
        seg = cough_segment.astype(float)
        if len(seg) < 512:
            return None
        mfccs = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=seg))
        sc = np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr))
        sro = np.mean(librosa.feature.spectral_rolloff(y=seg, sr=sr))
        sf = np.mean(librosa.feature.spectral_flatness(y=seg))
        return np.concatenate([mfcc_means, [zcr, sc, sro, sf]])
    except Exception:
        return None


def analyze_cough(audio_signal, sample_rate=44100):
    """Analyze cough audio. All scoring logic unchanged from TB Checker.py."""
    audio_normalized = audio_signal / (np.max(np.abs(audio_signal)) + 1e-8)

    rms = np.array([np.sqrt(np.mean(audio_normalized[i:i+CHUNK]**2))
                    for i in range(0, len(audio_normalized)-CHUNK, CHUNK//2)])

    cough_threshold = 0.1
    cough_indices = np.where(rms > cough_threshold)[0]

    if len(cough_indices) == 0:
        return {
            'cough_count': 0, 'avg_duration': 0, 'avg_frequency': 0,
            'risk_level': 'No cough detected', 'risk_score': 0,
            'confidence': 0, 'durations': [], 'frequencies': [],
            'features_extracted': False
        }

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

    durations = []; frequencies = []; all_features = []
    for event in cough_events:
        duration_frames = len(event)
        duration_seconds = (duration_frames * (CHUNK//2)) / sample_rate
        durations.append(duration_seconds)
        start_sample = event[0] * (CHUNK//2)
        end_sample = min(event[-1] * (CHUNK//2) + CHUNK, len(audio_normalized))
        cough_segment = audio_normalized[start_sample:end_sample]
        if len(cough_segment) > 100:
            f, Pxx = signal.periodogram(cough_segment, sample_rate)
            dominant_freq = f[np.argmax(Pxx)]
            frequencies.append(float(dominant_freq))
        feat = compute_cough_features(cough_segment, sample_rate)
        if feat is not None:
            all_features.append(feat)

    avg_duration = float(np.mean(durations)) if durations else 0
    avg_frequency = float(np.mean(frequencies)) if frequencies else 0
    cough_count = len(cough_events)

    confidence = 0.0
    if all_features:
        mean_feat = np.mean(all_features, axis=0)
        healthy_ref = np.zeros(17)
        healthy_ref[14] = 3000; healthy_ref[13] = 0.08
        healthy_ref[15] = 6000; healthy_ref[16] = 0.3
        duration_risk = min(1.0, max(0, (avg_duration - 0.3) / 0.4))
        freq_risk = min(1.0, max(0, (1500 - avg_frequency) / 1500)) if avg_frequency > 0 else 0.3
        count_risk = min(1.0, cough_count / 5.0)
        if healthy_ref[14] > 0:
            centroid_risk = min(1.0, max(0, (healthy_ref[14] - mean_feat[14]) / healthy_ref[14]))
        else:
            centroid_risk = 0.3
        confidence = (duration_risk * 0.25 + freq_risk * 0.25 +
                      count_risk * 0.15 + centroid_risk * 0.35) * 100
    else:
        risk_score = 0
        if avg_duration > 0.4: risk_score += 2
        if avg_frequency < 1000 and avg_frequency > 0: risk_score += 2
        if cough_count >= 3: risk_score += 1
        confidence = (risk_score / 5.0) * 100

    confidence = max(0, min(100, confidence))
    if confidence > 60: risk_level = "HIGH RISK"
    elif confidence > 30: risk_level = "MODERATE RISK"
    else: risk_level = "LOW RISK"

    return {
        'cough_count': cough_count,
        'avg_duration': round(avg_duration, 3),
        'avg_frequency': round(avg_frequency, 1),
        'risk_level': risk_level,
        'risk_score': round(confidence / 20, 2),
        'confidence': round(confidence, 1),
        'durations': [round(d, 3) for d in durations],
        'frequencies': [round(f, 1) for f in frequencies],
        'features_extracted': len(all_features) > 0,
        'duration_flag': avg_duration > 0.4,
        'frequency_flag': avg_frequency < 1000 and avg_frequency > 0,
        'count_flag': cough_count >= 3,
    }


def load_wav_file(file_obj_or_path):
    import wave
    import numpy as np
    
    if isinstance(file_obj_or_path, str):
        wav = wave.open(file_obj_or_path, 'rb')
    else:
        file_obj_or_path.seek(0)
        wav = wave.open(file_obj_or_path, 'rb')
        
    try:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        n_frames = wav.getnframes()
        content = wav.readframes(n_frames)
    finally:
        wav.close()
        
    if sampwidth == 2:
        data = np.frombuffer(content, dtype=np.int16)
        data = data.astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = np.frombuffer(content, dtype=np.uint8)
        data = data.astype(np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
        
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
        
    return data, framerate


def analyze_cough_audio(audio_file_obj):
    """Load audio from a file-like object and analyze it."""
    try:
        y, sr = load_wav_file(audio_file_obj)
    except Exception as e:
        return {'error': f'Could not load audio: {e}', 'cough_count': 0}
    return analyze_cough(y, sr)
