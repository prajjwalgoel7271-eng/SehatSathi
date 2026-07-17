"""
Anemia detection logic extracted from anemia2.py.
All thresholds, formulas, and scoring preserved EXACTLY as-is.
"""
import cv2
import numpy as np
import mediapipe as mp
import base64

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                   refine_landmarks=True, min_detection_confidence=0.5)

# Landmark indices (exact copy)
PALM_CENTER_LANDMARKS = [0, 5, 9, 13, 17]
FINGERTIP_LANDMARKS = [4, 8, 12, 16, 20]
LOWER_EYELID_LANDMARKS = [145, 146, 147, 148, 149, 150, 151, 152,
                          374, 375, 376, 377, 378, 379, 380, 381]
SCLERA_LANDMARKS = [33, 133, 362, 263]

# Thresholds (exact copy)
THRESHOLDS = {
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


def decode_base64_image(b64_string):
    """Decode a base64-encoded image string to an OpenCV BGR numpy array."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def _get_default_analysis():
    return {
        'rgb': (128, 128, 128), 'saturation': 50, 'brightness': 128,
        'red_ratio': 0.33, 'redness_index': 0.0, 'pallor_index': 0.5,
        'pink_score': 0, 'color_variance': 0, 'valid_analysis': False
    }


def advanced_color_analysis(roi):
    """Color analysis (unchanged from anemia2.py)."""
    try:
        if roi.size == 0:
            return _get_default_analysis()
        b, g, r = cv2.split(roi)
        r_mean, r_std = float(np.mean(r)), float(np.std(r))
        g_mean, g_std = float(np.mean(g)), float(np.std(g))
        b_mean, b_std = float(np.mean(b)), float(np.std(b))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_mean = float(np.mean(s))
        v_mean = float(np.mean(v))
        total = r_mean + g_mean + b_mean + 1e-6
        red_ratio = r_mean / total
        redness_index = (r_mean - g_mean) / (r_mean + g_mean + 1e-6)
        pallor_index = (255 - r_mean) / 255.0
        pink_score = r_mean - g_mean
        color_variance = np.mean([r_std, g_std, b_std])
        return {
            'rgb': (r_mean, g_mean, b_mean), 'saturation': s_mean,
            'brightness': v_mean, 'red_ratio': red_ratio,
            'redness_index': redness_index, 'pallor_index': pallor_index,
            'pink_score': pink_score, 'color_variance': color_variance,
            'valid_analysis': True
        }
    except Exception:
        return _get_default_analysis()


def assess_palm(analysis):
    """Palm assessment (unchanged)."""
    if not analysis.get('valid_analysis', True):
        return 0, ["Invalid analysis"], 0.1
    score = 0; warnings = []
    if analysis['red_ratio'] < THRESHOLDS['palm']['anemic_red_ratio']:
        score += 1.5; warnings.append("Mild palm pallor")
    elif analysis['red_ratio'] < THRESHOLDS['palm']['healthy_red_ratio']:
        score += 0.5; warnings.append("Slight color reduction")
    if analysis['saturation'] < THRESHOLDS['palm']['saturation_min']:
        score += 0.5; warnings.append("Moderate saturation")
    confidence = max(0.5, 1.0 - (analysis['color_variance'] / 100.0))
    return min(score, 5), warnings, confidence


def assess_nail(analysis):
    """Nail assessment (unchanged)."""
    if not analysis.get('valid_analysis', True):
        return 0, ["Invalid analysis"], 0.1
    score = 0; warnings = []
    if analysis['pink_score'] < THRESHOLDS['nail']['anemic_pink_score']:
        score += 1.5; warnings.append("Pale nail beds")
    elif analysis['pink_score'] < THRESHOLDS['nail']['healthy_pink_score']:
        score += 0.5; warnings.append("Slight nail pallor")
    confidence = 0.7 if analysis['color_variance'] < 30 else 0.5
    return min(score, 5), warnings, confidence


def assess_conjunctiva(analysis, sclera_roi=None):
    """Conjunctiva assessment using clinical pallor ratio with sclera white-balance."""
    if not analysis.get('valid_analysis', True):
        return 0, ["Invalid analysis"], 0.1
    r_mean, g_mean, b_mean = analysis['rgb']
    pallor_ratio = r_mean / (g_mean + b_mean + 1e-6)
    if sclera_roi is not None and sclera_roi.size > 0:
        sb, sg, sr = cv2.split(sclera_roi)
        sclera_r = float(np.mean(sr))
        sclera_gb = float(np.mean(sg)) + float(np.mean(sb))
        sclera_ratio = sclera_r / (sclera_gb + 1e-6)
        if sclera_ratio > 0:
            pallor_ratio = pallor_ratio / sclera_ratio
    score = 0; warnings = []
    if pallor_ratio > 0.85:
        score = 0; warnings.append("Conjunctiva appears healthy")
    elif pallor_ratio > 0.70:
        score = 1.5; warnings.append("Mild conjunctival pallor detected")
    elif pallor_ratio > 0.55:
        score = 3.0; warnings.append("Moderate conjunctival pallor detected")
    else:
        score = 4.5; warnings.append("Severe conjunctival pallor detected")
    return min(score, 5), warnings, 0.8


def _extract_roi(frame, landmarks, indices, margin=15):
    """Generic ROI extraction from landmarks with support for negative margins (shrinkage)."""
    h, w, _ = frame.shape
    points = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            points.append((int(lm.x * w), int(lm.y * h)))
    if len(points) < 2:
        return None, None
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    if margin < 0:
        # Shrink the box but ensure it doesn't collapse below 24x24 pixels centered on palm
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        x_min = min(x_min - margin, cx - 12)
        y_min = min(y_min - margin, cy - 12)
        x_max = max(x_max + margin, cx + 12)
        y_max = max(y_max + margin, cy + 12)
    else:
        x_min = x_min - margin
        y_min = y_min - margin
        x_max = x_max + margin
        y_max = y_max + margin
        
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    if x_max <= x_min or y_max <= y_min:
        return None, None
        
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return None, None
    return roi, (x_min, y_min, x_max, y_max)


def analyze_frame(image_b64, scan_type):
    """Process a single frame for anemia analysis. Returns results and a base64 marked image showing detected ROIs."""
    frame = decode_base64_image(image_b64)
    if frame is None:
        return {'error': 'Could not decode image', 'detected': False}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = None; sclera_roi = None
    coords = None; sclera_coords = None
    marked_frame = frame.copy()

    if scan_type in ('palm', 'nail'):
        results = hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return {'error': f'No hand detected for {scan_type} scan', 'detected': False}
        hand_lm = results.multi_hand_landmarks[0]
        
        if scan_type == 'palm':
            # Use a negative margin (-20) to shrink the ROI so it's focused directly on the center of the palm
            roi, coords = _extract_roi(frame, hand_lm.landmark, PALM_CENTER_LANDMARKS, margin=-20)
            if coords is not None:
                cv2.rectangle(marked_frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
        else:
            # Nail: extract multiple fingertip regions and draw boxes
            h, w, _ = frame.shape
            nail_rois = []
            for tip_idx in FINGERTIP_LANDMARKS[:3]:
                lm = hand_lm.landmark[tip_idx]
                x, y = int(lm.x * w), int(lm.y * h)
                ns = 12
                x1, y1 = max(0, x-ns), max(0, y-ns)
                x2, y2 = min(w, x+ns), min(h, y+ns)
                if x2 > x1 and y2 > y1:
                    nr = frame[y1:y2, x1:x2]
                    if nr.size > 0 and nr.shape[0] > 5 and nr.shape[1] > 5:
                        nail_rois.append(nr)
                        cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if len(nail_rois) < 2:
                return {'error': 'Not enough nail regions detected', 'detected': False}
            roi = np.vstack([cv2.resize(r, (30, 30)) for r in nail_rois[:3]])

    elif scan_type == 'conjunctiva':
        results = face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return {'error': 'No face detected for conjunctiva scan', 'detected': False}
        face_lm = results.multi_face_landmarks[0]
        
        roi, coords = _extract_roi(frame, face_lm.landmark, LOWER_EYELID_LANDMARKS, margin=3)
        if coords is not None:
            cv2.rectangle(marked_frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)
            
        # Sclera white-balance reference
        sclera_roi_data, sclera_coords = _extract_roi(frame, face_lm.landmark, SCLERA_LANDMARKS, margin=2)
        sclera_roi = sclera_roi_data
        if sclera_coords is not None:
            cv2.rectangle(marked_frame, (sclera_coords[0], sclera_coords[1]), (sclera_coords[2], sclera_coords[3]), (0, 0, 255), 2)

    if roi is None:
        return {'error': f'Could not extract {scan_type} region', 'detected': False}

    analysis = advanced_color_analysis(roi)
    if scan_type == 'palm':
        score, warnings, confidence = assess_palm(analysis)
    elif scan_type == 'nail':
        score, warnings, confidence = assess_nail(analysis)
    else:
        score, warnings, confidence = assess_conjunctiva(analysis, sclera_roi)

    # Encode marked preview image to base64
    _, buf = cv2.imencode('.jpg', marked_frame)
    marked_b64 = base64.b64encode(buf).decode('utf-8')
    marked_image_url = f"data:image/jpeg;base64,{marked_b64}"

    return {
        'detected': True, 'scan_type': scan_type,
        'risk_score': round(score, 2), 'warnings': warnings,
        'confidence': round(confidence, 2),
        'rgb': [round(v, 1) for v in analysis['rgb']],
        'red_ratio': round(analysis['red_ratio'], 4),
        'marked_image': marked_image_url
    }


def combined_assessment(results_dict):
    """Combine palm/nail/conjunctiva results into overall assessment."""
    total_score = 0; all_warnings = []; scan_count = 0
    for scan_type, data in results_dict.items():
        if data and data.get('detected'):
            total_score += data['risk_score']
            all_warnings.extend(data.get('warnings', []))
            scan_count += 1
    if scan_count == 0:
        return {'error': 'No valid scans to assess'}
    avg_score = total_score / scan_count
    if avg_score >= 3.5:
        risk_level = "SEVERE"; recommendation = "Consult a doctor immediately for blood work"
    elif avg_score >= 2.0:
        risk_level = "MODERATE"; recommendation = "Consider a medical check-up soon"
    elif avg_score >= 1.0:
        risk_level = "MILD"; recommendation = "Monitor nutrition and iron intake"
    else:
        risk_level = "HEALTHY"; recommendation = "Indicators appear normal"
    return {
        'risk_level': risk_level, 'avg_score': round(avg_score, 2),
        'scans_completed': scan_count,
        'warnings': list(set(all_warnings)), 'recommendation': recommendation,
    }
