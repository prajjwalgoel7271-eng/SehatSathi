import sys
sys.modules['tensorflow'] = None

import os
import io
import json
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

from detection.parkinson import (
    analyze_motor_data,
    analyze_voice_audio,
    analyze_spiral_data,
    analyze_reaction_data,
    calculate_health_index,
    generate_reference_spiral
)
from detection.anemia import analyze_frame, combined_assessment
from detection.tb import analyze_cough_audio

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sehatsaathi_secret_key_12345")

# ── Disclaimer Gate Middleware ──
@app.before_request
def check_disclaimer():
    # Allow static files and API calls / disclaimer submit
    if request.path.startswith("/static") or request.path == "/" or request.path == "/accept-disclaimer":
        return
    # If session does not have disclaimer accepted, redirect to landing page
    if not session.get("disclaimer_accepted"):
        return redirect(url_for("landing"))

# ── Frontend Views ──
@app.route("/")
def landing():
    return render_template("disclaimer.html")

@app.route("/accept-disclaimer", methods=["POST"])
def accept_disclaimer():
    session["disclaimer_accepted"] = True
    return jsonify({"status": "success"})

@app.route("/menu")
def menu():
    return render_template("menu.html")

# Parkinson views
@app.route("/test/parkinson")
def parkinson_hub():
    return render_template("parkinson/hub.html")

@app.route("/test/tapping")
def motor_test():
    return render_template("parkinson/motor.html")

@app.route("/test/voice")
def voice_test():
    return render_template("parkinson/voice.html")

@app.route("/test/spiral")
def spiral_test():
    # Pass reference spiral coordinates to the template so it can draw them
    ref_spiral = generate_reference_spiral(cx=230, cy=210, num_turns=4, max_r=180)
    return render_template("parkinson/spiral.html", ref_spiral=ref_spiral)

@app.route("/test/reaction")
def reaction_test():
    return render_template("parkinson/reaction.html")

# Anemia views
@app.route("/test/anemia")
def anemia_scanner():
    return render_template("anemia.html")

# TB views
@app.route("/test/tb")
def tb_analyzer():
    return render_template("tb.html")


# ── API Endpoints ──

@app.route("/api/parkinson/motor", methods=["POST"])
def api_parkinson_motor():
    try:
        data = request.get_json()
        distances = data.get("distances", [])
        timestamps = data.get("timestamps", [])
        result = analyze_motor_data(distances, timestamps)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/parkinson/voice", methods=["POST"])
def api_parkinson_voice():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files["audio"]
        # Save to buffer
        audio_bytes = audio_file.read()
        audio_io = io.BytesIO(audio_bytes)
        result = analyze_voice_audio(audio_io)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/parkinson/spiral", methods=["POST"])
def api_parkinson_spiral():
    try:
        data = request.get_json()
        points = data.get("points", [])
        result = analyze_spiral_data(points)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/parkinson/reaction", methods=["POST"])
def api_parkinson_reaction():
    try:
        data = request.get_json()
        latencies = data.get("latencies", [])
        mouse_paths = data.get("mouse_paths", [])
        result = analyze_reaction_data(latencies, mouse_paths)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/parkinson/overall", methods=["POST"])
def api_parkinson_overall():
    try:
        data = request.get_json()
        motor = data.get("motor", 0.0)
        voice = data.get("voice", 0.0)
        spiral = data.get("spiral", 0.0)
        reaction = data.get("reaction", 0.0)
        result = calculate_health_index(motor, voice, spiral, reaction)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/anemia/frame", methods=["POST"])
def api_anemia_frame():
    try:
        data = request.get_json()
        image_b64 = data.get("image")
        scan_type = data.get("scan_type") # 'palm', 'nail', or 'conjunctiva'
        if not image_b64 or not scan_type:
            return jsonify({"error": "Missing image or scan_type"}), 400
        result = analyze_frame(image_b64, scan_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/anemia/overall", methods=["POST"])
def api_anemia_overall():
    try:
        data = request.get_json()
        results_dict = data.get("results", {})
        result = combined_assessment(results_dict)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/tb/analyze", methods=["POST"])
def api_tb_analyze():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()
        audio_io = io.BytesIO(audio_bytes)
        result = analyze_cough_audio(audio_io)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
