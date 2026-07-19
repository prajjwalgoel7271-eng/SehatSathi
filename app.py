import sys
# Block the broken TensorFlow installation from being imported by mediapipe.
# MediaPipe 0.10.14 tries to import tensorflow.tools.docs.doc_controls which
# fails due to a protobuf version mismatch on this system. Setting this stub
# forces mediapipe to use its built-in TFLite runtime instead.
sys.modules['tensorflow'] = None

import os
import io
import json
import base64
import traceback
import tempfile
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
    # Allow static files, home page, about page, and disclaimer pages
    allowed_paths = [
        "/",
        "/about",
        "/disclaimer",
        "/accept-disclaimer"
    ]
    if request.path.startswith("/static") or request.path in allowed_paths:
        return
    # If session does not have disclaimer accepted, redirect to disclaimer page
    if not session.get("disclaimer_accepted"):
        return redirect(url_for("disclaimer_view"))

# ── Frontend Views ──
@app.route("/")
def landing():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/disclaimer")
def disclaimer_view():
    return render_template("disclaimer.html")

@app.route("/accept-disclaimer", methods=["POST"])
def accept_disclaimer():
    session["disclaimer_accepted"] = True
    return jsonify({"status": "success"})

@app.route("/menu")
def menu():
    return render_template("menu.html")

# Health Score view
@app.route("/test/health-score")
def health_score():
    return render_template("health_score.html")

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
        traceback.print_exc()
        return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 400

@app.route("/api/parkinson/voice", methods=["POST"])
def api_parkinson_voice():
    temp_path = None
    try:
        print("api_parkinson_voice: Received voice audio upload request.")
        if "audio" not in request.files:
            print("api_parkinson_voice: No audio file in request.files")
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        
        # Write to a temp file in the OS temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"ss_voice_{os.urandom(8).hex()}.wav")
        print(f"api_parkinson_voice: Saving audio file to temp path: {temp_path}")
        audio_file.save(temp_path)
        
        # Call voice analysis logic
        print("api_parkinson_voice: Running voice analysis...")
        result = analyze_voice_audio(temp_path)
        
        if result and "error" in result:
            print(f"api_parkinson_voice: Analysis returned error: {result['error']}")
            return jsonify({
                "error": "Audio processing failed",
                "detail": result["error"]
            }), 500
            
        print("api_parkinson_voice: Voice analysis completed successfully.")
        return jsonify(result)
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"api_parkinson_voice: Exception caught during processing:\n{err_msg}", file=sys.stderr)
        return jsonify({
            "error": "Audio processing failed",
            "detail": str(e)
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                print(f"api_parkinson_voice: Cleaning up temp file: {temp_path}")
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"api_parkinson_voice: Failed to delete temp file {temp_path}: {cleanup_err}", file=sys.stderr)

@app.route("/api/parkinson/spiral", methods=["POST"])
def api_parkinson_spiral():
    try:
        data = request.get_json()
        points = data.get("points", [])
        result = analyze_spiral_data(points)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 400

@app.route("/api/parkinson/reaction", methods=["POST"])
def api_parkinson_reaction():
    try:
        data = request.get_json()
        latencies = data.get("latencies", [])
        mouse_paths = data.get("mouse_paths", [])
        result = analyze_reaction_data(latencies, mouse_paths)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 400

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
        traceback.print_exc()
        return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 400

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
        traceback.print_exc()
        return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 400

@app.route("/api/anemia/overall", methods=["POST"])
def api_anemia_overall():
    try:
        data = request.get_json()
        results_dict = data.get("results", {})
        result = combined_assessment(results_dict)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{str(e)}\n{traceback.format_exc()}"}), 400

@app.route("/api/tb/analyze", methods=["POST"])
def api_tb_analyze():
    temp_path = None
    try:
        print("api_tb_analyze: Received TB cough audio upload request.")
        if "audio" not in request.files:
            print("api_tb_analyze: No audio file in request.files")
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        
        # Write to a temp file in the OS temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"ss_tb_{os.urandom(8).hex()}.wav")
        print(f"api_tb_analyze: Saving audio file to temp path: {temp_path}")
        audio_file.save(temp_path)
        
        # Call cough analysis logic
        print("api_tb_analyze: Running TB analysis...")
        with open(temp_path, "rb") as f:
            result = analyze_cough_audio(f)
            
        if result and "error" in result:
            print(f"api_tb_analyze: Analysis returned error: {result['error']}")
            return jsonify({
                "error": "Audio processing failed",
                "detail": result["error"]
            }), 500
            
        print("api_tb_analyze: TB analysis completed successfully.")
        return jsonify(result)
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"api_tb_analyze: Exception caught during processing:\n{err_msg}", file=sys.stderr)
        return jsonify({
            "error": "Audio processing failed",
            "detail": str(e)
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                print(f"api_tb_analyze: Cleaning up temp file: {temp_path}")
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"api_tb_analyze: Failed to delete temp file {temp_path}: {cleanup_err}", file=sys.stderr)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
