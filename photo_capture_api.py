from flask import Flask, jsonify, send_file
import cv2
import time
import os

app = Flask(__name__)

RTSP_URL = "rtsp://admin:hikvision101@192.168.169.244/Streaming/channels/101?tcp"

@app.route("/capture", methods=["GET"])
def capture_photo():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        return jsonify({"status": "error", "message": "Failed to open RTSP stream"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return jsonify({"status": "error", "message": "Failed to read frame"}), 500

    timestamp = int(time.time())
    filename = f"captured_{timestamp}.bmp"
    save_path = os.path.join("calib_images", filename)
    os.makedirs("captured_images", exist_ok=True)

    cv2.imwrite(save_path, frame)

    return send_file(save_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
