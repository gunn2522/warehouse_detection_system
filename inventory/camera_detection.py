import cv2
import time
from threading import Thread, Lock
from .roboflow_utils import detect_and_classify_cylinders
from .models import CylinderInventory, TruckLog
from django.utils.timezone import now

camera_url = "videos/demo.mp4"


import cv2

cap = cv2.VideoCapture(camera_url)

# Global state
cap = None
frame = None
lock = Lock()
is_running = False
last_detected_trucks = []
last_inventory_count = None


def camera_loop():
    global cap, frame, last_detected_trucks, last_inventory_count, is_running
    print(f"📡 Connecting to RTSP camera at {camera_url}...")
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("❌ Failed to connect to camera.")
        is_running = False
        return

    while is_running:
        ret, img = cap.read()
        if not ret:
            print("⚠️ Failed to read frame. Attempting reconnect...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(camera_url)
            continue

        annotated_frame, result = detect_and_classify_cylinders(img)

        # Save detection results
        cylinder_count = result.get("cylinder_count", 0)
        trucks_detected = result.get("truck_count", 0)

        # Update inventory if changed
        if last_inventory_count != cylinder_count:
            CylinderInventory.objects.create(count=cylinder_count)
            last_inventory_count = cylinder_count
            print(f"📦 Inventory updated: {cylinder_count}")

        # Log trucks only if count has changed
        if trucks_detected != len(last_detected_trucks):
            TruckLog.objects.create(truck_type="loaded", cylinder_count=cylinder_count)
            last_detected_trucks = [None] * trucks_detected
            print(f"🚛 Truck logged: loaded — {cylinder_count} cylinders")

        with lock:
            frame = annotated_frame

    cap.release()
    print("📴 Camera stream stopped.")


def start_camera_detection():
    global is_running
    if not is_running:
        is_running = True  # ✅ MUST SET TRUE before starting
        t = Thread(target=camera_loop)
        t.daemon = True
        t.start()
        time.sleep(1)  # allow startup time


    def generate():
        while True:
            with lock:
                if frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return generate()


def get_detection_result():
    return {
        "success": True,
        "message": "Detection working.",
        "timestamp": str(now())
    }
