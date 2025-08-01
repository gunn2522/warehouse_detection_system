# roboflow_utils.py

import os
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
from django.utils import timezone
from inventory.models import CylinderInventory, TruckLog

from datetime import datetime

# === Roboflow Config ===
ROBOFLOW_API_KEY = "m1I2grWDcCMSQSpDhZP0"
WORKSPACE = "arshpreet-singh"
WORKFLOW_ID = "detect-count-and-visualize-18"

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# Paths

OUTPUT_FRAMES = "output_frames"
os.makedirs(OUTPUT_FRAMES, exist_ok=True)

# === Truck Tracking State ===
active_trucks = {}  # {truck_id: {polygon, cylinders, missed_frames}}
truck_counter = 1
MAX_MISSED_FRAMES = 10


def _get_bounding_polygon(pred):
    x, y = pred["x"], pred["y"]
    w, h = pred["width"], pred["height"]
    return Polygon([
        (x - w / 2, y - h / 2),
        (x + w / 2, y - h / 2),
        (x + w / 2, y + h / 2),
        (x - w / 2, y + h / 2)
    ])


def assign_truck_ids(truck_polygons):
    global active_trucks, truck_counter
    new_map = {}

    for poly in truck_polygons:
        center = poly.centroid.coords[0]
        matched = False

        for tid, data in active_trucks.items():
            prev_center = data["polygon"].centroid.coords[0]
            if np.linalg.norm(np.array(center) - np.array(prev_center)) < 50:
                new_map[tid] = poly
                matched = True
                break

        if not matched:
            tid = f"Truck {truck_counter}"
            truck_counter += 1
            new_map[tid] = poly

    return new_map


def update_cylinder_stock(delta):
    inv, _ = CylinderInventory.objects.get_or_create(pk=1)
    inv.count += delta
    inv.last_updated = timezone.now()
    inv.save()
    print(f"📦 Inventory updated: {inv.count}")
    return inv.count


def log_truck_activity(delta, truck_id=None):
    inv, _ = CylinderInventory.objects.get_or_create(pk=1)
    truck_type = "loaded" if delta < 0 else "unloaded"

    TruckLog.objects.create(
        truck_type=truck_type,
        cylinder_count=abs(delta),
        cylinders_left=inv.count,
        total_cylinders=inv.count + abs(delta) if truck_type == "loaded" else inv.count,
        truck_identifier=truck_id
    )
    print(f"🚛 Truck {truck_id or 'N/A'} logged: {truck_type} — {abs(delta)} cylinders")


def detect_and_classify_cylinders(frame, confidence_threshold=0.4):
    global active_trucks

    import tempfile

    # Save frame to a secure temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, frame)

    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": temp_path},
            use_cache=False
        )

        predictions = result[0]["predictions"]["predictions"]
        if not predictions:
            print("⚠️ No objects detected.")
            return frame, {"cylinder_count": 0, "truck_count": 0}

        trucks = [p for p in predictions if p["class"] == "truck" and p["confidence"] > confidence_threshold]
        cylinders = [p for p in predictions if p["class"] == "cylinder" and p["confidence"] > confidence_threshold]

        truck_polygons = [_get_bounding_polygon(t) for t in trucks]
        truck_id_map = assign_truck_ids(truck_polygons)
        truck_cylinder_counts = defaultdict(int)

        for cyl in cylinders:
            cyl_point = Point(cyl["x"], cyl["y"])
            for tid, poly in truck_id_map.items():
                if poly.contains(cyl_point):
                    truck_cylinder_counts[tid] += 1
                    x, y, w, h = cyl["x"], cyl["y"], cyl["width"], cyl["height"]
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Cylinder ({tid})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

        # Draw bounding boxes and labels for trucks
        for tid, poly in truck_id_map.items():
            minx, miny, maxx, maxy = poly.bounds
            cv2.rectangle(frame, (int(minx), int(miny)), (int(maxx), int(maxy)), (255, 0, 0), 2)
            cv2.putText(frame, f"{tid} [{truck_cylinder_counts[tid]} cyls]", (int(minx), int(miny - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Print per-truck summary
        print("\n📊 Truck Cylinder Summary:")
        for tid, count in truck_cylinder_counts.items():
            print(f"{tid} has {count} items")

        # Handle disappeared trucks
        current_ids = set(truck_id_map.keys())
        for tid in list(active_trucks.keys()):
            if tid not in current_ids:
                active_trucks[tid]["missed"] += 1
                if active_trucks[tid]["missed"] >= MAX_MISSED_FRAMES:
                    delta = -active_trucks[tid]["cylinders"]
                    update_cylinder_stock(delta)
                    log_truck_activity(delta, truck_id=tid)
                    print(f"✅ {tid} departed → {abs(delta)} cylinders")
                    del active_trucks[tid]
            else:
                active_trucks[tid]["missed"] = 0

        # Update active trucks with current frame data
        for tid, poly in truck_id_map.items():
            active_trucks[tid] = {
                "polygon": poly,
                "cylinders": truck_cylinder_counts[tid],
                "missed": 0
            }

        # Save the annotated frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_filename = os.path.join(OUTPUT_FRAMES, f"frame_{timestamp}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"🖼️ Frame saved to {frame_filename}")

        return frame, {
            "cylinder_count": sum(truck_cylinder_counts.values()),
            "truck_count": len(truck_cylinder_counts)
        }

    except Exception as e:
        import traceback
        print(f"❌ Roboflow error: {e}")
        traceback.print_exc()
        return frame, {"cylinder_count": 0, "truck_count": 0}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
