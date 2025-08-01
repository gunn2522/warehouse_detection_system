# import cv2
# import numpy as np
#
# # === Load calibration ===
# data = np.load("fisheye_calibration_result.npz")
# K = data["K"]
# D = data["D"]
#
# # ✅ Ensure distortion coefficients are properly shaped
# D = D.astype(np.float64).reshape(4, 1)
#
# # === Connect to Fisheye Camera ===
# cap = cv2.VideoCapture("rtsp://admin:hikvision101@192.168.169.216:554/Streaming/channels/101")
# if not cap.isOpened():
#     print("❌ Cannot open RTSP stream.")
#     exit()
#
# # === Read one frame to get dimensions ===
# ret, frame = cap.read()
# if not ret:
#     print("❌ Failed to read frame.")
#     cap.release()
#     exit()
#
# h, w = frame.shape[:2]
#
# # === Create Undistortion Maps ===
# balance = 0.0  # 0 = max crop, 1 = max FOV
# new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
#     K, D, (w, h), np.eye(3), balance=balance
# )
# map1, map2 = cv2.fisheye.initUndistortRectifyMap(
#     K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
# )
#
# print("🎥 Undistortion stream started. Press [q] or [ESC] to quit.")
#
# # === Stream Loop ===
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Frame capture failed.")
#         break
#
#     # === Undistort frame ===
#     undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
#
#     # === Show Views ===
#     cv2.imshow("🎥 Original Fisheye", frame)
#     cv2.imshow("📐 Undistorted View", undistorted)
#
#     key = cv2.waitKey(1)
#     if key == 27 or key == ord('q'):
#         break
#
# # === Cleanup ===
# cap.release()
# cv2.destroyAllWindows()
# print("✅ Stream closed.")

import cv2
import numpy as np

# === Load calibration parameters ===
data = np.load("fisheye_calibration_result.npz")
K = data["K"]
D = data["D"].astype(np.float64).reshape(4, 1)

print("📌 Calibration parameters loaded.")
print("K =\n", K)
print("D =\n", D)

# === RTSP stream settings ===
rtsp_url = "rtsp://admin:hikvision101@192.168.169.244/Streaming/channels/101"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("❌ Cannot open RTSP stream.")
    exit()

# === Read one frame to get dimensions ===
ret, frame = cap.read()
if not ret:
    print("❌ Failed to read initial frame.")
    cap.release()
    exit()

h, w = frame.shape[:2]
print(f"🎞️ RTSP resolution confirmed: {w} x {h}")

# === Undistortion Maps ===
balance = 0.6  # 0 = max crop, 1 = max FOV
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

# === Chessboard settings ===
chessboard_size = (6, 8)

print("🎥 Undistortion stream started. Press [q] or [ESC] to quit.")

# === Main loop ===
frame_count = 0
fail_count = 0
max_fails = 10
found_fisheye = 0
found_undistorted = 0
sample_saved = False

while True:
    ret, frame = cap.read()
    if not ret:
        fail_count += 1
        print(f"⚠️ Frame read failed ({fail_count}/{max_fails})")
        if fail_count >= max_fails:
            print("❌ Too many frame failures. Exiting.")
            break
        continue

    fail_count = 0
    frame_count += 1

    # === Undistort ===
    undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    if np.mean(undistorted) < 10:
        print(f"⚫ Frame #{frame_count} skipped — undistorted too dark.")
        continue

    # === Convert to grayscale ===
    gray_fish = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_undist = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    # === Detect chessboards ===
    found_fish, corners_fish = cv2.findChessboardCorners(gray_fish, chessboard_size)
    found_undist, corners_undist = cv2.findChessboardCorners(gray_undist, chessboard_size)

    # === Refine and draw ===
    if found_fish:
        found_fisheye += 1
        corners_fish = cv2.cornerSubPix(gray_fish, corners_fish, (3, 3), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        cv2.drawChessboardCorners(frame, chessboard_size, corners_fish, found_fish)

    if found_undist:
        found_undistorted += 1
        corners_undist = cv2.cornerSubPix(gray_undist, corners_undist, (3, 3), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        cv2.drawChessboardCorners(undistorted, chessboard_size, corners_undist, found_undist)

    # === Save one sample ===
    if found_fish and found_undist and not sample_saved:
        cv2.imwrite("sample_fisheye.jpg", frame)
        cv2.imwrite("sample_undistorted.jpg", undistorted)
        print("📸 Saved sample detection images.")
        sample_saved = True

    # === Display ===
    combined = np.hstack((undistorted, frame))
    resized = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("🪞 Undistorted (Left) | Fisheye (Right)", resized)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        print("⏹️ Exit by user.")
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

print("\n✅ Stream closed.")
print(f"📊 Total frames: {frame_count}")
print(f"✔️ Chessboards found in Fisheye: {found_fisheye}")
print(f"✔️ Chessboards found in Undistorted: {found_undistorted}")
