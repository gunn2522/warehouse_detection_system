# import cv2
# import numpy as np
# import os
# import glob
#
# # === CONFIGURATION ===
# CHECKERBOARD = (6, 8)  # (columns, rows) of inner corners
# square_size = 0.025  # in meters
# image_dir = "final_calib"
# output_dir = "calib_final_check"
# os.makedirs(output_dir, exist_ok=True)
#
# # === Prepare object points (3D) ===
# objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# objp *= square_size
#
# objpoints = []  # 3D points
# imgpoints = []  # 2D points
#
# # === Load Images ===
# images = glob.glob(os.path.join(image_dir, "*.bmp")) + glob.glob(os.path.join(image_dir, "*.jpg"))
# print(f"📁 Found {len(images)} images.")
#
# if not images:
#     print("❌ No calibration images found.")
#     exit()
#
# # === Detect Checkerboards ===
# for fname in images:
#     img = cv2.imread(fname)
#     if img is None:
#         print(f"⚠️ Cannot read image: {fname}")
#         continue
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Optional: Contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)
#
#     # Try both detection methods
#     ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags=0)
#     if not ret:
#         ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
#                                                  cv2.CALIB_CB_ADAPTIVE_THRESH +
#                                                  cv2.CALIB_CB_NORMALIZE_IMAGE)
#
#     if ret:
#         objpoints.append(objp.copy())
#         corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
#                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
#         imgpoints.append(corners2)
#
#         cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
#         out_path = os.path.join(output_dir, os.path.basename(fname))
#         cv2.imwrite(out_path, img)
#         print(f"✅ Checkerboard detected: {os.path.basename(fname)}")
#         cv2.imshow("Detected", img)
#         cv2.waitKey(100)
#     else:
#         print(f"❌ Not detected: {os.path.basename(fname)}")
#
# cv2.destroyAllWindows()
#
# # === Calibration ===
# N_OK = len(objpoints)
# print(f"\n📸 Using {N_OK} valid images for calibration.")
#
# if N_OK < 5:
#     print("❌ Need at least 5 valid detections.")
#     exit()
#
# # === Diagnostic Check ===
# print("🔍 Object and image point shapes:")
# for i in range(N_OK):
#     print(f" Image {i+1}: objp = {objpoints[i].shape}, imgp = {imgpoints[i].shape}")
#
# K = np.zeros((3, 3))
# D = np.zeros((4, 1))
# rvecs = []
# tvecs = []
# img_shape = gray.shape[::-1]
#
# # === Fisheye Calibration Flags ===
# flags = (
#     cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
#     cv2.fisheye.CALIB_FIX_SKEW
#     # Removed CALIB_CHECK_COND to reduce strictness
# )
#
# try:
#     rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
#         objpoints,
#         imgpoints,
#         img_shape,
#         K,
#         D,
#         rvecs,
#         tvecs,
#         flags,
#         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
#     )
#
#     print("\n✅ Fisheye calibration successful!")
#     print("📏 RMS error:", rms)
#     print("🎯 Camera matrix (K):\n", K)
#     print("🎯 Distortion coefficients (D):\n", D)
#
#     np.savez("fisheye_calibration_result.npz", K=K, D=D, rms=rms)
#     print("💾 Saved to fisheye_calibration_result.npz")
#
#     # === Undistortion preview ===
#     img = cv2.imread(images[0])
#     h, w = img.shape[:2]
#     new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
#     undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
#
#     cv2.imshow("Original", img)
#     cv2.imshow("Undistorted", undistorted)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# except cv2.error as e:
#     print("❌ Fisheye calibration failed.")
#     print("🔎 OpenCV error details:\n", e)
#
#     # === Optional fallback: standard calibration ===
#     print("\n⚠️ Attempting standard calibration as fallback...")
#     try:
#         rms_std, K_std, D_std, rvecs_std, tvecs_std = cv2.calibrateCamera(
#             [o.reshape(-1, 3) for o in objpoints],
#             [i.reshape(-1, 2) for i in imgpoints],
#             img_shape,
#             None,
#             None
#         )
#         print("✅ Standard calibration successful.")
#         print("📏 RMS error:", rms_std)
#         print("🎯 Camera matrix (K):\n", K_std)
#         print("🎯 Distortion coefficients (D):\n", D_std)
#
#         np.savez("standard_calibration_result.npz", K=K_std, D=D_std, rms=rms_std)
#         print("💾 Saved to standard_calibration_result.npz")
#     except cv2.error as e2:
#         print("❌ Standard calibration also failed.")
#         print("🔎 OpenCV error details:\n", e2)






#


import cv2
import numpy as np
import glob
import os

# === OpenCV Version Check ===
major, minor, _ = cv2.__version__.split(".")
assert int(major) >= 3, "The fisheye module requires OpenCV version >= 3.0.0"

# === Configuration ===
CHECKERBOARD = (6, 8)  # inner corners (not squares!)
square_size = 2.5  # meters per square
image_dir = "ff"  # Folder with calibration images
show_detections = True  # Show each detected frame
use_strict_check = False  # Optional: toggle strict checking

# === Calibration Flags ===
calibration_flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
    (cv2.fisheye.CALIB_CHECK_COND if use_strict_check else 0) +
    cv2.fisheye.CALIB_FIX_SKEW
)

# === Prepare Object Points ===
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []
_img_shape = None

# === Load Images ===
images = glob.glob(os.path.join(image_dir, "*.bmp")) + \
         glob.glob(os.path.join(image_dir, "*.jpg")) + \
         glob.glob(os.path.join(image_dir, "*.png"))

if not images:
    raise RuntimeError("❌ No calibration images found in folder: " + image_dir)

print(f"📷 Found {len(images)} images for calibration.")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"❌ Could not read image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if _img_shape is None:
        _img_shape = gray.shape
    elif _img_shape != gray.shape:
        print(f"❌ Image size mismatch: {fname}")
        continue

    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        corners_subpix = cv2.cornerSubPix(
            gray, corners, (3, 3), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        )
        objpoints.append(objp)
        imgpoints.append(corners_subpix)
        print(f"✅ Corners detected in: {fname}")

        if show_detections:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_subpix, ret)
            cv2.imshow("Detected Checkerboard", vis)
            cv2.waitKey(300)
    else:
        print(f"⚠️  No corners detected in: {fname}")

cv2.destroyAllWindows()

# === Validation Before Calibration ===
N_OK = len(objpoints)
print(f"\n📊 Valid images with detected corners: {N_OK}")

if N_OK < 5:
    raise RuntimeError(f"❌ Only {N_OK} valid images. Need at least 5 for fisheye calibration.")

if len(objpoints) != len(imgpoints):
    raise RuntimeError("❌ Mismatch between object points and image points!")

print("📐 Image size for calibration:", _img_shape[::-1])

# === Calibration ===
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

print("\n⚙️ Starting fisheye calibration...")

try:
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        _img_shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
except cv2.error as e:
    print("❌ OpenCV calibration error:", e)
    exit()

# === Output Calibration Results ===
print("\n✅ Calibration Successful!")
print(f"📉 RMS Error: {rms:.6f}")
print("📏 Image Dimensions:", _img_shape[::-1])
print("📷 K = np.array(" + str(K.tolist()) + ")")
print("🎯 D = np.array(" + str(D.tolist()) + ")")

# === Save Calibration ===
np.savez("fisheye_calibration_result.npz", K=K, D=D, DIM=_img_shape[::-1])
print("💾 Saved to fisheye_calibration_result.npz")






# import cv2
# import numpy as np
# import os
# from datetime import datetime
#
# # === CONFIGURATION ===
# CHECKERBOARD = (6, 8)  # inner corners (columns, rows)
# square_size = 0.025  # physical square size in meters
# save_dir = "detected_boards"
# os.makedirs(save_dir, exist_ok=True)
#
# # === Load Calibration ===
# calib = np.load("fisheye_calibration_result.npz")
# K = calib["K"]
# D = calib["D"]
# print("📷 Calibration loaded.")
# print("K:\n", K)
# print("D:\n", D)
#
# # === Connect to RTSP Camera ===
# cap = cv2.VideoCapture("rtsp://admin:hikvision101@192.168.169.244/Streaming/channels/101?tcp")
# if not cap.isOpened():
#     print("❌ Cannot open RTSP stream.")
#     exit()
#
# # === Read Frame to Get Dimensions ===
# ret, frame = cap.read()
# if not ret:
#     print("❌ Failed to read frame.")
#     cap.release()
#     exit()
#
# h, w = frame.shape[:2]
#
# # === Undistortion Setup ===
# balance = 0.0  # 0.0: fully rectilinear, 1.0: keep fisheye FOV
# new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
# map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
#
# print("🎥 RTSP camera ready. Press [q] or [ESC] to quit.")
#
# # === Prepare Checkerboard Object Points ===
# objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# objp *= square_size
#
# saved_count = 0
#
# # === Main Loop ===
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Frame capture failed.")
#         break
#
#     # === Undistort Fisheye to Rectilinear ===
#     rectilinear = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#
#     # === Optional: Crop valid area from remapped image ===
#     gray_crop = cv2.cvtColor(rectilinear, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray_crop, 1, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         x, y, w_crop, h_crop = cv2.boundingRect(contours[0])
#         rectilinear_cropped = rectilinear[y:y + h_crop, x:x + w_crop]
#     else:
#         rectilinear_cropped = rectilinear  # fallback if no valid contour
#
#     # === Checkerboard Detection ===
#     gray = cv2.cvtColor(rectilinear_cropped, cv2.COLOR_BGR2GRAY)
#     found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags=0)
#     if not found:
#         found, corners = cv2.findChessboardCorners(
#             gray, CHECKERBOARD,
#             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
#         )
#
#     if found:
#         # Refine corners
#         corners2 = cv2.cornerSubPix(
#             gray, corners, (3, 3), (-1, -1),
#             criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         )
#         # Draw detected corners
#         cv2.drawChessboardCorners(rectilinear_cropped, CHECKERBOARD, corners2, found)
#
#         # === Save the frame ===
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#         filename = os.path.join(save_dir, f"checkerboard_{timestamp}.png")
#         cv2.imwrite(filename, rectilinear_cropped)
#         saved_count += 1
#         print(f"💾 Saved checkerboard frame: {filename}")
#
#     # === Display ===
#     cv2.imshow("📷 Fisheye (Original)", frame)
#     cv2.imshow("📐 Rectilinear Undistorted", rectilinear_cropped)
#
#     key = cv2.waitKey(1)
#     if key == 27 or key == ord('q'):
#         break
#
# # === Cleanup ===
# cap.release()
# cv2.destroyAllWindows()
# print(f"📸 Total saved checkerboard frames: {saved_count}")
