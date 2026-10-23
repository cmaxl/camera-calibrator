#!/usr/bin/env python3
"""
camera_calibration.py
=====================
Calibrates a camera from a folder of chessboard / charuco images and outputs
results ready for use in ROS 2 (camera_info topic, YAML format).

Supported distortion models
----------------------------
  plumb_bob      – OpenCV radial + tangential (k1,k2,p1,p2,k3) — ROS default
  rational_poly  – OpenCV rational polynomial (k1–k6, p1, p2)
  equidistant    – fisheye / equidistant (k1–k4) via cv2.fisheye
  none           – undistorted (all zeros)

Usage
-----
  python camera_calibration.py \
      --images_dir ./calib_images \
      --board_type chessboard \
      --board_cols 9 \
      --board_rows 6 \
      --square_size 0.025 \
      --distortion_model plumb_bob \
      --reproj_threshold 0.5 \
      --output camera_info.yaml

Optional flags
--------------
  --reproj_threshold T    Remove images whose reprojection error > T px and
                          recalibrate once (optional; no default)
  --fix_principal_point   Fix cx, cy to image centre
  --fix_aspect_ratio      Force fx == fy
  --fix_k3                Fix k3 = 0
  --zero_tangent_dist     Fix p1 = p2 = 0
  --subpix_winsize N      Corner sub-pixel refinement window half-size (default 5)
  --show_corners          Display detected corners per image
"""

import argparse
import os
import sys
import glob
import yaml
import numpy as np
import cv2

# ─────────────────────────────────────────────────────── argument parsing ──

def parse_args():
    p = argparse.ArgumentParser(
        description="Camera calibration → ROS 2 camera_info YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--images_dir", required=True,
                   help="Folder containing calibration images (jpg/png/bmp)")
    p.add_argument("--board_type", choices=["chessboard", "charuco"],
                   default="chessboard", help="Calibration target type")
    p.add_argument("--board_cols", type=int, default=9,
                   help="Number of inner corners / squares along columns "
                        "(chessboard: inner corners; charuco: total squares)")
    p.add_argument("--board_rows", type=int, default=6,
                   help="Number of inner corners / squares along rows")
    p.add_argument("--square_size", type=float, default=0.025,
                   help="Size of one square in metres (default 0.025 = 25 mm)")
    p.add_argument("--marker_size", type=float, default=None,
                   help="ChArUco: ArUco marker size in metres (default 0.75 * square_size)")
    p.add_argument("--aruco_dict", default="DICT_6X6_250",
                   help="ChArUco: ArUco dictionary name (default DICT_6X6_250)")
    p.add_argument("--distortion_model",
                   choices=["plumb_bob", "rational_poly", "equidistant", "none"],
                   default="plumb_bob",
                   help="Distortion model (default plumb_bob)")
    p.add_argument("--reproj_threshold", type=float, default=None,
                   metavar="T",
                   help="(Optional) Remove images with per-image RMS reprojection "
                        "error > T px and run one recalibration pass")
    p.add_argument("--output", default="camera_info.yaml",
                   help="Output YAML file path (default camera_info.yaml)")
    p.add_argument("--camera_name", default="camera",
                   help="ROS 2 camera name field (default 'camera')")
    # Calibration flags
    p.add_argument("--fix_principal_point", action="store_true",
                   help="Fix principal point at image centre")
    p.add_argument("--fix_aspect_ratio", action="store_true",
                   help="Force fx == fy")
    p.add_argument("--fix_k3", action="store_true",
                   help="Fix k3 = 0")
    p.add_argument("--zero_tangent_dist", action="store_true",
                   help="Fix tangential distortion p1=p2=0")
    p.add_argument("--subpix_winsize", type=int, default=5,
                   help="Half window size for sub-pixel corner refinement (default 5)")
    p.add_argument("--show_corners", action="store_true",
                   help="Display detected corners for each image")
    return p.parse_args()

# ─────────────────────────────────────────────────────── helpers ──

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")

def load_images(images_dir):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
        paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    paths = sorted(set(paths))
    if not paths:
        sys.exit(f"[ERROR] No images found in: {images_dir}")
    return paths


def build_calib_flags(args, model):
    flags = 0
    if args.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if args.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if model == "rational_poly":
        flags |= cv2.CALIB_RATIONAL_MODEL
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if args.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    return flags


def subpix_criteria(winsize):
    return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ─────────────────────────────────────────────────────── chessboard ──

def detect_chessboard(paths, rows, cols, square_size, subpix_ws, show):
    pattern_size = (cols, rows)
    obj_pts_proto = np.zeros((rows * cols, 3), np.float32)
    obj_pts_proto[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points, img_points = [], []
    image_size = None
    ok_paths = []

    criteria = subpix_criteria(subpix_ws)
    win = (subpix_ws, subpix_ws)

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [WARN] Cannot read: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (w, h)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not ret:
            print(f"  [SKIP] No corners found: {os.path.basename(path)}")
            continue

        corners_sub = cv2.cornerSubPix(gray, corners, win, (-1, -1), criteria)
        obj_points.append(obj_pts_proto.copy())
        img_points.append(corners_sub)
        ok_paths.append(path)

        if show:
            disp = img.copy()
            cv2.drawChessboardCorners(disp, pattern_size, corners_sub, ret)
            cv2.imshow("Corners", disp)
            cv2.waitKey(300)

    if show:
        cv2.destroyAllWindows()

    return obj_points, img_points, image_size, ok_paths


# ─────────────────────────────────────────────────────── charuco ──

def detect_charuco(paths, rows, cols, square_size, marker_size, aruco_dict_name, subpix_ws, show):
    aruco_dict_id = getattr(cv2.aruco, aruco_dict_name, None)
    if aruco_dict_id is None:
        sys.exit(f"[ERROR] Unknown ArUco dictionary: {aruco_dict_name}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)

    board = cv2.aruco.CharucoBoard((cols, rows), square_size, marker_size, aruco_dict)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    all_corners, all_ids = [], []
    image_size = None
    ok_paths = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [WARN] Cannot read: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        if marker_ids is None or len(marker_ids) < 4:
            print(f"  [SKIP] Too few markers: {os.path.basename(path)}")
            continue

        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board)
        if not ret or ch_ids is None or len(ch_ids) < 4:
            print(f"  [SKIP] Too few ChArUco corners: {os.path.basename(path)}")
            continue

        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
        ok_paths.append(path)

        if show:
            disp = img.copy()
            cv2.aruco.drawDetectedCornersCharuco(disp, ch_corners, ch_ids)
            cv2.imshow("ChArUco", disp)
            cv2.waitKey(300)

    if show:
        cv2.destroyAllWindows()

    return all_corners, all_ids, board, image_size, ok_paths


# ───────────────────────���─────────────────────────────── calibration ──

def calibrate_standard(obj_points, img_points, image_size, flags):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags)
    return rms, K, dist, rvecs, tvecs


def calibrate_fisheye(obj_points, img_points, image_size, args):
    obj_pts_fe = [o.reshape(-1, 1, 3) for o in obj_points]
    img_pts_fe = [i.reshape(-1, 1, 2) for i in img_points]

    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    if args.fix_principal_point:
        flags |= cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_pts_fe, img_pts_fe, image_size, K, D, flags=flags)
    return rms, K, D, rvecs, tvecs


def calibrate_charuco_standard(all_corners, all_ids, board, image_size, flags):
    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None, flags=flags)
    return rms, K, dist, rvecs, tvecs


def per_image_reprojection_error(obj_points, img_points, rvecs, tvecs, K, dist, model):
    errors = []
    for i, (op, ip) in enumerate(zip(obj_points, img_points)):
        if model == "equidistant":
            proj, _ = cv2.fisheye.projectPoints(
                op.reshape(-1, 1, 3), rvecs[i], tvecs[i], K, dist)
            proj = proj.reshape(-1, 2)
            ip2 = ip.reshape(-1, 2)
        else:
            proj, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], K, dist)
            proj = proj.reshape(-1, 2)
            ip2 = ip.reshape(-1, 2)
        err = np.sqrt(np.mean((proj - ip2) ** 2))
        errors.append(err)
    return errors


# ─────────────────────────────────────────────────────── ROS 2 YAML output ──

def distortion_model_name(model):
    if model == "rational_poly":
        return "rational_polynomial"
    if model == "equidistant":
        return "equidistant"
    return "plumb_bob"


def make_ros2_yaml(camera_name, image_size, K, dist_vec, model, rvecs, tvecs):
    w, h = image_size
    P = np.zeros((3, 4))
    P[:3, :3] = K
    R = np.eye(3)

    extrinsics = []
    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        R_mat, _ = cv2.Rodrigues(rv)
        T = tv.flatten()
        Rt = np.eye(4)
        Rt[:3, :3] = R_mat
        Rt[:3, 3] = T
        extrinsics.append({
            "image_index": i,
            "rvec": rv.flatten().tolist(),
            "tvec": T.tolist(),
            "rotation_matrix": R_mat.tolist(),
            "transform_camera_to_world_4x4": np.linalg.inv(Rt).tolist(),
        })

    return {
        "image_width": int(w),
        "image_height": int(h),
        "camera_name": camera_name,
        "camera_matrix": {
            "rows": 3, "cols": 3,
            "data": K.flatten().tolist(),
        },
        "distortion_model": distortion_model_name(model),
        "distortion_coefficients": {
            "rows": 1, "cols": int(dist_vec.size),
            "data": dist_vec.flatten().tolist(),
        },
        "rectification_matrix": {
            "rows": 3, "cols": 3,
            "data": R.flatten().tolist(),
        },
        "projection_matrix": {
            "rows": 3, "cols": 4,
            "data": P.flatten().tolist(),
        },
        "extrinsics": extrinsics,
    }


# ─────────────────────────────────────────────────────── printing ──

def print_reproj_errors(ok_paths, errors, rms, threshold=None):
    """Print per-image reprojection errors and overall RMS in a consistent table."""
    print(f"  {'─'*50}")
    print(f"  Per-image reprojection error (RMS, pixels)")
    print(f"  {'─'*50}")
    for path, err in zip(ok_paths, errors):
        over = threshold is not None and err > threshold
        marker = f"  ← ABOVE {threshold:.2f} px" if over else ("  ← HIGH" if err > 1.0 else "")
        print(f"  {os.path.basename(path):40s}  {err:.4f} px{marker}")
    print(f"  {'─'*50}")
    print(f"  Overall RMS: {rms:.4f} px")
    print(f"  {'─'*50}")


def print_intrinsics(K, dist):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"\nIntrinsics (camera matrix K):")
    print(f"  fx = {fx:.4f} px   fy = {fy:.4f} px")
    print(f"  cx = {cx:.4f} px   cy = {cy:.4f} px")
    print(f"  Skew (s) = {K[0,1]:.6f}")
    print(f"  Distortion coefficients: {dist.flatten().tolist()}")


# ─────────────────────────────────────────────────────── main ──

def run_calibration(obj_points, img_points, image_size, model, args, charuco_data=None):
    """Run the appropriate calibration and return (rms, K, dist, rvecs, tvecs)."""
    if charuco_data is not None:
        all_corners, all_ids, board = charuco_data
        flags = build_calib_flags(args, model)
        return calibrate_charuco_standard(all_corners, all_ids, board, image_size, flags)
    if model == "equidistant":
        return calibrate_fisheye(obj_points, img_points, image_size, args)
    flags = build_calib_flags(args, model)
    if model == "none":
        flags |= (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                  cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
                  cv2.CALIB_ZERO_TANGENT_DIST)
    return calibrate_standard(obj_points, img_points, image_size, flags)


def main():
    args = parse_args()
    model = args.distortion_model

    if args.marker_size is None:
        args.marker_size = 0.75 * args.square_size

    print(f"\n{'='*60}")
    print(f"  Camera Calibration Tool → ROS 2 camera_info")
    print(f"{'='*60}")
    print(f"  Images dir       : {args.images_dir}")
    print(f"  Board type       : {args.board_type}")
    print(f"  Board size       : {args.board_cols} × {args.board_rows}")
    print(f"  Square size      : {args.square_size*1000:.1f} mm")
    print(f"  Distortion model : {model}")
    thr_str = f"{args.reproj_threshold:.4f} px" if args.reproj_threshold is not None else "not set"
    print(f"  Reproj threshold : {thr_str}")
    print(f"{'='*60}\n")

    paths = load_images(args.images_dir)
    print(f"Found {len(paths)} image(s).")

    # ── detect corners ──
    charuco_data = None
    if args.board_type == "chessboard":
        obj_points, img_points, image_size, ok_paths = detect_chessboard(
            paths, args.board_rows, args.board_cols,
            args.square_size, args.subpix_winsize, args.show_corners)
        if len(ok_paths) < 3:
            sys.exit("[ERROR] Need at least 3 images with detected corners.")
    else:
        all_corners, all_ids, board, image_size, ok_paths = detect_charuco(
            paths, args.board_rows, args.board_cols,
            args.square_size, args.marker_size, args.aruco_dict,
            args.subpix_winsize, args.show_corners)
        if len(ok_paths) < 3:
            sys.exit("[ERROR] Need at least 3 valid ChArUco images.")
        # build obj/img_points for reprojection error computation
        obj_points, img_points = [], []
        for ch_ids, ch_corners in zip(all_ids, all_corners):
            op, ip = board.matchImagePoints(ch_corners, ch_ids)
            obj_points.append(op)
            img_points.append(ip)
        charuco_data = (all_corners, all_ids, board)

    # ── initial calibration ──
    total = len(paths)
    print(f"\nUsed {len(ok_paths)} / {total} images for initial calibration.")

    rms, K, dist, rvecs, tvecs = run_calibration(
        obj_points, img_points, image_size, model, args,
        charuco_data=charuco_data)

    errors = per_image_reprojection_error(obj_points, img_points, rvecs, tvecs, K, dist, model)

    print(f"\n── Initial calibration ──")
    print_reproj_errors(ok_paths, errors, rms, threshold=args.reproj_threshold)
    print_intrinsics(K, dist)

    # ── optional threshold-based outlier removal + recalibration ──
    if args.reproj_threshold is not None:
        rejected = [(p, e) for p, e in zip(ok_paths, errors) if e > args.reproj_threshold]

        if not rejected:
            print(f"\n✓ All images are within the threshold of {args.reproj_threshold:.4f} px. "
                  f"No recalibration needed.")
        else:
            print(f"\n── Outlier removal (threshold = {args.reproj_threshold:.4f} px) ──")
            print(f"  Removing {len(rejected)} image(s):")
            for p, e in rejected:
                print(f"    {os.path.basename(p):40s}  {e:.4f} px")

            keep_mask = [e <= args.reproj_threshold for e in errors]
            ok_paths_f   = [p for p, k in zip(ok_paths,   keep_mask) if k]
            obj_points_f = [o for o, k in zip(obj_points, keep_mask) if k]
            img_points_f = [i for i, k in zip(img_points, keep_mask) if k]

            if len(ok_paths_f) < 3:
                print("  [WARN] Fewer than 3 images remain after filtering. "
                      "Skipping recalibration — keeping initial results.")
            else:
                # rebuild charuco_data if needed
                if charuco_data is not None:
                    all_corners_f = [c for c, k in zip(all_corners, keep_mask) if k]
                    all_ids_f     = [i for i, k in zip(all_ids,     keep_mask) if k]
                    charuco_data_f = (all_corners_f, all_ids_f, board)
                else:
                    charuco_data_f = None

                print(f"  Recalibrating with {len(ok_paths_f)} image(s)…")

                rms, K, dist, rvecs, tvecs = run_calibration(
                    obj_points_f, img_points_f, image_size, model, args,
                    charuco_data=charuco_data_f)

                errors_f = per_image_reprojection_error(
                    obj_points_f, img_points_f, rvecs, tvecs, K, dist, model)

                print(f"\n── Recalibration result ({len(ok_paths_f)} / {total} images) ──")
                print_reproj_errors(ok_paths_f, errors_f, rms, threshold=args.reproj_threshold)
                print_intrinsics(K, dist)

                # update variables used for YAML output
                ok_paths   = ok_paths_f
                obj_points = obj_points_f
                img_points = img_points_f
                errors     = errors_f

    # ── write YAML ──
    ros2_data = make_ros2_yaml(args.camera_name, image_size, K, dist, model, rvecs, tvecs)
    with open(args.output, "w") as f:
        yaml.dump(ros2_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ ROS 2 camera_info YAML written to: {args.output}")
    print(f"\nTo use in ROS 2, publish with:")
    print(f"  ros2 run camera_info_manager camera_info_publisher \\")
    print(f"      --ros-args -p camera_info_url:=file://$(pwd)/{args.output}")
    print(f"\nOr load in your node:")
    print(f"  camera_info_manager.CameraInfoManager(self, '{args.camera_name}',")
    print(f"      camera_info_url='file:///path/to/{args.output}')")


if __name__ == "__main__":
    main()
