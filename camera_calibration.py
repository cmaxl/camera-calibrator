#!/usr/bin/env python3
"""
camera_calibration.py
=====================
Calibrates a camera from a folder of chessboard or ChArUco images and writes
the result as a ROS 2 style camera_info YAML file.

Requires OpenCV >= 4.7 for ChArUco support (the old cv2.aruco.interpolateCornersCharuco
/ calibrateCameraCharuco API no longer exists there).

Supported distortion models
---------------------------
  plumb_bob      OpenCV radial + tangential (k1, k2, p1, p2, k3)  - ROS default
  rational_poly  OpenCV rational polynomial (k1-k6, p1, p2)
  equidistant    Fisheye / equidistant (k1-k4) via cv2.fisheye
  none           No distortion (all coefficients fixed to zero)

Example
-------
  python camera_calibration.py \
      --images_dir ./calib_images \
      --board_type charuco \
      --board_cols 5 --board_rows 7 \
      --square_size 0.04 --marker_size 0.02 \
      --reproj_threshold 0.5 \
      --output camera_info.yaml

Board size conventions
----------------------
  chessboard : board_cols / board_rows = number of INNER corners
  charuco    : board_cols / board_rows = number of SQUARES (squaresX / squaresY)

ChArUco boards use the legacy marker layout by default (--legacy_pattern).
Pass --no-legacy_pattern if your board was generated with OpenCV >= 4.6 defaults.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

# ───────────────────────────────────────────────────────────── constants ──

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")

MIN_VIEWS = 3  # fewest usable images we accept for a calibration
MIN_CHARUCO_CORNERS = 6  # fewest ChArUco corners for an image to count
HIGH_ERROR_PX = 1.0  # per-image RMS above this is flagged as "HIGH"
PREVIEW_DELAY_MS = 300  # how long --show_corners displays each image

# Flags that only make sense for the pinhole (non-fisheye) models.
PINHOLE_ONLY_OPTIONS = ("fix_aspect_ratio", "fix_k3", "zero_tangent_dist")


# ─────────────────────────────────────────────────────────── data classes ──


@dataclass
class View:
    """One calibration image: matched 3D board points and 2D detections."""

    path: str
    obj_pts: np.ndarray  # (N, 1, 3) float32, board frame, metres
    img_pts: np.ndarray  # (N, 1, 2) float32, pixels

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class Calibration:
    """Result of one calibration run."""

    rms: float  # overall RMS reprojection error from OpenCV
    K: np.ndarray  # 3x3 camera matrix
    dist: np.ndarray  # distortion coefficients
    rvecs: list  # per-view rotation  (board -> camera)
    tvecs: list  # per-view translation (board -> camera)
    views: list[View]  # views used for this run
    errors: list[float]  # per-view RMS reprojection error (px)


# ─────────────────────────────────────────────────────────── report helpers ──


def _to_list(value):
    """Convert numpy/OpenCV values into JSON-safe Python values."""
    if value is None:
        return None

    if hasattr(value, "tolist"):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return [_to_list(v) for v in value]

    if isinstance(value, (np.floating, np.integer)):
        return value.item()

    return value


def make_report_data(cal, args, image_size, overall_rms=None):
    """
    Build JSON data consumed by calibration_report.py.

    Expected calibration object fields:
      cal.K
      cal.dist
      cal.rvecs
      cal.tvecs
      cal.views
      cal.errors

    Each view is expected to contain:
      view.filename or view.path
      view.obj_pts
      view.img_pts
    """

    width, height = image_size

    views = []

    project_fn = (
        cv2.fisheye.projectPoints
        if args.distortion_model == "equidistant"
        else cv2.projectPoints
    )

    for index, (view, rvec, tvec, rms_error) in enumerate(
        zip(cal.views, cal.rvecs, cal.tvecs, cal.errors)
    ):
        try:
            projected, _ = project_fn(
                view.obj_pts,
                rvec,
                tvec,
                cal.K,
                cal.dist,
            )

            projected = projected.reshape(-1, 2)
            observed = view.img_pts.reshape(-1, 2)
            residuals = observed - projected

        except Exception as exc:
            print(
                f"Warning: could not calculate report residuals for view {index}: {exc}"
            )
            observed = None
            projected = None
            residuals = None

        filename = getattr(view, "filename", None)
        if filename is None:
            filename = getattr(view, "path", None)
        if filename is None:
            filename = f"image_{index:04d}"

        views.append(
            {
                "index": index,
                "filename": str(filename),
                "object_points": _to_list(view.obj_pts),
                "image_points": _to_list(observed),
                "projected_points": _to_list(projected),
                "residual_vectors": _to_list(residuals),
                "rvec": _to_list(rvec),
                "tvec": _to_list(tvec),
                "rms_error_px": float(rms_error),
                "detected_point_count": (
                    int(len(view.img_pts)) if hasattr(view, "img_pts") else None
                ),
            }
        )

    board_info = {
        "board_type": getattr(args, "board_type", None),
        "board_cols": getattr(args, "board_cols", None),
        "board_rows": getattr(args, "board_rows", None),
        "square_size_m": getattr(args, "square_size", None),
        "marker_size_m": getattr(args, "marker_size", None),
    }

    return {
        "schema_version": 1,
        "camera_name": getattr(args, "camera_name", None),
        "image_width": int(width),
        "image_height": int(height),
        "distortion_model": getattr(args, "distortion_model", None),
        "camera_matrix": _to_list(cal.K),
        "distortion_coefficients": _to_list(cal.dist),
        "overall_rms_error_px": (
            float(overall_rms) if overall_rms is not None else None
        ),
        "board": board_info,
        "views": views,
        "calibration_command": " ".join(sys.argv),
        "opencv_version": cv2.__version__,
    }


def write_report_data(cal, args, image_size, yaml_output_path, overall_rms=None):
    """Write diagnostic JSON beside the camera YAML file."""
    yaml_path = Path(yaml_output_path)
    json_path = yaml_path.with_suffix(".report_data.json")

    report_data = make_report_data(
        cal=cal,
        args=args,
        image_size=image_size,
        overall_rms=overall_rms,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return json_path


def run_pdf_report(args, yaml_output_path, report_data_path):
    """Run the separate PDF report generator."""
    yaml_path = Path(yaml_output_path)

    report_output = (
        Path(args.report_output)
        if args.report_output
        else yaml_path.with_name(f"{yaml_path.stem}_report.pdf")
    )

    coordinate_image = Path("coordinate_convention.jpg")

    command = [
        sys.executable,
        args.report_script,
        "--report-data",
        str(report_data_path),
        "--camera-yaml",
        str(yaml_output_path),
        "--output",
        str(report_output),
    ]

    if coordinate_image.exists():
        command.extend(
            [
                "--coordinate-convention-image",
                str(coordinate_image),
            ]
        )
    else:
        print(
            "Warning: coordinate_convention.jpg was not found. "
            "The report will mark the coordinate convention image as Unknown."
        )

    print("\nGenerating calibration PDF report...")
    print(" ".join(command))

    subprocess.run(command, check=True)

    print(f"PDF report written to: {report_output}")


# ──────────────────────────────────────────────────────── argument parsing ──


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera calibration -> ROS 2 camera_info YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--images_dir", required=True, help="Folder containing calibration images"
    )
    p.add_argument(
        "--board_type",
        choices=["chessboard", "charuco"],
        default="chessboard",
        help="Calibration target type",
    )
    p.add_argument(
        "--board_cols",
        type=int,
        default=9,
        help="chessboard: inner corners along X; charuco: squares along X",
    )
    p.add_argument(
        "--board_rows",
        type=int,
        default=6,
        help="chessboard: inner corners along Y; charuco: squares along Y",
    )
    p.add_argument(
        "--square_size",
        type=float,
        default=0.025,
        help="Square side length in metres (default 0.025)",
    )
    p.add_argument(
        "--marker_size",
        type=float,
        default=None,
        help="ChArUco: marker side length in metres (default 0.75 * square_size)",
    )
    p.add_argument(
        "--aruco_dict",
        default="DICT_6X6_250",
        help="ChArUco: ArUco dictionary name (default DICT_6X6_250)",
    )
    p.add_argument(
        "--distortion_model",
        choices=["plumb_bob", "rational_poly", "equidistant", "none"],
        default="plumb_bob",
        help="Distortion model (default plumb_bob)",
    )
    p.add_argument(
        "--reproj_threshold",
        type=float,
        default=None,
        metavar="T",
        help="Drop images whose RMS reprojection error exceeds T px "
        "and recalibrate once (off by default)",
    )
    p.add_argument(
        "--output",
        default="camera_info.yaml",
        help="Output YAML path (default camera_info.yaml)",
    )
    p.add_argument(
        "--camera_name",
        default="camera",
        help="camera_name field in the YAML (default 'camera')",
    )
    p.add_argument(
        "--legacy_pattern",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ChArUco: board uses the legacy (pre-OpenCV 4.6) marker "
        "layout. On by default; use --no-legacy_pattern for "
        "boards generated with the current layout",
    )
    p.add_argument(
        "--save_extrinsics",
        action="store_true",
        help="Also store the per-image board poses in the YAML",
    )

    # Calibration flags
    p.add_argument(
        "--fix_principal_point",
        action="store_true",
        help="Fix the principal point at the image centre",
    )
    p.add_argument(
        "--fix_aspect_ratio",
        action="store_true",
        help="Force fx == fy (pinhole models only)",
    )
    p.add_argument(
        "--fix_k3", action="store_true", help="Fix k3 = 0 (pinhole models only)"
    )
    p.add_argument(
        "--zero_tangent_dist",
        action="store_true",
        help="Fix p1 = p2 = 0 (pinhole models only)",
    )

    p.add_argument(
        "--subpix_winsize",
        type=int,
        default=5,
        help="Chessboard: half window size for sub-pixel refinement (default 5)",
    )
    p.add_argument(
        "--show_corners",
        action="store_true",
        help="Display the detected corners for each image",
    )

    # Report flags
    p.add_argument(
        "--create_report",
        action="store_true",
        help="Generate a PDF camera calibration report after calibration",
    )
    p.add_argument(
        "--report_output",
        default=None,
        help="Optional output PDF path. Default: <output_yaml_stem>_report.pdf",
    )
    p.add_argument(
        "--report_script",
        default="calibration_report.py",
        help="Path to the PDF report generator script",
    )
    return p.parse_args()


# ───────────────────────────────────────────────────────── image handling ──


def find_images(images_dir: str) -> list[str]:
    """Return a sorted list of image paths in `images_dir`."""
    paths: set[str] = set()
    for pattern in IMAGE_PATTERNS:
        # Match both lower- and upper-case extensions on case-sensitive systems.
        paths.update(glob.glob(os.path.join(images_dir, pattern)))
        paths.update(glob.glob(os.path.join(images_dir, pattern.upper())))
    if not paths:
        sys.exit(f"[ERROR] No images found in: {images_dir}")
    return sorted(paths)


def iter_gray_images(paths: list[str]) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (path, grayscale image), skipping unreadable files and any image
    whose resolution differs from the first readable one (mixing resolutions
    silently ruins a calibration)."""
    expected_shape = None
    for path in paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  [WARN] Cannot read: {path}")
            continue
        if expected_shape is None:
            expected_shape = gray.shape
        elif gray.shape != expected_shape:
            print(
                f"  [SKIP] Different resolution "
                f"({gray.shape[1]}x{gray.shape[0]}): {os.path.basename(path)}"
            )
            continue
        yield path, gray


def show_preview(window: str, gray: np.ndarray, draw) -> None:
    """Show `gray` as a colour image after `draw(image)` has annotated it."""
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    draw(canvas)
    cv2.imshow(window, canvas)
    cv2.waitKey(PREVIEW_DELAY_MS)


# ─────────────────────────────────────────────────────────── detection ──


def detect_chessboard(paths, rows, cols, square_size, subpix_ws, show):
    """Find chessboard corners in every image.

    Returns (views, image_size) where image_size is (width, height).
    """
    pattern_size = (cols, rows)

    # The 3D board points are identical for every image: a flat grid at z = 0.
    board_pts = np.zeros((rows * cols, 3), np.float32)
    board_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    board_pts = board_pts.reshape(-1, 1, 3)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    subpix_window = (subpix_ws, subpix_ws)
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    views, image_size = [], None
    for path, gray in iter_gray_images(paths):
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=find_flags)
        if not found:
            print(f"  [SKIP] No corners found: {os.path.basename(path)}")
            continue

        corners = cv2.cornerSubPix(
            gray, corners, subpix_window, (-1, -1), subpix_criteria
        )
        views.append(View(path, board_pts.copy(), corners.astype(np.float32)))

        if show:
            show_preview(
                "Chessboard",
                gray,
                lambda img: cv2.drawChessboardCorners(
                    img,
                    pattern_size,
                    np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2).copy(),
                    np.asarray(found, dtype=np.int32).reshape(-1, 1).copy(),
                ),
            )

    if show:
        cv2.destroyAllWindows()
    return views, image_size


def detect_charuco(
    paths, rows, cols, square_size, marker_size, dict_name, legacy_pattern, show
):
    """Find ChArUco corners in every image (OpenCV >= 4.7 API).

    Partial views are fine: each image may contribute a different number of
    corners. Returns (views, image_size) where image_size is (width, height).
    """
    if not hasattr(cv2.aruco, "CharucoDetector"):
        sys.exit(
            "[ERROR] ChArUco support needs OpenCV >= 4.7 "
            f"(found {cv2.__version__}). Try: pip install -U opencv-contrib-python"
        )

    dict_id = (
        getattr(cv2.aruco, dict_name, None) if dict_name.startswith("DICT_") else None
    )
    if dict_id is None:
        sys.exit(f"[ERROR] Unknown ArUco dictionary: {dict_name}")
    if marker_size >= square_size:
        sys.exit("[ERROR] marker_size must be smaller than square_size.")

    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard((cols, rows), square_size, marker_size, dictionary)
    # Boards printed before OpenCV 4.6 use a different marker layout. This must
    # be set before the detector is built, or detection/matching will not agree
    # with the physical board.
    board.setLegacyPattern(legacy_pattern)

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters(), detector_params
    )

    views, image_size = [], None
    for path, gray in iter_gray_images(paths):
        image_size = gray.shape[::-1]

        # One call detects the markers AND interpolates the chessboard corners.
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < MIN_CHARUCO_CORNERS:
            found = 0 if ch_ids is None else len(ch_ids)
            print(
                f"  [SKIP] Only {found} ChArUco corners "
                f"(need {MIN_CHARUCO_CORNERS}): {os.path.basename(path)}"
            )
            continue

        # Look up the 3D board position of every corner that was detected.
        obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)
        if obj_pts is None or len(obj_pts) < MIN_CHARUCO_CORNERS:
            print(f"  [SKIP] Point matching failed: {os.path.basename(path)}")
            continue

        views.append(
            View(
                path,
                obj_pts.reshape(-1, 1, 3).astype(np.float32),
                img_pts.reshape(-1, 1, 2).astype(np.float32),
            )
        )

        if show:
            show_preview(
                "ChArUco",
                gray,
                lambda img: cv2.aruco.drawDetectedCornersCharuco(
                    img,
                    np.asarray(ch_corners, dtype=np.float32).reshape(-1, 1, 2).copy(),
                    np.asarray(ch_ids, dtype=np.int32).reshape(-1, 1).copy(),
                ),
            )

    if show:
        cv2.destroyAllWindows()
    return views, image_size


# ───────────────────────────────────────────────────────── calibration ──


def build_pinhole_flags(args, model: str) -> int:
    """Translate CLI options into cv2.calibrateCamera flags."""
    flags = 0
    if args.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if args.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if args.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if model == "rational_poly":
        flags |= cv2.CALIB_RATIONAL_MODEL
    if model == "none":
        # Pin every distortion coefficient to its initial value of zero.
        flags |= (
            cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
            | cv2.CALIB_ZERO_TANGENT_DIST
        )
    return flags


def run_pinhole(views, image_size, model, args):
    """cv2.calibrateCamera for plumb_bob / rational_poly / none."""
    # With CALIB_FIX_ASPECT_RATIO OpenCV only reads the fx/fy *ratio* from the
    # initial matrix, so seed it with fx = fy (ratio of 1).
    camera_matrix = np.eye(3) if args.fix_aspect_ratio else None

    return cv2.calibrateCamera(
        [v.obj_pts for v in views],
        [v.img_pts for v in views],
        image_size,
        camera_matrix,
        None,
        flags=build_pinhole_flags(args, model),
    )


def fisheye_flag(name: str) -> int:
    """Look up a fisheye calibration flag.

    OpenCV 4.x keeps them in cv2.fisheye, OpenCV 5.x in the top-level cv2.
    """
    flag = getattr(cv2.fisheye, name, None)
    return flag if flag is not None else getattr(cv2, name)


def run_fisheye(views, image_size, args):
    """cv2.fisheye.calibrate for the equidistant model."""
    flags = fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC") | fisheye_flag("CALIB_FIX_SKEW")
    if args.fix_principal_point:
        flags |= fisheye_flag("CALIB_FIX_PRINCIPAL_POINT")

    # cv2.fisheye is pickier than calibrateCamera: it wants float64 points.
    obj = [v.obj_pts.reshape(1, -1, 3).astype(np.float64) for v in views]
    img = [v.img_pts.reshape(1, -1, 2).astype(np.float64) for v in views]
    return cv2.fisheye.calibrate(
        obj, img, image_size, np.zeros((3, 3)), np.zeros((4, 1)), flags=flags
    )


def per_view_errors(views, rvecs, tvecs, K, dist, model) -> list[float]:
    """RMS reprojection error (pixels) for each view."""
    project = cv2.fisheye.projectPoints if model == "equidistant" else cv2.projectPoints
    errors = []
    for view, rvec, tvec in zip(views, rvecs, tvecs):
        projected, _ = project(view.obj_pts, rvec, tvec, K, dist)
        diff = projected.reshape(-1, 2) - view.img_pts.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(diff**2))))
    return errors


def calibrate(views, image_size, model, args) -> Calibration:
    """Run the calibration appropriate for `model` and score every view."""
    if model == "equidistant":
        rms, K, dist, rvecs, tvecs = run_fisheye(views, image_size, args)
    else:
        rms, K, dist, rvecs, tvecs = run_pinhole(views, image_size, model, args)

    errors = per_view_errors(views, rvecs, tvecs, K, dist, model)
    return Calibration(float(rms), K, dist, list(rvecs), list(tvecs), views, errors)


# ─────────────────────────────────────────────────────────── YAML output ──


def ros_distortion_model(model: str) -> str:
    """Map our model name to the string ROS expects in camera_info."""
    return {"rational_poly": "rational_polynomial", "equidistant": "equidistant"}.get(
        model, "plumb_bob"
    )


def matrix_entry(matrix: np.ndarray) -> dict:
    """ROS-style {rows, cols, data} dictionary."""
    matrix = np.atleast_2d(matrix)
    return {
        "rows": matrix.shape[0],
        "cols": matrix.shape[1],
        "data": matrix.flatten().tolist(),
    }


def board_poses(cal: Calibration) -> list[dict]:
    """Per-image board poses; `camera_pose_in_board_4x4` is the inverse transform."""
    poses = []
    for i, (view, rvec, tvec) in enumerate(zip(cal.views, cal.rvecs, cal.tvecs)):
        rotation, _ = cv2.Rodrigues(rvec)
        board_to_camera = np.eye(4)
        board_to_camera[:3, :3] = rotation
        board_to_camera[:3, 3] = np.asarray(tvec).flatten()
        poses.append(
            {
                "image": view.name,
                "index": i,
                "rvec": np.asarray(rvec).flatten().tolist(),
                "tvec": np.asarray(tvec).flatten().tolist(),
                "reprojection_error_px": cal.errors[i],
                "camera_pose_in_board_4x4": np.linalg.inv(board_to_camera).tolist(),
            }
        )
    return poses


def make_camera_info(camera_name, image_size, cal, model, save_extrinsics) -> dict:
    """Assemble the camera_info dictionary that gets dumped to YAML."""
    width, height = image_size

    # Monocular camera: identity rectification, projection = [K | 0].
    projection = np.hstack([cal.K, np.zeros((3, 1))])

    info = {
        "image_width": int(width),
        "image_height": int(height),
        "camera_name": camera_name,
        "camera_matrix": matrix_entry(cal.K),
        "distortion_model": ros_distortion_model(model),
        "distortion_coefficients": matrix_entry(cal.dist.reshape(1, -1)),
        "rectification_matrix": matrix_entry(np.eye(3)),
        "projection_matrix": matrix_entry(projection),
        "rms_reprojection_error_px": cal.rms,
    }
    if save_extrinsics:
        info["extrinsics"] = board_poses(cal)
    return info


# ──────────────────────────────────────────────────────────── reporting ──


def print_errors(cal: Calibration, threshold) -> None:
    line = f"  {'-' * 56}"
    print(line)
    print("  Per-image reprojection error (RMS, pixels)")
    print(line)
    for view, err in zip(cal.views, cal.errors):
        if threshold is not None and err > threshold:
            note = f"  <- ABOVE {threshold:.2f} px"
        elif err > HIGH_ERROR_PX:
            note = "  <- HIGH"
        else:
            note = ""
        print(f"  {view.name:40s}  {err:.4f} px{note}")
    print(line)
    print(f"  Overall RMS: {cal.rms:.4f} px")
    print(line)


def print_intrinsics(cal: Calibration) -> None:
    K = cal.K
    print("\nIntrinsics (camera matrix K):")
    print(f"  fx = {K[0, 0]:.4f} px   fy = {K[1, 1]:.4f} px")
    print(f"  cx = {K[0, 2]:.4f} px   cy = {K[1, 2]:.4f} px")
    print(f"  skew = {K[0, 1]:.6f}")
    print(f"  Distortion coefficients: {cal.dist.flatten().tolist()}")


def print_header(args) -> None:
    threshold = (
        f"{args.reproj_threshold:.4f} px"
        if args.reproj_threshold is not None
        else "not set"
    )
    print(f"\n{'=' * 60}")
    print("  Camera Calibration Tool -> ROS 2 camera_info")
    print(f"{'=' * 60}")
    print(f"  Images dir       : {args.images_dir}")
    print(f"  Board type       : {args.board_type}")
    print(f"  Board size       : {args.board_cols} x {args.board_rows}")
    if args.board_type == "charuco":
        print(f"  Legacy pattern   : {args.legacy_pattern}")
    print(f"  Square size      : {args.square_size * 1000:.1f} mm")
    print(f"  Distortion model : {args.distortion_model}")
    print(f"  Reproj threshold : {threshold}")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────── main ──


def remove_outliers(
    cal: Calibration, threshold, image_size, model, args
) -> Calibration:
    """Drop views above `threshold` and recalibrate once.

    Returns the original calibration if nothing was rejected or too few views
    would remain.
    """
    rejected = [v for v, e in zip(cal.views, cal.errors) if e > threshold]
    if not rejected:
        print(f"\nAll images are within {threshold:.4f} px. No recalibration needed.")
        return cal

    kept = [v for v, e in zip(cal.views, cal.errors) if e <= threshold]
    print(f"\n-- Outlier removal (threshold = {threshold:.4f} px) --")
    print(f"  Removing {len(rejected)} image(s):")
    for view, err in ((v, e) for v, e in zip(cal.views, cal.errors) if e > threshold):
        print(f"    {view.name:40s}  {err:.4f} px")

    if len(kept) < MIN_VIEWS:
        print(
            f"  [WARN] Fewer than {MIN_VIEWS} images would remain. "
            "Skipping recalibration, keeping initial results."
        )
        return cal

    print(f"  Recalibrating with {len(kept)} image(s)...")
    new_cal = calibrate(kept, image_size, model, args)
    print(f"\n-- Recalibration result ({len(kept)} / {len(cal.views)} images) --")
    print_errors(new_cal, threshold)
    print_intrinsics(new_cal)
    return new_cal


def main() -> None:
    args = parse_args()
    model = args.distortion_model

    if args.marker_size is None:
        args.marker_size = 0.75 * args.square_size

    # Be upfront about options that the fisheye model cannot honour.
    if model == "equidistant":
        ignored = [o for o in PINHOLE_ONLY_OPTIONS if getattr(args, o)]
        if ignored:
            print(
                f"[WARN] Ignored for the equidistant model: "
                f"{', '.join('--' + o for o in ignored)}"
            )

    print_header(args)
    paths = find_images(args.images_dir)
    print(f"Found {len(paths)} image(s).")

    # 1. Detect the board in every image.
    if args.board_type == "chessboard":
        views, image_size = detect_chessboard(
            paths,
            args.board_rows,
            args.board_cols,
            args.square_size,
            args.subpix_winsize,
            args.show_corners,
        )
    else:
        views, image_size = detect_charuco(
            paths,
            args.board_rows,
            args.board_cols,
            args.square_size,
            args.marker_size,
            args.aruco_dict,
            args.legacy_pattern,
            args.show_corners,
        )

    if len(views) < MIN_VIEWS:
        sys.exit(
            f"[ERROR] Need at least {MIN_VIEWS} images with a detected "
            f"board, got {len(views)}."
        )
    print(f"\nUsed {len(views)} / {len(paths)} images for initial calibration.")

    # 2. Calibrate.
    cal = calibrate(views, image_size, model, args)
    print("\n-- Initial calibration --")
    print_errors(cal, args.reproj_threshold)
    print_intrinsics(cal)

    # 3. Optionally reject bad images and calibrate again (one pass only).
    if args.reproj_threshold is not None:
        cal = remove_outliers(cal, args.reproj_threshold, image_size, model, args)

    # 4. Write the YAML.
    info = make_camera_info(
        args.camera_name, image_size, cal, model, args.save_extrinsics
    )
    with open(args.output, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)

    print(f"\ncamera_info YAML written to: {args.output}")
    print(
        "Point your camera driver's camera_info_url parameter at it, e.g. "
        f"file://{os.path.abspath(args.output)}"
    )

    if args.create_report:
        width, height = image_size
        report_data_path = write_report_data(
            cal=cal,
            args=args,
            image_size=(width, height),
            yaml_output_path=args.output,
            overall_rms=cal.rms,
        )

        run_pdf_report(
            args=args,
            yaml_output_path=args.output,
            report_data_path=report_data_path,
        )


if __name__ == "__main__":
    main()
