#!/usr/bin/env python3
"""
calibration_report.py

Generate a PDF quality report from camera calibration diagnostic JSON.

Example:
    python calibration_report.py \
        --report-data camera_info.report_data.json \
        --camera-yaml camera_info.yaml \
        --output camera_report.pdf \
        --coordinate-convention-image coordinate_convention.jpg
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# USER-EDITABLE PASS / FAIL CRITERIA
# ============================================================================
#
# These defaults are reasonable starting points for normal monocular
# calibration, but must be adjusted for lens type, image resolution,
# corner-detection quality, and application requirements.
#
# For high-resolution cameras, a 0.5 px RMS requirement may be appropriate.
# For wide-angle/fisheye lenses or difficult targets, larger error can still
# be acceptable if residuals are unstructured and deployment validation passes.
#

MAX_GLOBAL_RMS_ERROR_PX = 0.50
MAX_PER_IMAGE_RMS_ERROR_PX = 1.00
MAX_P95_REPROJECTION_ERROR_PX = 1.00
MAX_P99_REPROJECTION_ERROR_PX = 2.00

MIN_VALID_IMAGES = 15
MIN_POINTS_PER_IMAGE = 20

# Fraction of image-grid cells that must contain at least one detected point.
MIN_SENSOR_GRID_COVERAGE_FRACTION = 0.50
SENSOR_COVERAGE_GRID_COLS = 12
SENSOR_COVERAGE_GRID_ROWS = 8

# Calibration-board pose diversity recommendations.
MIN_MAX_BOARD_TILT_DEG = 20.0
MIN_DISTANCE_RATIO = 1.5

# Working-distance comparison tolerance.
WORKING_DISTANCE_WARNING_RATIO = 1.25

# Warning threshold for principal-point displacement from image center.
MAX_PRINCIPAL_POINT_OFFSET_PERCENT = 10.0


# ============================================================================
# BASIC HELPERS
# ============================================================================

UNKNOWN = "Unknown"


def prompt_optional(label, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else (default if default not in (None, "") else UNKNOWN)


def prompt_optional_float(label):
    value = input(f"{label} (leave blank for Unknown): ").strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        print("Invalid number; using Unknown.")
        return None


def safe_value(value, precision=4):
    if value is None:
        return UNKNOWN

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return UNKNOWN
        return f"{value:.{precision}f}"

    if value == "":
        return UNKNOWN

    return str(value)


def as_array(value, shape_last=None):
    if value is None:
        return None

    try:
        array = np.asarray(value, dtype=float)
        if shape_last is not None:
            array = array.reshape(-1, shape_last)
        return array
    except Exception:
        return None


def add_title_page(pdf, data, metadata):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Monocular Camera Calibration Report", fontsize=22, y=0.96)

    lines = [
        ("Camera name", metadata["camera_name"]),
        ("Calibration date", metadata["calibration_date"]),
        ("Operator", metadata["operator"]),
        ("Camera serial number", metadata["camera_serial"]),
        ("Lens identifier", metadata["lens_identifier"]),
        ("Lens focus setting", metadata["focus_setting"]),
        ("Lens aperture", metadata["aperture_setting"]),
        ("Image resolution", f"{data['image_width']} x {data['image_height']} px"),
        ("Distortion model", data.get("distortion_model", UNKNOWN)),
        ("Board type", data.get("board", {}).get("board_type", UNKNOWN)),
        ("OpenCV version", data.get("opencv_version", UNKNOWN)),
        ("Input images used", len(data.get("views", []))),
        ("Overall RMS error", f"{safe_value(data.get('overall_rms_error_px'))} px"),
    ]

    y = 0.85
    for label, value in lines:
        fig.text(0.10, y, f"{label}:", fontsize=11, weight="bold")
        fig.text(0.42, y, safe_value(value), fontsize=11)
        y -= 0.045

    fig.text(
        0.10,
        0.12,
        "Coordinate convention: standard pinhole camera model. "
        "Pixel coordinate (0, 0) is at the top-left image corner. "
        "Camera coordinate convention is shown later in this report.",
        fontsize=10,
        wrap=True,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf, title, rows, col_widths=(0.45, 0.55)):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=20)

    table = ax.table(
        cellText=[[str(a), str(b)] for a, b in rows],
        colLabels=["Field", "Value"],
        colWidths=col_widths,
        loc="upper center",
        cellLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def compute_fov(k, width, height):
    if k is None or k.shape != (3, 3):
        return None, None, None

    fx = k[0, 0]
    fy = k[1, 1]

    if fx <= 0 or fy <= 0:
        return None, None, None

    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    diagonal_px = math.sqrt(width**2 + height**2)
    f_diag = math.sqrt(fx * fy)
    dfov = math.degrees(2.0 * math.atan(diagonal_px / (2.0 * f_diag)))

    return hfov, vfov, dfov


def board_pose_metrics(rvec, tvec):
    if rvec is None or tvec is None:
        return None, None, None

    try:
        rvec = np.asarray(rvec, dtype=float).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=float).reshape(3)

        rotation, _ = cv2.Rodrigues(rvec)

        # Board normal in camera coordinates.
        board_normal = rotation[:, 2]
        optical_axis = np.array([0.0, 0.0, 1.0])

        cosine = np.clip(np.dot(board_normal, optical_axis), -1.0, 1.0)
        tilt_deg = math.degrees(math.acos(abs(cosine)))

        distance = float(np.linalg.norm(tvec))

        return distance, tilt_deg, rotation
    except Exception:
        return None, None, None


def sensor_coverage(image_points, width, height):
    occupancy = np.zeros(
        (SENSOR_COVERAGE_GRID_ROWS, SENSOR_COVERAGE_GRID_COLS),
        dtype=int,
    )

    if image_points is None or len(image_points) == 0:
        return occupancy, 0.0

    x = image_points[:, 0]
    y = image_points[:, 1]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    x = x[valid]
    y = y[valid]

    cols = np.clip(
        (x / width * SENSOR_COVERAGE_GRID_COLS).astype(int),
        0,
        SENSOR_COVERAGE_GRID_COLS - 1,
    )
    rows = np.clip(
        (y / height * SENSOR_COVERAGE_GRID_ROWS).astype(int),
        0,
        SENSOR_COVERAGE_GRID_ROWS - 1,
    )

    for row, col in zip(rows, cols):
        occupancy[row, col] += 1

    fraction = np.count_nonzero(occupancy) / occupancy.size
    return occupancy, fraction


def evaluate_quality(data, residual_magnitudes, coverage_fraction, distances, tilts):
    checks = []

    global_rms = data.get("overall_rms_error_px")
    if global_rms is None:
        checks.append(("Overall RMS reprojection error", UNKNOWN, "Unknown"))
    else:
        status = "PASS" if global_rms <= MAX_GLOBAL_RMS_ERROR_PX else "FAIL"
        checks.append(
            (
                "Overall RMS reprojection error",
                f"{global_rms:.3f} px (limit {MAX_GLOBAL_RMS_ERROR_PX:.3f} px)",
                status,
            )
        )

    valid_views = data.get("views", [])
    image_count = len(valid_views)
    status = "PASS" if image_count >= MIN_VALID_IMAGES else "WARNING"
    checks.append(
        (
            "Number of valid calibration images",
            f"{image_count} (recommended >= {MIN_VALID_IMAGES})",
            status,
        )
    )

    if residual_magnitudes.size:
        p95 = float(np.percentile(residual_magnitudes, 95))
        p99 = float(np.percentile(residual_magnitudes, 99))

        checks.append(
            (
                "95th percentile residual",
                f"{p95:.3f} px (limit {MAX_P95_REPROJECTION_ERROR_PX:.3f} px)",
                "PASS" if p95 <= MAX_P95_REPROJECTION_ERROR_PX else "WARNING",
            )
        )
        checks.append(
            (
                "99th percentile residual",
                f"{p99:.3f} px (limit {MAX_P99_REPROJECTION_ERROR_PX:.3f} px)",
                "PASS" if p99 <= MAX_P99_REPROJECTION_ERROR_PX else "WARNING",
            )
        )
    else:
        checks.append(("Residual percentile statistics", UNKNOWN, "Unknown"))

    coverage_status = (
        "PASS" if coverage_fraction >= MIN_SENSOR_GRID_COVERAGE_FRACTION else "WARNING"
    )
    checks.append(
        (
            "Sensor grid coverage",
            (
                f"{coverage_fraction * 100:.1f}% "
                f"(recommended >= {MIN_SENSOR_GRID_COVERAGE_FRACTION * 100:.1f}%)"
            ),
            coverage_status,
        )
    )

    if tilts:
        maximum_tilt = max(tilts)
        checks.append(
            (
                "Maximum board tilt",
                (
                    f"{maximum_tilt:.1f} deg "
                    f"(recommended >= {MIN_MAX_BOARD_TILT_DEG:.1f} deg)"
                ),
                "PASS" if maximum_tilt >= MIN_MAX_BOARD_TILT_DEG else "WARNING",
            )
        )

    if len(distances) >= 2 and min(distances) > 0:
        ratio = max(distances) / min(distances)
        checks.append(
            (
                "Calibration distance range ratio",
                f"{ratio:.2f}x (recommended >= {MIN_DISTANCE_RATIO:.2f}x)",
                "PASS" if ratio >= MIN_DISTANCE_RATIO else "WARNING",
            )
        )

    return checks


# ============================================================================
# REPORT PAGES
# ============================================================================


def add_intrinsics_page(pdf, data, metadata):
    width = data["image_width"]
    height = data["image_height"]
    k = as_array(data.get("camera_matrix"))

    if k is not None and k.shape == (3, 3):
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]
        hfov, vfov, dfov = compute_fov(k, width, height)

        principal_offset_x = 100.0 * (cx - width / 2.0) / width
        principal_offset_y = 100.0 * (cy - height / 2.0) / height
    else:
        fx = fy = cx = cy = hfov = vfov = dfov = None
        principal_offset_x = principal_offset_y = None

    pixel_pitch_um = metadata.get("pixel_pitch_um")
    sensor_width_mm = metadata.get("sensor_width_mm")
    sensor_height_mm = metadata.get("sensor_height_mm")

    if pixel_pitch_um is not None:
        derived_width_mm = width * pixel_pitch_um / 1000.0
        derived_height_mm = height * pixel_pitch_um / 1000.0
    else:
        derived_width_mm = None
        derived_height_mm = None

    effective_sensor_width_mm = sensor_width_mm or derived_width_mm
    effective_sensor_height_mm = sensor_height_mm or derived_height_mm

    focal_mm = None
    if effective_sensor_width_mm and fx:
        focal_mm = fx * effective_sensor_width_mm / width

    dist = as_array(data.get("distortion_coefficients"))
    dist_text = (
        UNKNOWN
        if dist is None
        else np.array2string(
            dist.reshape(-1),
            precision=8,
            separator=", ",
        )
    )

    rows = [
        ("Image size", f"{width} x {height} px"),
        ("Distortion model", safe_value(data.get("distortion_model"))),
        ("fx", safe_value(fx)),
        ("fy", safe_value(fy)),
        ("cx", safe_value(cx)),
        ("cy", safe_value(cy)),
        (
            "Principal-point X offset from center",
            f"{safe_value(principal_offset_x, 2)} %",
        ),
        (
            "Principal-point Y offset from center",
            f"{safe_value(principal_offset_y, 2)} %",
        ),
        ("Horizontal field of view", f"{safe_value(hfov, 2)} deg"),
        ("Vertical field of view", f"{safe_value(vfov, 2)} deg"),
        ("Diagonal field of view", f"{safe_value(dfov, 2)} deg"),
        ("Pixel pitch", f"{safe_value(pixel_pitch_um)} um"),
        ("Sensor width", f"{safe_value(effective_sensor_width_mm)} mm"),
        ("Sensor height", f"{safe_value(effective_sensor_height_mm)} mm"),
        ("Estimated focal length", f"{safe_value(focal_mm)} mm"),
        ("Distortion coefficients", dist_text),
    ]

    add_table_page(pdf, "Intrinsic Parameters and Lens Model", rows)


def add_coordinate_page(pdf, coordinate_image):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title("Coordinate Convention", fontsize=16, pad=20)

    text = (
        "Image coordinate system:\n"
        "• Pixel origin (0, 0): top-left image corner\n"
        "• +x: right\n"
        "• +y: down\n\n"
        "Camera coordinate system:\n"
        "• Standard pinhole convention\n"
        "• +X: right\n"
        "• +Y: down\n"
        "• +Z: forward along the optical axis\n"
    )

    ax.text(
        0.08,
        0.82,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    if coordinate_image and coordinate_image.exists():
        image = plt.imread(coordinate_image)
        ax.imshow(image, extent=(0.08, 0.92, 0.05, 0.58), aspect="auto")
    else:
        ax.text(
            0.08,
            0.35,
            "coordinate_convention.jpg: Unknown / file not supplied",
            transform=ax.transAxes,
            fontsize=11,
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_error_pages(pdf, data):
    view_errors = []
    all_residuals = []
    all_points = []
    all_vectors = []

    for view in data.get("views", []):
        rms = view.get("rms_error_px")
        if rms is not None:
            view_errors.append((view.get("index", 0), rms))

        points = as_array(view.get("image_points"), 2)
        vectors = as_array(view.get("residual_vectors"), 2)

        if points is not None:
            all_points.append(points)

        if vectors is not None:
            all_vectors.append(vectors)
            all_residuals.append(np.linalg.norm(vectors, axis=1))

    residuals = np.concatenate(all_residuals) if all_residuals else np.array([])
    points = np.vstack(all_points) if all_points else None
    vectors = np.vstack(all_vectors) if all_vectors else None

    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.suptitle("Reprojection Error Analysis", fontsize=16)

    if view_errors:
        indices, errors = zip(*view_errors)
        colors = [
            "tab:red" if error > MAX_PER_IMAGE_RMS_ERROR_PX else "tab:blue"
            for error in errors
        ]
        axes[0].bar(indices, errors, color=colors)
        axes[0].axhline(
            MAX_PER_IMAGE_RMS_ERROR_PX,
            color="red",
            linestyle="--",
            label="Per-image warning limit",
        )
        axes[0].set_title("Per-Image RMS Reprojection Error")
        axes[0].set_xlabel("Image index")
        axes[0].set_ylabel("RMS error [px]")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

    if residuals.size:
        axes[1].hist(residuals, bins=40, color="tab:blue", edgecolor="black")
        axes[1].axvline(
            np.percentile(residuals, 95),
            color="orange",
            linestyle="--",
            label=f"95th percentile: {np.percentile(residuals, 95):.3f} px",
        )
        axes[1].set_title("Point-Level Reprojection Residual Distribution")
        axes[1].set_xlabel("Residual magnitude [px]")
        axes[1].set_ylabel("Point count")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------------
    # Spatial reprojection residual diagnostics
    #
    # This produces:
    #   1. Residual magnitude accumulation:
    #      Sum of all residual magnitudes in each sensor bin.
    #
    #   2. Mean reprojection error heatmap:
    #      Mean residual magnitude per populated sensor bin.
    #
    #   3. Residual vector field:
    #      Residual vectors enlarged for report visibility. Arrow colors encode
    #      true residual magnitude, while arrow length is visually amplified.
    # ------------------------------------------------------------------------
    if points is not None and residuals.size:
        width = data["image_width"]
        height = data["image_height"]

        # Number of bins across the sensor. Keep bin aspect approximately
        # consistent with the image aspect ratio.
        grid_cols = 48
        grid_rows = max(1, int(round(grid_cols * height / width)))

        # Controls only visual display of residual arrows.
        #
        # Example:
        # A true residual vector of 0.1 px will be displayed as 4 px when
        # RESIDUAL_VECTOR_VISUAL_SCALE = 40.
        #
        # Increase this if arrows are still too small.
        RESIDUAL_VECTOR_VISUAL_SCALE = 40.0

        # Limit the number of plotted arrows so dense calibration datasets
        # remain readable.
        MAX_VECTOR_ARROWS = 750

        # Build bin edges in image pixel coordinates.
        x_edges = np.linspace(0, width, grid_cols + 1)
        y_edges = np.linspace(0, height, grid_rows + 1)

        # Count observations in every sensor bin.
        point_counts, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=[x_edges, y_edges],
        )

        # Sum of residual magnitudes in every sensor bin.
        residual_sum, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=[x_edges, y_edges],
            weights=residuals,
        )

        # Mean residual magnitude in every populated sensor bin.
        #
        # Empty bins are set to NaN, so they appear as "no calibration data"
        # rather than falsely appearing as zero reprojection error.
        mean_residual = np.divide(
            residual_sum,
            point_counts,
            out=np.full_like(residual_sum, np.nan, dtype=float),
            where=point_counts > 0,
        )

        # histogram2d produces [x_bin, y_bin], while imshow expects
        # [row/y_bin, column/x_bin], so transpose the matrices.
        residual_sum_display = residual_sum.T
        mean_residual_display = mean_residual.T

        # Make missing-data regions clearly visible.
        mean_cmap = plt.colormaps["viridis"].copy()
        mean_cmap.set_bad(color="#d9d9d9")

        accumulation_cmap = plt.colormaps["magma"].copy()
        accumulation_cmap.set_bad(color="#d9d9d9")

        # Use a width that allows three image-aspect plots side by side.
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6.5),
            constrained_layout=True,
        )
        fig.suptitle(
            "Spatial Reprojection Residual Analysis",
            fontsize=16,
        )

        # --------------------------------------------------------------------
        # Plot 1: Total residual magnitude accumulated per sensor region.
        # This highlights regions with both frequent observations and/or large
        # residuals.
        # --------------------------------------------------------------------
        accumulation_image = axes[0].imshow(
            residual_sum_display,
            extent=[0, width, height, 0],
            origin="upper",
            interpolation="nearest",
            cmap=accumulation_cmap,
            aspect="equal",
        )

        axes[0].set_title("Residual Magnitude Spatial Accumulation")
        axes[0].set_xlabel("Image x [px]")
        axes[0].set_ylabel("Image y [px]")
        axes[0].set_xlim(0, width)
        axes[0].set_ylim(height, 0)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].grid(alpha=0.20, linewidth=0.5)

        accumulation_bar = fig.colorbar(
            accumulation_image,
            ax=axes[0],
            shrink=0.85,
        )
        accumulation_bar.set_label("Sum of residual magnitudes [px]")

        # --------------------------------------------------------------------
        # Plot 2: Mean reprojection error per sensor region.
        #
        # Unlike the accumulation plot, this plot is not dominated by areas
        # that merely have many calibration points.
        # --------------------------------------------------------------------
        mean_error_image = axes[1].imshow(
            mean_residual_display,
            extent=[0, width, height, 0],
            origin="upper",
            interpolation="nearest",
            cmap=mean_cmap,
            aspect="equal",
        )

        axes[1].set_title("Mean Reprojection Error Over Sensor")
        axes[1].set_xlabel("Image x [px]")
        axes[1].set_ylabel("Image y [px]")
        axes[1].set_xlim(0, width)
        axes[1].set_ylim(height, 0)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].grid(alpha=0.20, linewidth=0.5)

        mean_error_bar = fig.colorbar(
            mean_error_image,
            ax=axes[1],
            shrink=0.85,
        )
        mean_error_bar.set_label("Mean residual magnitude [px]")

        # --------------------------------------------------------------------
        # Plot 3: Residual vector field.
        #
        # The arrow direction is based on the true residual vector.
        # Arrow color represents the true residual magnitude.
        # Arrow size is enlarged only for visibility.
        # --------------------------------------------------------------------
        if vectors is not None and len(vectors) > 0:
            vector_count = min(MAX_VECTOR_ARROWS, len(points))

            # Uniformly sample through the point list for readability.
            indexes = np.linspace(
                0,
                len(points) - 1,
                vector_count,
            ).astype(int)

            vector_points = points[indexes]
            vector_components = vectors[indexes]
            vector_magnitudes = residuals[indexes]

            quiver = axes[2].quiver(
                vector_points[:, 0],
                vector_points[:, 1],
                # Increase arrow display size without changing the underlying
                # residual magnitude represented by color.
                vector_components[:, 0] * RESIDUAL_VECTOR_VISUAL_SCALE,
                vector_components[:, 1] * RESIDUAL_VECTOR_VISUAL_SCALE,
                vector_magnitudes,
                cmap="plasma",
                # Important: values are interpreted in image pixel coordinates.
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.0035,
                headwidth=4.5,
                headlength=5.5,
                headaxislength=4.5,
                alpha=0.90,
                pivot="tail",
            )

            # Add visible dots at arrow origins. This makes sparse or very
            # short residuals easier to inspect.
            axes[2].scatter(
                vector_points[:, 0],
                vector_points[:, 1],
                s=10,
                c=vector_magnitudes,
                cmap="plasma",
                alpha=0.90,
                edgecolors="black",
                linewidths=0.15,
                zorder=3,
            )

            vector_bar = fig.colorbar(
                quiver,
                ax=axes[2],
                shrink=0.85,
            )
            vector_bar.set_label("True residual magnitude [px]")

            # The quiver key explicitly documents that vectors have been
            # visually enlarged.
            axes[2].quiverkey(
                quiver,
                X=0.84,
                Y=1.05,
                U=1.0 * RESIDUAL_VECTOR_VISUAL_SCALE,
                label=(
                    "1 px true residual\n"
                    f"(shown {RESIDUAL_VECTOR_VISUAL_SCALE:.0f}x larger)"
                ),
                labelpos="E",
                coordinates="axes",
                fontproperties={"size": 8},
            )

        axes[2].set_title("Residual Vector Field")
        axes[2].set_xlabel("Image x [px]")
        axes[2].set_ylabel("Image y [px]")
        axes[2].set_xlim(0, width)
        axes[2].set_ylim(height, 0)
        axes[2].set_aspect("equal", adjustable="box")
        axes[2].grid(alpha=0.20, linewidth=0.5)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return residuals, points


def add_coverage_page(pdf, data, points):
    width = data["image_width"]
    height = data["image_height"]

    occupancy, coverage_fraction = sensor_coverage(points, width, height)

    fig, axes = plt.subplots(1, 2, figsize=(11.69, 5.85))
    fig.suptitle("Calibration Target Sensor Coverage", fontsize=16)

    if points is not None:
        axes[0].scatter(points[:, 0], points[:, 1], s=2, alpha=0.20)
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)
    axes[0].set_aspect("equal")
    axes[0].set_title("All Detected Calibration Points")
    axes[0].set_xlabel("Image x [px]")
    axes[0].set_ylabel("Image y [px]")
    axes[0].grid(alpha=0.25)

    im = axes[1].imshow(occupancy, cmap="viridis", origin="upper")
    axes[1].set_title(f"Grid Occupancy: {coverage_fraction * 100:.1f}% Covered")
    axes[1].set_xlabel("Grid x")
    axes[1].set_ylabel("Grid y")
    fig.colorbar(im, ax=axes[1], label="Detected points")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return coverage_fraction


def add_pose_page(pdf, data):
    distances = []
    tilts = []
    translations = []
    errors = []

    for view in data.get("views", []):
        distance, tilt, _ = board_pose_metrics(
            view.get("rvec"),
            view.get("tvec"),
        )

        if distance is not None:
            distances.append(distance)
            translations.append(np.asarray(view["tvec"], dtype=float).reshape(3))

        if tilt is not None:
            tilts.append(tilt)

        errors.append(view.get("rms_error_px"))

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("Calibration Target Pose and Distance Distribution", fontsize=16)

    ax3d = fig.add_subplot(221, projection="3d")
    if translations:
        xyz = np.vstack(translations)
        sc = ax3d.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=distances,
            cmap="viridis",
            s=30,
        )
        fig.colorbar(sc, ax=ax3d, shrink=0.7, label="Distance")
    ax3d.scatter([0], [0], [0], c="red", marker="^", s=100, label="Camera")
    ax3d.set_title("Target Origins in Camera Coordinates")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend()

    ax_distance = fig.add_subplot(222)
    if distances:
        ax_distance.hist(
            distances, bins=min(15, max(5, len(distances))), color="tab:blue"
        )
    ax_distance.set_title("Camera-to-Target Distance")
    ax_distance.set_xlabel("Distance [board units]")
    ax_distance.set_ylabel("Image count")
    ax_distance.grid(alpha=0.25)

    ax_tilt = fig.add_subplot(223)
    if tilts:
        ax_tilt.hist(tilts, bins=min(15, max(5, len(tilts))), color="tab:green")
    ax_tilt.set_title("Board Tilt Distribution")
    ax_tilt.set_xlabel("Tilt [deg]")
    ax_tilt.set_ylabel("Image count")
    ax_tilt.grid(alpha=0.25)

    ax_error = fig.add_subplot(224)
    if distances and errors:
        usable = [(d, e) for d, e in zip(distances, errors) if e is not None]
        if usable:
            x, y = zip(*usable)
            ax_error.scatter(x, y, color="tab:red")
    ax_error.set_title("Reprojection Error vs. Distance")
    ax_error.set_xlabel("Distance [board units]")
    ax_error.set_ylabel("RMS error [px]")
    ax_error.grid(alpha=0.25)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return distances, tilts


def add_per_image_page(pdf, data):
    rows = []

    for view in data.get("views", []):
        distance, tilt, _ = board_pose_metrics(
            view.get("rvec"),
            view.get("tvec"),
        )

        rms = view.get("rms_error_px")
        status = (
            "WARNING" if rms is not None and rms > MAX_PER_IMAGE_RMS_ERROR_PX else "OK"
        )

        rows.append(
            [
                view.get("index", UNKNOWN),
                Path(str(view.get("filename", UNKNOWN))).name,
                view.get("detected_point_count", UNKNOWN),
                safe_value(rms, 3),
                safe_value(distance, 4),
                safe_value(tilt, 1),
                status,
            ]
        )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title("Per-Image Calibration Diagnostics", fontsize=16, pad=20)

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Index",
            "Image",
            "Points",
            "RMS [px]",
            "Distance",
            "Tilt [deg]",
            "Status",
        ],
        loc="upper center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_quality_page(pdf, checks, metadata, distances):
    rows = [[name, result, status] for name, result, status in checks]

    estimated_working_distance = metadata.get("working_distance")
    if estimated_working_distance is not None and distances:
        nearest = min(distances)
        farthest = max(distances)

        if (
            estimated_working_distance < nearest / WORKING_DISTANCE_WARNING_RATIO
            or estimated_working_distance > farthest * WORKING_DISTANCE_WARNING_RATIO
        ):
            deployment_status = "WARNING"
        else:
            deployment_status = "PASS"

        rows.append(
            [
                "Expected working distance coverage",
                (
                    f"Expected: {estimated_working_distance}; "
                    f"calibrated range: {nearest:.4f} to {farthest:.4f}"
                ),
                deployment_status,
            ]
        )
    else:
        rows.append(
            [
                "Expected working distance coverage",
                UNKNOWN,
                "Unknown",
            ]
        )

    has_fail = any(row[2] == "FAIL" for row in rows)
    has_warning = any(row[2] == "WARNING" for row in rows)

    overall_status = (
        "FAIL" if has_fail else ("PASS WITH WARNINGS" if has_warning else "PASS")
    )

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(
        f"Calibration Quality Assessment: {overall_status}", fontsize=16, pad=20
    )

    table = ax.table(
        cellText=rows,
        colLabels=["Check", "Measured value", "Status"],
        loc="upper center",
        cellLoc="left",
        colWidths=[0.34, 0.48, 0.18],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        elif col == 2:
            value = cell.get_text().get_text()
            if value == "PASS":
                cell.set_facecolor("#c8e6c9")
            elif value == "WARNING":
                cell.set_facecolor("#fff3cd")
            elif value == "FAIL":
                cell.set_facecolor("#f8d7da")

    fig.text(
        0.08,
        0.08,
        "Important: reprojection error alone is not sufficient to establish "
        "deployment accuracy. Review spatial residuals, coverage, pose "
        "diversity, and validate with application-specific measurements.",
        fontsize=9,
        wrap=True,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate a monocular camera calibration PDF report."
    )
    parser.add_argument("--report-data", required=True)
    parser.add_argument("--camera-yaml", required=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coordinate-convention-image", default=None)
    args = parser.parse_args()

    with open(args.report_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\nCalibration report metadata")
    print("Leave optional values blank to mark them as Unknown.\n")

    metadata = {
        "camera_name": prompt_optional(
            "Camera name",
            data.get("camera_name"),
        ),
        "calibration_date": prompt_optional(
            "Calibration date",
            datetime.now().strftime("%Y-%m-%d"),
        ),
        "operator": prompt_optional("Operator"),
        "camera_serial": prompt_optional("Camera serial number"),
        "lens_identifier": prompt_optional("Lens identifier / serial number"),
        "focus_setting": prompt_optional("Focus setting"),
        "aperture_setting": prompt_optional("Aperture / f-number"),
        "working_distance": prompt_optional_float(
            "Estimated deployment working distance "
            "(must use the same units as board square size)"
        ),
        "pixel_pitch_um": prompt_optional_float("Square pixel pitch in micrometers"),
        "sensor_width_mm": prompt_optional_float("Sensor width in mm"),
        "sensor_height_mm": prompt_optional_float("Sensor height in mm"),
    }

    coordinate_image = (
        Path(args.coordinate_convention_image)
        if args.coordinate_convention_image
        else None
    )

    residuals, points = None, None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        add_title_page(pdf, data, metadata)
        add_coordinate_page(pdf, coordinate_image)
        add_intrinsics_page(pdf, data, metadata)

        residuals, points = add_error_pages(pdf, data)
        coverage_fraction = add_coverage_page(pdf, data, points)
        distances, tilts = add_pose_page(pdf, data)
        add_per_image_page(pdf, data)

        checks = evaluate_quality(
            data=data,
            residual_magnitudes=residuals,
            coverage_fraction=coverage_fraction,
            distances=distances,
            tilts=tilts,
        )

        add_quality_page(
            pdf=pdf,
            checks=checks,
            metadata=metadata,
            distances=distances,
        )

    print(f"\nCalibration PDF report generated: {output_path}")


if __name__ == "__main__":
    main()
