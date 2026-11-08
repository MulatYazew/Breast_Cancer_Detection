"""Auxiliary shape measurements from a lesion mask.

These numbers describe a binary mask. They are not BI-RADS descriptors.
Orientation in the ultrasound lexicon is the long axis relative to the skin
(Lee 2017, Ultrasonography 36:71-81, printed p. 76). A mask has no skin line,
and a round lesion has no long axis, so this module never emits parallel or
not-parallel.
"""

from __future__ import annotations

import cv2
import numpy as np

AUXILIARY_NOTE = (
    "Auxiliary mask geometry only. Not a BI-RADS shape, margin, orientation, "
    "or echo-pattern decision. A round lesion is not assigned an orientation."
)


def geometry_from_mask(mask: np.ndarray, round_axis_ratio: float = 0.9) -> dict:
    """Measure the largest foreground component. Empty masks return null measures."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary.max(axis=2)
    if binary.sum() == 0:
        return {
            "empty_mask": True,
            "area_px": 0,
            "perimeter_px": None,
            "major_axis_px": None,
            "minor_axis_px": None,
            "aspect_ratio": None,
            "eccentricity": None,
            "solidity": None,
            "extent": None,
            "circularity": None,
            "orientation_assignment": None,
            "note": AUXILIARY_NOTE + " The mask is empty, so no measurement was made.",
        }

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        component = binary
    else:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = (labels == largest).astype(np.uint8)

    area = float(component.sum())
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        perimeter = 0.0
        solidity = None
        major = minor = None
    else:
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area / hull_area) if hull_area > 0 else None
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            axes = sorted((float(ellipse[1][0]), float(ellipse[1][1])))
            minor, major = axes[0], axes[1]
        else:
            _x, _y, w, h = cv2.boundingRect(contour)
            minor, major = float(min(w, h)), float(max(w, h))

    ys, xs = np.where(component > 0)
    extent_box = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    extent = float(area / extent_box) if extent_box else None
    circularity = float(4 * np.pi * area / (perimeter * perimeter)) if perimeter else None
    aspect = float(minor / major) if major else None
    if major and minor is not None and major > 0:
        ratio = minor / major
        eccentricity = float(np.sqrt(max(0.0, 1.0 - ratio * ratio)))
    else:
        eccentricity = None

    orientation_assignment = None
    orientation_reason = "No skin line is annotated, so parallel/not-parallel is not computed."
    if aspect is not None and aspect >= round_axis_ratio:
        orientation_reason = (
            "The component is nearly round, so it has no longest axis. "
            "Lee 2017 states that a round mass is neither parallel nor not parallel. "
            + orientation_reason
        )

    return {
        "empty_mask": False,
        "area_px": area,
        "perimeter_px": perimeter,
        "major_axis_px": major,
        "minor_axis_px": minor,
        "aspect_ratio": aspect,
        "eccentricity": eccentricity,
        "solidity": solidity,
        "extent": extent,
        "circularity": circularity,
        "orientation_assignment": orientation_assignment,
        "orientation_reason": orientation_reason,
        "units": "pixels",
        "spacing_available": False,
        "note": AUXILIARY_NOTE,
    }
