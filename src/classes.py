"""Unified class index for the combined traffic sign + traffic light detector.

The project merges two datasets with different, unrelated label sets into one
YOLO training set. That requires a single index space, which this module defines:
indices 0-19 are traffic signs, 20-32 are traffic light states. YOLO label files
refer to the integer index; predictions are mapped back to the original dataset
label through ``CLASS_NAMES``.

Indices are frozen: they are baked into every generated ``.txt`` label file and
into the trained model's output layer, so they must not be reordered.
"""

from __future__ import annotations

#: Dataset label -> YOLO class index. Order is significant; see module docstring.
CLASSES: dict[str, int] = {
    # --- traffic signs (0-19) ---
    "100_SIGN": 0,
    "110_SIGN": 1,
    "120_SIGN": 2,
    "30_SIGN": 3,
    "50_SIGN": 4,
    "60_SIGN": 5,
    "70_SIGN": 6,
    "80_SIGN": 7,
    "90_SIGN": 8,
    "GIVE_WAY": 9,
    "NO_PARKING": 10,
    "NO_STOPPING_NO_STANDING": 11,
    "OTHER": 12,
    "PASS_EITHER_SIDE": 13,
    "PASS_LEFT_SIDE": 14,
    "PASS_RIGHT_SIDE": 15,
    "PEDESTRIAN_CROSSING": 16,
    "PRIORITY_ROAD": 17,
    "STOP": 18,
    "URDBL": 19,
    # --- traffic light states (20-32) ---
    "Yellow": 20,
    "RedLeft": 21,
    "Red": 22,
    "GreenLeft": 23,
    "Green": 24,
    "off": 25,
    "GreenRight": 26,
    "GreenStraight": 27,
    "GreenStraightRight": 28,
    "RedRight": 29,
    "RedStraight": 30,
    "RedStraightLeft": 31,
    "GreenStraightLeft": 32,
}

#: YOLO class index as a string -> dataset label, for rendering predictions.
#: Derived from ``CLASSES`` so the two can never drift apart.
CLASS_NAMES: dict[str, str] = {str(v): k for k, v in CLASSES.items()}

#: Number of classes the detector predicts.
NUM_CLASSES: int = len(CLASSES)

#: Indices 0-19 are traffic signs, 20+ are traffic light states. Useful for
#: per-group evaluation and for reading results at a glance.
SIGN_CLASSES: tuple[int, ...] = tuple(range(20))
LIGHT_CLASSES: tuple[int, ...] = tuple(range(20, NUM_CLASSES))


def name_for(index: int | str) -> str:
    """Return the dataset label for a YOLO class index.

    Accepts an ``int`` or its string form, because Roboflow returns the class
    index as a string in its JSON responses.
    """
    return CLASS_NAMES.get(str(index), f"unknown({index})")


def index_for(label: str) -> int:
    """Return the YOLO class index for a dataset label.

    Raises:
        KeyError: if the label is not part of the unified class set. Callers
            converting dataset annotations rely on this to surface unmapped
            labels rather than silently emitting a wrong class.
    """
    return CLASSES[label]
