"""
Synthetic object-relation prompt metadata for objectRel pilots.

``ShapesDataset`` mirrors the interface expected by the split-workflow notebooks:
only ``spatial_phrases`` and ``num_images`` are used in practice.
"""

from __future__ import annotations

from typing import Any

# Relation key -> natural-language fragments for "{color1} {shape1} is {rel_text} {color2} {shape2}".
# Skips ``in_front`` / ``behind`` in combinatorial prompt generators (see notebooks).
DEFAULT_SPATIAL_PHRASES: dict[str, list[str]] = {
    "above": ["above", "above and to the left of", "above and to the right of"],
    "below": ["below", "below and to the left of", "below and to the right of"],
    "left": ["to the left of", "to the upper left of", "to the lower left of"],
    "right": ["to the right of", "to the upper right of", "to the lower right of"],
    "upper_left": ["to the upper left of"],
    "upper_right": ["to the upper right of", "above and to the right of"],
    "lower_left": ["to the lower left of", "below and to the left of"],
    "lower_right": ["to the lower right of", "below and to the right of"],
    "in_front": ["in front of"],
    "behind": ["behind"],
}


class ShapesDataset:
    """Lightweight stand-in; training images are not loaded in these notebooks."""

    def __init__(
        self,
        num_images: int = 10_000,
        spatial_phrases: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ):
        self.num_images = int(num_images)
        self.spatial_phrases = dict(spatial_phrases or DEFAULT_SPATIAL_PHRASES)
        self.kwargs = kwargs

    def __len__(self) -> int:
        return self.num_images
