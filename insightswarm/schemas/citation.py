from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int

    def validate(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("text_span must have 0 <= start < end")


@dataclass(frozen=True)
class ImageBBox:
    bbox: tuple[float, float, float, float]
    original_width: int
    original_height: int
    format: str = "ymin_xmin_ymax_xmax"

    def validate(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError("image bbox must contain four values")
        ymin, xmin, ymax, xmax = self.bbox
        if any(value < 0.0 or value > 1.0 for value in self.bbox):
            raise ValueError("image bbox values must be normalized between 0 and 1")
        if ymin >= ymax:
            raise ValueError("image bbox requires ymin < ymax")
        if xmin >= xmax:
            raise ValueError("image bbox requires xmin < xmax")
        if self.original_width <= 0 or self.original_height <= 0:
            raise ValueError("image dimensions must be positive")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "bbox": list(self.bbox),
            "format": self.format,
            "original_width": self.original_width,
            "original_height": self.original_height,
        }

