"""Piecewise affine view-to-source time mapping."""
from __future__ import annotations
from typing import Iterable
from .contracts import TimeWarpSegment, _range
from .errors import TimebaseInvariantViolation

class TimeWarp:
    def __init__(self, segments: Iterable[TimeWarpSegment]):
        self.segments = tuple(segments)
        if not self.segments: raise TimebaseInvariantViolation("at least one time-warp segment is required")
        view_id = self.segments[0].view_id
        prev_v = prev_s = None
        for s in self.segments:
            if s.view_id != view_id: raise TimebaseInvariantViolation("segments must share view_id")
            if prev_v is not None and (s.view_start_sample < prev_v or s.source_start_us < prev_s): raise TimebaseInvariantViolation("time-warp segments must be monotonic and non-overlapping")
            prev_v, prev_s = s.view_end_sample, s.source_end_us

    def map_sample(self, sample: int) -> int:
        for s in self.segments:
            if s.view_start_sample <= sample <= s.view_end_sample:
                return s.source_us(sample)
        raise TimebaseInvariantViolation("view sample outside time-warp")

    def map_range(self, start_sample: int, end_sample: int) -> tuple[int, int]:
        if type(start_sample) is not int or type(end_sample) is not int or start_sample >= end_sample: raise TimebaseInvariantViolation("invalid view range")
        start, end = self.map_sample(start_sample), self.map_sample(end_sample)
        _range(start, end)
        return start, end

    def source_range_for_segment(self, index: int) -> tuple[int, int]:
        try: s = self.segments[index]
        except IndexError as e: raise TimebaseInvariantViolation("unknown segment") from e
        return s.source_start_us, s.source_end_us
