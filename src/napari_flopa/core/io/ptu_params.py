from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TagParam:
    """One instrument constant read directly from a PTU header tag."""

    name: str  # key in the constants dict
    tag: str  # PTU header tag to read
    default: Any = None  # value used when the tag is absent
    transform: Callable[[Any], Any] = lambda v: v  # applied to the raw tag


TAG_PARAMS: list[TagParam] = [
    TagParam("repetition_rate", "TTResult_SyncRate", 40e6),
    TagParam("tcspc_resolution", "MeasDesc_Resolution", 1e-9),
    TagParam("wrap", "TTResultFormat_WrapAround", 1024, int),
    TagParam("pixels_x", "ImgHdr_PixX", None, int),
    TagParam("pixels_y", "ImgHdr_PixY", None, int),
    TagParam("frames", "ImgHdr_NumberOfFrames", None, int),
]

PARAMS_BY_NAME: dict[str, TagParam] = {p.name: p for p in TAG_PARAMS}


def read_tag(header_tags: dict, name: str):
    """Return a single param's value from *header_tags* (value only)."""
    p = PARAMS_BY_NAME[name]
    if p.tag in header_tags:
        return p.transform(header_tags[p.tag])
    return p.default
