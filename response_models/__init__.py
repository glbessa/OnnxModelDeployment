from pydantic import BaseModel

type Segment = list[tuple[int, int]]

class InstanceSegmentationResponse(BaseModel):
    segments: list[Segment]
    segments_class: list[int]