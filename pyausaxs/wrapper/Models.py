from __future__ import annotations
from enum import Enum as enum

class ExvModel(enum):
    simple = "simple"; average = "average"; fraser = "fraser"; grid_base = "grid-base"; grid_scalable = "grid-scalable",
    grid = "grid"; crysol = "crysol"; foxs = "foxs"; pepsi = "pepsi"; none = "none"; waxsis = "waxsis"

    def validate(model: ExvModel | str) -> ExvModel:
        if not isinstance(model, ExvModel):
            try:
                model = ExvModel(model)
            except ValueError:
                raise ValueError(f"Invalid ExvModel: {model}. Valid models are: {[m.value for m in ExvModel]}")
        return model

class WaterModel(enum):
    radial = "radial"; axes = "axes"; none = "none"

    def validate(model: WaterModel | str) -> WaterModel:
        if not isinstance(model, WaterModel):
            try:
                model = WaterModel(model)
            except ValueError:
                raise ValueError(f"Invalid WaterModel: {model}. Valid models are: {[m.value for m in WaterModel]}")
        return model