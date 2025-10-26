from __future__ import annotations
from enum import Enum as enum

class ExvModel(enum):
    simple = "simple"
    average = "average"
    fraser = "fraser"
    grid_base = "grid-base"
    grid_scalable = "grid-scalable"
    grid = "grid"
    crysol = "crysol"
    foxs = "foxs"
    pepsi = "pepsi"
    waxsis = "waxsis"
    none = "none"

    @staticmethod
    def validate(model: ExvModel | str) -> ExvModel:
        if not isinstance(model, ExvModel):
            try:
                model = ExvModel(model)
            except ValueError:
                raise ValueError(f"Invalid ExvModel: {model}. Valid models are: {[m.value for m in ExvModel]}")
        return model

class ExvTable(enum):
    traube = "traube",
    voronoi_implicit_H = "voronoi_implicit_h"
    voronoi_explicit_H = "voronoi_explicit_h"
    minimum_fluctutation_implicit_H = "minimum_fluctuation_implicit_h"
    minimum_fluctutation_explicit_H = "minimum_fluctuation_explicit_h"
    vdw = "vdw"

    @staticmethod
    def alternate_names(self) -> list[str]:
        return ["voronoi", "mf"]

    @staticmethod
    def validate(table: ExvTable | str) -> ExvTable:
        if not isinstance(table, ExvTable):
            match table.lower(): # match alternate names
                case "voronoi": table = ExvTable.voronoi_implicit_H
                case "mf": table = ExvTable.minimum_fluctutation_implicit_H
            try:
                table = ExvTable(table)
            except ValueError:
                raise ValueError(f"Invalid ExvTable: {table}. Valid tables are: {[t.value for t in ExvTable]}")
        return table

class WaterModel(enum):
    radial = "radial" 
    axes = "axes" 
    none = "none"

    @staticmethod
    def validate(model: WaterModel | str) -> WaterModel:
        if not isinstance(model, WaterModel):
            try:
                model = WaterModel(model)
            except ValueError:
                raise ValueError(f"Invalid WaterModel: {model}. Valid models are: {[m.value for m in WaterModel]}")
        return model