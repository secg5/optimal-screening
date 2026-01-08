from __future__ import annotations

from typing import Any

from .base import BaseDataset


class LandmineDataset(BaseDataset):
    """Landmines dataset encoder.

    - CSV expected at `data/landmines_processed.csv` unless overridden.
    - Outcome column: `mines_outcome` (binary 0/1 or False/True)
    - Location column: `Municipio` (string)
    - Decile features (encoded as decile ranks 1–10 or 0–9 depending on data):
      ['elevation', 'rainfall', 'temperature', 'population_2012',
       'No. Víctimas por Declaración', 'airports_dist', 'seaport_dist',
       'settlement_dist', 'finance_dist', 'edu_dist', 'buildings_dist',
       'waterways_dist', 'coca_dist', 'dist_roads_t1', 'dist_roads_t2',
       'dist_roads_t3', 'dist_old_mine', 'soil_texture15_trans1',
       'soil_texture15_trans2', 'nighttime_lights_2012', 'hist_mines',
       'retro_pobl_tot', 'indrural', 'areaoficialkm2', 'altura',
       'discapital', 'pib_percapita_cons']
    - All other features are treated as binary indicators when they appear
      to be 0/1, otherwise encoded generically.

    Usage example:
        ds = LandmineDataset()
        rows = ds.preprocess_data(ds.load_data())
        prompt = ds.get_prompt(rows[0])
    """

    # Columns that represent decile bins in the processed dataset
    DECILE_FEATURES: set[str] = {
        "elevation",
        "rainfall",
        "temperature",
        "population_2012",
        "No. Víctimas por Declaración",
        "airports_dist",
        "seaport_dist",
        "settlement_dist",
        "finance_dist",
        "edu_dist",
        "buildings_dist",
        "waterways_dist",
        "coca_dist",
        "dist_roads_t1",
        "dist_roads_t2",
        "dist_roads_t3",
        "dist_old_mine",
        "soil_texture15_trans1",
        "soil_texture15_trans2",
        "nighttime_lights_2012",
        "hist_mines",
        "retro_pobl_tot",
        "indrural",
        "areaoficialkm2",
        "altura",
        "discapital",
        "pib_percapita_cons",
    }

    def __init__(
        self,
        csv_path: str = "data/landmines_processed.csv",
        variables: list[str] | None = None,
        outcome_col: str = "mines_outcome",
    ) -> None:
        super().__init__(csv_path=csv_path, variables=variables, outcome_col=outcome_col)

    def _to_int(self, v: Any) -> int | None:
        try:
            if v is None or v == "":
                return None
            # Some CSVs may store as float-like strings (e.g., "3.0")
            return int(float(v))
        except Exception:
            return None

    def _as_yes_no(self, v: Any) -> str:
        # Interpret common binary representations
        if isinstance(v, (bool,)):
            return "yes" if v else "no"
        iv = self._to_int(v)
        if iv is None:
            return "(missing)"
        if iv == 1:
            return "yes"
        if iv == 0:
            return "no"
        # Fall back to raw value if it doesn't look binary
        return str(v)

    def encode_variable(self, name: str, value: Any) -> str:
        # Outcome is handled separately by BaseDataset.get_prompt; we keep a readable encoding if needed
        if name == self.outcome_col:
            return f"{name}: {self._as_yes_no(value)}"

        # Municipality/location
        if name == "Municipio":
            if value is None or value == "":
                return "municipality: (unknown)"
            return f"municipality: {value}"

        # Decile features -> present as decile rank (out of 10 when possible)
        if name in self.DECILE_FEATURES:
            iv = self._to_int(value)
            if iv is None:
                return f"{name} (decile): (missing)"

            # Heuristic: If values look 0–9, map to 1–10 for readability, otherwise leave as-is.
            pretty = iv + 1 if 0 <= iv <= 9 else iv
            return f"{name} (decile): {pretty}/10"

        # Try to render booleans/binaries as yes/no
        yn = self._as_yes_no(value)
        if yn in {"yes", "no", "(missing)"}:
            return f"{name}: {yn}"

        # Fallback generic printing
        if value is None or value == "":
            return f"{name}: (missing)"
        return f"{name}: {value}"


class GranadaDataset(LandmineDataset):
    """Granada subset of the landmines dataset.

    Identical encoding to LandmineDataset, but defaults to
    `data/landmines_granada.csv` as the source CSV.
    """

    def __init__(
        self,
        csv_path: str = "data/landmines_granada.csv",
        variables: list[str] | None = None,
        outcome_col: str = "mines_outcome",
    ) -> None:
        super().__init__(csv_path=csv_path, variables=variables, outcome_col=outcome_col)
