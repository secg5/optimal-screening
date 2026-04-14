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


class LandmineRawDataset(BaseDataset):
    """Landmines raw dataset with humanized LLM prompts.

    Designed for `data/landmines_raw.csv`, which contains continuous features
    (not decile-binned) plus one-hot encoded land_use, weather, and relief columns.

    Produces prompts that explain the Colombia landmine context and present
    features with units and plain-language descriptions so the LLM can reason
    meaningfully about accident risk.
    """

    # Features with units and plain-language labels
    CONTINUOUS_FEATURES: dict[str, tuple[str, str]] = {
        "elevation": ("Elevation", "m above sea level"),
        "rainfall": ("Average annual rainfall", "mm"),
        "temperature": ("Average temperature", "°C"),
        "population_2012": ("Municipal population (2012)", "people"),
        "No. Víctimas por Declaración": ("Declared conflict victims in municipality", "people"),
        "airports_dist": ("Distance to nearest airport", "km"),
        "seaport_dist": ("Distance to nearest seaport", "km"),
        "settlement_dist": ("Distance to nearest settlement", "km"),
        "coca_dist": ("Distance to nearest coca cultivation area", "km"),
        "dist_old_mine": ("Distance to nearest historical landmine accident", "km"),
        "nighttime_lights_2012": ("Nighttime light intensity (2012)", "index 0–63"),
        "hist_mines": ("Number of historical landmine accidents in municipality", "incidents"),
        "indrural": ("Rural index of municipality", "0=urban, 1=fully rural"),
        "discapital": ("Distance to departmental capital", "km"),
        "pib_percapita_cons": ("GDP per capita", "millions COP"),
        "areaoficialkm2": ("Municipal area", "km²"),
        "waterways_dist": ("Distance to nearest waterway", "km"),
        "dist_roads_t1": ("Distance to primary road", "km"),
        "dist_roads_t2": ("Distance to secondary road", "km"),
        "dist_roads_t3": ("Distance to tertiary road", "km"),
    }

    def __init__(
        self,
        csv_path: str = "data/landmines_raw.csv",
        variables: list[str] | None = None,
        outcome_col: str = "mines_outcome",
    ) -> None:
        super().__init__(csv_path=csv_path, variables=variables, outcome_col=outcome_col)

    def _decode_onehot(self, row: dict[str, Any], prefix: str) -> str | None:
        """Return the active category for a one-hot group, or None if all zero."""
        for key, val in row.items():
            if key.startswith(prefix + "_"):
                try:
                    if float(val) == 1.0:
                        return key[len(prefix) + 1:]
                except (ValueError, TypeError):
                    pass
        return None

    def encode_variable(self, name: str, value: Any) -> str:
        if name == "Municipio":
            return f"Municipality: {value}" if value else "Municipality: (unknown)"
        if name in self.CONTINUOUS_FEATURES:
            label, unit = self.CONTINUOUS_FEATURES[name]
            try:
                v = float(value)
                return f"{label}: {v:.2f} {unit}"
            except (ValueError, TypeError):
                return f"{label}: (missing)"
        return f"{name}: {value}"

    def get_prompt(self, row: dict[str, Any], variables: list[str] | None = None) -> str:
        import json

        land_use = self._decode_onehot(row, "land_use") or "unknown"
        weather = self._decode_onehot(row, "weather") or "unknown"
        relief = self._decode_onehot(row, "relief") or "unknown"

        priority_keys = [
            "Municipio",
            "hist_mines",
            "dist_old_mine",
            "coca_dist",
            "No. Víctimas por Declaración",
            "elevation",
            "nighttime_lights_2012",
            "indrural",
            "discapital",
            "dist_roads_t1",
            "waterways_dist",
        ]

        lines: list[str] = [
            "Task: Estimate the probability that a spatial unit in Colombia experienced a landmine accident.",
            "",
            "Context: Landmine accidents in Colombia are concentrated in rural conflict zones, "
            "particularly near areas with historical mine placements, coca cultivation, and guerrilla activity. "
            "Higher proximity to past accidents and coca fields strongly predicts risk.",
            "",
        ]
        for key in priority_keys:
            if key in row:
                lines.append(self.encode_variable(key, row[key]))
        lines.append(f"Land use: {land_use}")
        lines.append(f"Climate zone: {weather}")
        lines.append(f"Relief type: {relief}")

        features = "\n".join(lines)

        probability_header = (
            "You are an expert in conflict geography and landmine risk in Colombia. "
            "Given the features below, estimate the probability (0 to 1) that this spatial unit "
            "experienced a landmine accident. "
            "Return only a JSON object with one key: 'probability' (float 0-1). "
            "Do not include explanations or other fields."
        )
        probability_prompt = probability_header + "\n\n" + features

        bucket_header = (
            "You are an expert in conflict geography and landmine risk in Colombia. "
            "Given the features below, classify this spatial unit as low risk (1) or high risk (2) "
            "of having experienced a landmine accident. "
            "Return only a JSON object with one key: 'bucket' (integer 1 or 2). "
            "Do not include explanations or other fields."
        )
        bucket_prompt = bucket_header + "\n\n" + features

        naive_header = (
            "You are an expert in conflict geography and landmine risk in Colombia. "
            "Given the features below, provide a screening score (0 to 1) indicating how valuable "
            "it would be to inspect this spatial unit for landmine accidents. "
            "Screening is most valuable for units with uncertain, moderate risk. "
            "Return only a JSON object with one key: 'naive' (float 0-1). "
            "Do not include explanations or other fields."
        )
        naive_prompt = naive_header + "\n\n" + features

        return json.dumps(
            {"bucket_prompt": bucket_prompt, "probability_prompt": probability_prompt, "naive_prompt": naive_prompt}
        )


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
