from __future__ import annotations

from typing import Any

from .base import BaseDataset


class ACSDataset(BaseDataset):
    """American Community Survey (ACS) income-focused dataset encoder.

    Encodes common ACS columns into human-readable text suitable for LLM prompts.
    Mappings are adapted from the ACS codebook and the folktexts project.
    """

    _SEX = {1: "Male", 2: "Female"}
    _MAR = {
        1: "Married",
        2: "Widowed",
        3: "Divorced",
        4: "Separated",
        5: "Never married",
    }
    _RAC1P = {
        1: "White",
        2: "Black or African American",
        3: "American Indian",
        4: "Alaska Native",
        5: (
            "American Indian and Alaska Native tribes specified; or American "
            "Indian or Alaska Native, not specified and no other races"
        ),
        6: "Asian",
        7: "Native Hawaiian and Other Pacific Islander",
        8: "Some other race alone (non-White)",
        9: "Two or more races",
    }
    _SCHL = {
        1: "No schooling completed",
        2: "Nursery school / preschool",
        3: "Kindergarten",
        4: "1st grade",
        5: "2nd grade",
        6: "3rd grade",
        7: "4th grade",
        8: "5th grade",
        9: "6th grade",
        10: "7th grade",
        11: "8th grade",
        12: "9th grade",
        13: "10th grade",
        14: "11th grade",
        15: "12th grade, no diploma",
        16: "Regular high school diploma",
        17: "GED or alternative credential",
        18: "Some college, <1 year",
        19: "+1 year college, no degree",
        20: "Associate's degree",
        21: "Bachelor's degree",
        22: "Master's degree",
        23: "Professional degree",
        24: "Doctorate degree",
    }
    _ESR = {
        1: "Civilian employed, at work",
        2: "Civilian employed, with a job but not at work",
        3: "Unemployed",
        4: "Armed forces, at work",
        5: "Armed forces, with a job but not at work",
        6: "Not in labor force",
    }
    _COW = {
        1: "For-profit private company",
        2: "Non-profit organization",
        3: "Local government",
        4: "State government",
        5: "Federal government",
        6: "Self-employed (not incorporated)",
        7: "Self-employed (incorporated)",
        8: "Unpaid family worker",
        9: "Unemployed/never worked or >5 years",
    }

    def _fmt_int(self, v: Any) -> int | None:
        try:
            if v is None or v == "":
                return None
            return int(float(v))
        except Exception:
            return None

    def encode_variable(self, name: str, value: Any) -> str:
        key = name.strip().upper()

        # Age
        if key == "AGEP":
            iv = self._fmt_int(value)
            return f"age: {iv} years old" if iv is not None else "age: (missing)"

        # Sex
        if key == "SEX":
            iv = self._fmt_int(value)
            return f"sex: {self._SEX.get(iv, 'Unknown')}"

        # Race
        if key == "RAC1P":
            iv = self._fmt_int(value)
            return f"race: {self._RAC1P.get(iv, 'Unknown')}"

        # Marital status
        if key == "MAR":
            iv = self._fmt_int(value)
            return f"marital status: {self._MAR.get(iv, 'Unknown')}"

        # Educational attainment
        if key == "SCHL":
            iv = self._fmt_int(value)
            return f"education: {self._SCHL.get(iv, 'Unknown')}"

        # Employment status
        if key == "ESR":
            iv = self._fmt_int(value)
            suffix = " (N/A <16yo)" if value in (None, "") else ""
            return f"employment: {self._ESR.get(iv, 'Unknown')}{suffix}"

        # Class of worker
        if key == "COW":
            iv = self._fmt_int(value)
            return f"class of worker: {self._COW.get(iv, 'Unknown')}"

        # Usual hours per week
        if key == "WKHP":
            iv = self._fmt_int(value)
            if iv is None:
                return "hours worked per week: N/A"
            return f"hours worked per week: {iv} hours"

        # Yearly income
        if key == "PINCP":
            iv = self._fmt_int(value)
            return f"yearly income: ${iv:,}" if iv is not None else "yearly income: N/A"

        # Fallback generic formatting
        if value is None or value == "":
            return f"{name}: (missing)"
        return f"{name}: {value}"
