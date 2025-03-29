from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from typing_extensions import Self

from opera_utils.constants import DISP_FILE_REGEX

__all__ = ["DispProduct"]


@dataclass
class DispProduct:
    """Class for information from one DISP-S1 production filename."""

    sensor: str
    acquisition_mode: str
    frame_id: int
    polarization: str
    reference_datetime: datetime
    secondary_datetime: datetime
    version: str
    generation_datetime: datetime

    @classmethod
    def from_filename(cls, name: Path | str) -> Self:
        """Parse a filename to create a DispProduct.

        Parameters
        ----------
        name : str or Path
            Filename to parse for OPERA DISP-S1 information.

        Returns
        -------
        DispProduct
            Parsed file information.

        Raises
        ------
        ValueError
            If the filename format is invalid.
        """
        if not (match := DISP_FILE_REGEX.match(Path(name).name)):
            raise ValueError(f"Invalid filename format: {name}")

        data = match.groupdict()
        data["reference_datetime"] = datetime.fromisoformat(data["reference_datetime"])
        data["secondary_datetime"] = datetime.fromisoformat(data["secondary_datetime"])
        data["generation_datetime"] = datetime.fromisoformat(
            data["generation_datetime"]
        )
        data["frame_id"] = int(data["frame_id"])

        return cls(**data)  # type: ignore
