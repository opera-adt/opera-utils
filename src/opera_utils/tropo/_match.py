"""Match and apply tropospheric corrections to interferograms by secondary date."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rioxarray as rxr


def read_reference_point(reference_point_file: str | Path) -> tuple[int, int]:
    """Read ``(row, col)`` from a ``reference_point.txt`` file."""
    row_col = Path(reference_point_file).read_text().strip().split(",")
    return int(row_col[0]), int(row_col[1])


def apply_tropo_correction(
    ifg_path: str | Path,
    tropo_path: str | Path,
    output_path: str | Path,
    reference_point: tuple[int, int] | None = None,
    scale: float = 1.0,
) -> None:
    """Subtract a tropospheric delay correction from an interferogram.

    Parameters
    ----------
    ifg_path : str or Path
        Input interferogram GeoTIFF (radians or metres).
    tropo_path : str or Path
        Tropospheric correction GeoTIFF on the same or compatible grid.
    output_path : str or Path
        Output path for the corrected raster.
    reference_point : tuple[int, int], optional
        ``(row, col)`` pixel used to re-reference the correction before
        subtracting, consistent with the timeseries reference point.
    scale : float
        Multiplicative scale applied to the correction before subtracting.
        Default is 1.0 (no scaling).

    """
    ifg = rxr.open_rasterio(ifg_path, masked=True).squeeze()
    tropo = rxr.open_rasterio(tropo_path, masked=True).squeeze()

    if not (ifg.rio.crs == tropo.rio.crs and ifg.shape == tropo.shape):
        tropo = tropo.rio.reproject_match(ifg)

    tropo_data = tropo.values.astype(np.float32)

    if reference_point is not None:
        ref_row, ref_col = reference_point
        ref_val = tropo_data[ref_row, ref_col]
        if np.isfinite(ref_val):
            tropo_data -= ref_val
        else:
            import warnings

            warnings.warn(
                f"Reference pixel ({ref_row}, {ref_col}) is NaN in "
                f"{Path(tropo_path).name}; skipping re-referencing.",
                stacklevel=2,
            )

    corrected = ifg - (tropo_data * scale)
    corrected.rio.write_nodata(float("nan"), inplace=True)
    corrected.rio.to_raster(output_path, dtype="float32")


def match_and_apply_tropo(
    ifg_files: list[Path],
    tropo_dir: str | Path,
    output_suffix: str = ".iono_tropo_corrected.tif",
    input_suffix: str = ".iono_corrected.tif",
    reference_point_file: str | Path | None = None,
    overwrite: bool = False,
    scale: float = 1.0,
) -> list[tuple[Path, Path, Path]]:
    """Match interferograms to tropospheric corrections by secondary date and apply.

    Parameters
    ----------
    ifg_files : list[Path]
        Interferogram GeoTIFF paths named ``{ref}_{sec}<input_suffix>``.
    tropo_dir : str or Path
        Directory containing ``tropo_correction_{sec}T*.tif`` files.
    output_suffix : str
        Suffix for output corrected files.
    input_suffix : str
        Suffix to strip from input filenames when building output names.
    reference_point_file : str or Path, optional
        Path to ``reference_point.txt`` containing ``row,col``.  If provided,
        the correction is re-referenced to that pixel before subtracting.
    overwrite : bool
        If False (default), skip files that already exist.
    scale : float
        Multiplicative scale applied to each correction before subtracting.

    Returns
    -------
    list[tuple[Path, Path, Path]]
        ``(ifg_path, tropo_path, output_path)`` for each file processed.

    """
    tropo_dir = Path(tropo_dir)

    reference_point: tuple[int, int] | None = None
    if reference_point_file is not None:
        reference_point = read_reference_point(reference_point_file)

    tropo_by_date: dict[str, Path] = {}
    for p in tropo_dir.glob("tropo_correction_*.tif"):
        m = re.search(r"tropo_correction_(\d{8})", p.name)
        if m:
            tropo_by_date[m.group(1)] = p

    processed: list[tuple[Path, Path, Path]] = []
    for ifg_path in ifg_files:
        m = re.match(r"(\d{8})_(\d{8})", ifg_path.name)
        if not m:
            continue

        sec_date = m.group(2)
        if sec_date not in tropo_by_date:
            continue

        tropo_path = tropo_by_date[sec_date]
        stem = ifg_path.name[: -len(input_suffix)]
        output_path = ifg_path.parent / f"{stem}{output_suffix}"

        if output_path.exists() and not overwrite:
            continue

        apply_tropo_correction(
            ifg_path,
            tropo_path,
            output_path,
            reference_point=reference_point,
            scale=scale,
        )
        processed.append((ifg_path, tropo_path, output_path))

    return processed
