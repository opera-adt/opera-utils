from datetime import datetime
from pathlib import Path

import numpy as np
import tyro
from shapely import from_wkt

from opera_utils import disp
from opera_utils.disp._product import UrlType
from opera_utils.disp._reader import ReferenceMethod


def read_disp(
    frame_id: int,
    lon: float | None = None,
    lat: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    wkt: str | None = None,
    reference_method: ReferenceMethod = ReferenceMethod.NONE,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    output: Path | None = None,
    # ref_lon: Optional[float] = None,
    # ref_lat: Optional[float] = None,
    # ref_pixel: Optional[tuple[int, int]] = None,
    url_type: UrlType = UrlType.HTTPS,
    max_workers: int | None = None,
    dset: str = "displacement",
):
    """Read displacement data from a single product.

    Parameters
    ----------
    frame_id : int
        Frame ID of the product
    lon : float | None
        Longitude of the reference point
    lat : float | None
        Latitude of the reference point
    bbox : tuple[float, float, float, float] | None
        Bounding box of the region of interest
    wkt : str | None
        Well-known text representation of the region of interest
    reference_method : ReferenceMethod
        Method to use for spatially referencing displacement results
    start_datetime : datetime | None
        Start datetime of the product
    end_datetime : datetime | None
        End datetime of the product
    url_type : UrlType
        Type of URL to use for the product
    max_workers : int | None
        Maximum number of workers to use for reading
    dset : str
        Dataset name to read from the HDF5 file
    output : Path, optional
        Name of numpy file to save results to.
        If not provided, saves to current directory with default name
        of `F{frame_id:05d}_{dset}.npy`.

    Returns
    -------
    np.ndarray
        3D array of displacement values [time, height, width]
    """
    if lon is None and lat is None and bbox is None and wkt is None:
        raise ValueError("Must provide either lon, lat, or bbox")

    products = disp.search(
        frame_id=frame_id,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        url_type=url_type,
    )
    print(f"Found {len(products)} products for Frame {frame_id:05d}")
    stack = disp.DispProductStack(products)
    if wkt is not None:
        poly = from_wkt(wkt)
        bbox = poly.bounds

    if bbox is not None:
        lons: float | slice = slice(bbox[0], bbox[2])
        lats: float | slice = slice(bbox[1], bbox[3])
    else:
        assert lat is not None and lon is not None
        lons = lon
        lats = lat

    results = disp.reader.read_stack_lonlat(
        stack,
        lons=lons,
        lats=lats,
        reference_method=reference_method,
        max_workers=max_workers,
        dset=dset,
    )
    if output is None:
        output = Path(f"F{frame_id:05d}_{dset}.npy")
    np.save(output, results)


if __name__ == "__main__":
    tyro.cli(read_disp)
