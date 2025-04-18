import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ._product import DispProduct, DispProductStack
from ._remote import open_h5


def read_lonlat(
    product: DispProduct, lon_slice: slice, lat_slice: slice, dset: str = "displacement"
) -> np.ndarray:
    """Read data from a single product for a longitude/latitude box.

    Parameters
    ----------
    product : DispProduct
        The product to read from
    lon_slice : slice
        Longitude slice (min, max)
    lat_slice : slice
        Latitude slice (min, max)
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file

    Returns
    -------
    np.ndarray
        Data within the specified longitude/latitude box
    """
    # Convert lon/lat to row/col
    lon_start, lon_stop = lon_slice.start, lon_slice.stop
    lat_start, lat_stop = lat_slice.start, lat_slice.stop

    # Get row/col indices
    row_start, col_start = product._lonlat_to_rowcol(lon_start, lat_start)
    row_stop, col_stop = product._lonlat_to_rowcol(lon_stop, lat_stop)

    # Handle edge cases - ensure we have at least a 1x1 window
    if col_stop < col_start:
        raise ValueError(f"Invalid column range: {col_start}, {col_stop}")
    elif col_stop == col_start:
        col_stop += 1

    if row_stop < row_start:
        raise ValueError(f"Invalid row range: {row_start}, {row_stop}")
    elif row_stop == row_start:
        row_stop += 1

    # Read the data
    with open_h5(product) as hf:
        dset_obj = hf[dset]
        return dset_obj[row_start:row_stop, col_start:col_stop]


@dataclass
class ReadResult:
    data: np.ndarray
    epsg: int
    bounds: tuple[float, float, float, float]
    rows: tuple[int, int]
    cols: tuple[int, int]


def read_wkt(
    product: DispProduct,
    wkt_geom: str,
    buffer_m: float = 0.0,
    dset: str = "displacement",
) -> ReadResult:
    """Read data from a single product within a WKT geometry.

    Parameters
    ----------
    product : DispProduct
        The product to read from
    wkt_geom : str
        WKT geometry string (must be in WGS84/EPSG:4326)
    buffer_m : float, default=0.0
        Buffer around the geometry in meters
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file

    Returns
    -------
    ReadResult
        Object containing:
        - 'data': np.ndarray - The requested dataset
        - 'epsg': int - EPSG code
        - 'bounds': tuple - (xmin, ymin, xmax, ymax) in the target projection
    """
    import pyproj
    from shapely import wkt
    from shapely.ops import transform

    # Parse the WKT geometry
    geom = wkt.loads(wkt_geom)

    # Create projections
    src_crs = pyproj.CRS("EPSG:4326")
    dst_crs = pyproj.CRS(f"EPSG:{product.epsg}")
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform

    # Transform and buffer the geometry if needed
    geom_proj = transform(project, geom)
    if buffer_m > 0:
        geom_proj = geom_proj.buffer(buffer_m)

    # Get the bounds in projected coordinates
    bounds = geom_proj.bounds

    # Get rasterio transform
    profile = product.get_rasterio_profile()
    transform = profile["transform"]

    # Get min/max row/col for bounds
    # TODO: does making a minimal Affine class remove the need for this?
    from rasterio.transform import rowcol

    min_row, min_col = rowcol(transform, bounds[0], bounds[3])
    max_row, max_col = rowcol(transform, bounds[2], bounds[1])

    # Ensure proper order and at least a 1x1 window
    min_row, max_row = min(min_row, max_row), max(min_row, max_row) + 1
    min_col, max_col = min(min_col, max_col), max(min_col, max_col) + 1

    # Read the data
    with open_h5(product) as hf:
        dset_obj = hf[dset]
        data = dset_obj[min_row:max_row, min_col:max_col]

    return ReadResult(
        data=data,
        epsg=product.epsg,
        bounds=bounds,
        rows=(min_row, max_row),
        cols=(min_col, max_col),
    )


def read_point(
    product: DispProduct,
    lon: float,
    lat: float,
    window_size: int = 0,
    dset: str = "displacement",
) -> ReadResult:
    """Read data from a single product at a specified point with optional window.

    Parameters
    ----------
    product : DispProduct
        The product to read from
    lon : float
        Longitude of the point (WGS84)
    lat : float
        Latitude of the point (WGS84)
    window_size : int, default=0
        Size of window around the point (in pixels).
        0 returns single pixel, 1 returns 3x3, 2 returns 5x5, etc.
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file

    Returns
    -------
    dict
        Dictionary containing:
        - 'data': np.ndarray - The requested dataset
        - 'transform': affine.Affine - Geospatial transform
        - 'epsg': int - EPSG code
        - 'center': tuple - (row, col) of the center point
    """
    # Get row/col for the point
    row, col = product._lonlat_to_rowcol(lon, lat)

    # Calculate window bounds
    row_start = max(0, row - window_size)
    row_stop = row + window_size + 1
    col_start = max(0, col - window_size)
    col_stop = col + window_size + 1

    # Read the data
    with open_h5(product) as hf:
        dset_obj = hf[dset]
        data = dset_obj[row_start:row_stop, col_start:col_stop]

    # Get rasterio transform
    profile = product.get_rasterio_profile()
    transform = profile["transform"]

    return ReadResult(
        data=data,
        epsg=product.epsg,
        bounds=(transform * (col_start, row_start)),
        rows=(row_start, row_stop),
        cols=(col_start, col_stop),
    )


def _read_stack_lonlat(
    product: DispProduct,
    lons: slice | float,
    lats: slice | float,
    dset: str = "displacement",
) -> np.ndarray:
    """Helper function for reading from a single product with lon/lat inputs.

    Parameters
    ----------
    product : DispProduct
        The product to read from
    lons : slice or float
        Longitude slice or single value
    lats : slice or float
        Latitude slice or single value
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file

    Returns
    -------
    np.ndarray
        Data at the specified location
    """
    # Convert single values to slices if needed
    if isinstance(lons, float):
        lons = slice(lons, lons)
    if isinstance(lats, float):
        lats = slice(lats, lats)

    return read_lonlat(product, lons, lats, dset=dset)


class ReferenceMethod(Enum):
    NONE = "none"
    POINT = "point"
    CENTER = "center"
    BORDER = "border"


def read_stack_point(
    stack: DispProductStack,
    lon: float,
    lat: float,
    ref_lon: Optional[float] = None,
    ref_lat: Optional[float] = None,
    window_size: int = 0,
    ref_window_size: int = 0,
    dset: str = "displacement",
    max_workers: int = 1,
    reference_method: ReferenceMethod = ReferenceMethod.NONE,
) -> np.ndarray:
    """Read data from a stack of products at a specified point.

    Parameters
    ----------
    stack : DispProductStack
        Stack of products to read from
    lon : float
        Longitude of the point (WGS84)
    lat : float
        Latitude of the point (WGS84)
    ref_lon : float, optional
        Longitude of reference point for differencing
    ref_lat : float, optional
        Latitude of reference point for differencing
    window_size : int, default=0
        Size of window around the point (in pixels)
    ref_window_size : int, default=0
        Size of window around the reference point (in pixels)
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file
    max_workers : int, optional
        Number of processes to use for parallel reading
    reference_method : str, default="none"
        Method for referencing:
        - "none": No referencing
        - "point": Reference to specified ref_lon/ref_lat
        - "center": Reference to center of window
        - "border": Reference to border of window

    Returns
    -------
    ReadResult
        Object containing:
        - 'data': np.ndarray - Stack of data arrays
        - 'referenced_data': np.ndarray - Referenced data (if reference_method != "none")
        - 'products': list - List of products read
        - Other metadata from read_point
    """
    # Create a process pool
    ctx = mp.get_context("spawn")
    max_workers = max_workers or mp.cpu_count()

    # Function to read point data from a single product
    def _read_single_point(product):
        return read_point(product, lon, lat, window_size, dset)

    # Function to read reference point data if needed
    def _read_ref_point(product):
        if reference_method == "point" and ref_lon is not None and ref_lat is not None:
            return read_point(product, ref_lon, ref_lat, ref_window_size, dset)
        return None

    # Read data and reference data in parallel
    with ctx.Pool(processes=max_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_read_single_point, stack.products),
                total=len(stack.products),
                desc="Reading points",
            )
        )

        # Read reference points if needed
        if reference_method == "point" and ref_lon is not None and ref_lat is not None:
            ref_results = list(
                tqdm(
                    pool.imap(_read_ref_point, stack.products),
                    total=len(stack.products),
                    desc="Reading reference points",
                )
            )
        else:
            ref_results = [None] * len(stack.products)

    # Extract data arrays
    data_arrays = np.stack([r.data for r in results])

    # Apply referencing if needed
    if reference_method == "none":
        referenced_data = data_arrays
    elif reference_method == "point":
        ref_arrays = [r["data"] for r in ref_results]
        ref_values = np.stack([np.mean(r) for r in ref_arrays])
        referenced_data = data_arrays - ref_values[:, np.newaxis, np.newaxis]
    elif reference_method == "center":
        # Use center pixel of each window as reference
        centers = [r["center"] for r in results]
        ref_values = np.array(
            [data_arrays[i][centers[i]] for i in range(len(data_arrays))]
        )
        referenced_data = data_arrays - ref_values[:, np.newaxis, np.newaxis]
    elif reference_method == "border":
        referenced_data = _get_border(data_arrays)
    else:
        raise ValueError(f"Unknown {reference_method = }")

    return (referenced_data,)


# 3d array
def _get_border(data_arrays: NDArray[np.floating]) -> NDArray[np.floating]:
    top_row = data_arrays[:, 0, :]
    bottom_row = data_arrays[:, -1, :]
    left_col = data_arrays[:, :, 0]
    right_col = data_arrays[:, :, -1]
    all_pixels = np.hstack([top_row, bottom_row, left_col, right_col])
    return np.nanmedian(all_pixels, axis=1)[:, np.newaxis, np.newaxis]


def read_stack_wkt(
    stack: DispProductStack,
    wkt_geom: str,
    buffer_m: float = 0.0,
    dset: str = "displacement",
    max_workers: Optional[int] = None,
    ref_point: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Read data from a stack of products within a WKT geometry.

    Parameters
    ----------
    stack : DispProductStack
        Stack of products to read from
    wkt_geom : str
        WKT geometry string (must be in WGS84/EPSG:4326)
    buffer_m : float, default=0.0
        Buffer around the geometry in meters
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file
    max_workers : int, optional
        Number of processes to use for parallel reading
    ref_point : tuple, optional
        (lon, lat) of reference point for differencing

    Returns
    -------
    dict
        Dictionary containing:
        - 'data': np.ndarray - Stack of data arrays
        - 'referenced_data': np.ndarray - Referenced data (if ref_point is provided)
        - 'products': list - List of products read
        - 'transform': affine.Affine - Geospatial transform
        - 'epsg': int - EPSG code
        - 'bounds': tuple - (xmin, ymin, xmax, ymax) in the target projection
    """
    # Create a process pool
    ctx = mp.get_context("spawn")
    max_workers = max_workers or mp.cpu_count()

    # Function to read data from a single product
    read_func = partial(read_wkt, wkt_geom=wkt_geom, buffer_m=buffer_m, dset=dset)

    # Read data in parallel
    with ctx.Pool(processes=max_workers) as pool:
        results = list(
            tqdm(
                pool.imap(read_func, stack.products),
                total=len(stack.products),
                desc="Reading WKT regions",
            )
        )

    # Extract data arrays
    data_arrays = [r["data"] for r in results]
    stacked_data = np.stack(data_arrays)

    # Apply referencing if needed
    referenced_data = None
    if ref_point is not None:
        ref_lon, ref_lat = ref_point
        # Find the pixel coordinates of the reference point
        transform = results[0]["transform"]
        epsg = results[0]["epsg"]

        import pyproj
        from rasterio.transform import rowcol

        # Project reference point to the same CRS as the data
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        )
        ref_x, ref_y = transformer.transform(ref_lon, ref_lat)

        # Convert to row/col in the full frame
        full_row, full_col = rowcol(transform, ref_x, ref_y)

        # Adjust to the window we've read
        rows = results[0]["rows"]
        cols = results[0]["cols"]

        # Check if reference point is within our read window
        if (rows[0] <= full_row < rows[1]) and (cols[0] <= full_col < cols[1]):
            # Calculate relative position within window
            rel_row = full_row - rows[0]
            rel_col = full_col - cols[0]

            # Extract reference values and reference the stack
            ref_values = np.array([data[rel_row, rel_col] for data in data_arrays])
            referenced_data = stacked_data - ref_values[:, np.newaxis, np.newaxis]

    # Return results
    return {
        "data": stacked_data,
        "referenced_data": referenced_data,
        "products": stack.products,
        "transform": results[0]["transform"],
        "epsg": results[0]["epsg"],
        "bounds": results[0]["bounds"],
    }


def read_stack_lonlat(
    stack: DispProductStack,
    lons: Union[slice, float],
    lats: Union[slice, float],
    max_workers: Optional[int] = None,
    dset: str = "displacement",
    ref_pixel: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Process a stack using the lon/lat box method with optional multiprocessing.

    Parameters
    ----------
    stack : DispProductStack
        Stack of products to read from
    lons : slice or float
        Longitude slice or single value
    lats : slice or float
        Latitude slice or single value
    max_workers : int, optional
        Number of processes to use for parallel reading.
        If None, uses the number of CPU cores.
    dset : str, default="displacement"
        Dataset name to read from the HDF5 file
    ref_pixel : tuple, optional
        (row, col) coordinates of reference pixel relative to the read window

    Returns
    -------
    np.ndarray
        3D array with dimensions (time, lat, lon) containing the requested data
    """
    # Create a partial function with fixed parameters
    read_func = partial(_read_stack_lonlat, lons=lons, lats=lats, dset=dset)

    # Use the specified number of processes or default to CPU count
    max_workers = max_workers or mp.cpu_count()

    # Create a process pool
    ctx = mp.get_context("spawn")

    # Create a process pool and map read_func to all products
    with ctx.Pool(processes=max_workers) as pool:
        results = list(
            tqdm(
                pool.imap(read_func, stack.products),
                total=len(stack.products),
                desc="Reading products",
            )
        )

    # Stack the results
    stacked_data = np.stack(results)

    # Apply reference point if specified
    if ref_pixel is not None:
        ref_row, ref_col = ref_pixel
        ref_values = stacked_data[:, ref_row, ref_col]
        stacked_data = stacked_data - ref_values[:, np.newaxis, np.newaxis]

    return stacked_data
