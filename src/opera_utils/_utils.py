from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from os import fspath
from pathlib import Path
from typing import Generator

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from osgeo import gdal, osr
from scipy.interpolate import RegularGridInterpolator

from ._io import load_gdal
from ._types import Bbox, PathOrStr

__all__ = [
    "format_nc_filename",
    "scratch_directory",
    "create_yx_arrays",
    "get_snwe",
    "transform_xy_to_latlon",
    "compute_2d_delay",
]


def format_nc_filename(filename: PathOrStr, ds_name: str | None = None) -> str:
    """Format an HDF5/NetCDF filename with dataset for reading using GDAL.

    If `filename` is already formatted, or if `filename` is not an HDF5/NetCDF
    file (based on the file extension), it is returned unchanged.

    Parameters
    ----------
    filename : str or PathLike
        Filename to format.
    ds_name : str, optional
        Dataset name to use. If not provided for a .h5 or .nc file, an error is raised.

    Returns
    -------
    str
        Formatted filename like
        NETCDF:"filename.nc":"//ds_name"

    Raises
    ------
    ValueError
        If `ds_name` is not provided for a .h5 or .nc file.
    """
    # If we've already formatted the filename, return it
    if str(filename).startswith("NETCDF:") or str(filename).startswith("HDF5:"):
        return str(filename)

    if not (os.fspath(filename).endswith(".nc") or os.fspath(filename).endswith(".h5")):
        return os.fspath(filename)

    # Now we're definitely dealing with an HDF5/NetCDF file
    if ds_name is None:
        raise ValueError("Must provide dataset name for HDF5/NetCDF files")

    return f'NETCDF:"{filename}":"//{ds_name.lstrip("/")}"'


def _get_path_from_gdal_str(name: PathOrStr) -> Path:
    s = str(name)
    if s.upper().startswith("DERIVED_SUBDATASET"):
        # like DERIVED_SUBDATASET:AMPLITUDE:slc_filepath.tif
        p = s.split(":")[-1].strip('"').strip("'")
    elif ":" in s and (s.upper().startswith("NETCDF") or s.upper().startswith("HDF")):
        # like NETCDF:"slc_filepath.nc":subdataset
        p = s.split(":")[1].strip('"').strip("'")
    else:
        # Whole thing is the path
        p = str(name)
    return Path(p)


def numpy_to_gdal_type(np_dtype: DTypeLike) -> int:
    """Convert numpy dtype to gdal type.

    Parameters
    ----------
    np_dtype : DTypeLike
        Numpy dtype to convert.

    Returns
    -------
    int
        GDAL type code corresponding to `np_dtype`.

    Raises
    ------
    TypeError
        If `np_dtype` is not a numpy dtype, or if the provided dtype is not
        supported by GDAL (for example, `np.dtype('>i4')`)
    """
    from osgeo import gdal_array, gdalconst

    np_dtype = np.dtype(np_dtype)

    if np.issubdtype(bool, np_dtype):
        return gdalconst.GDT_Byte
    gdal_code = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)
    if gdal_code is None:
        raise TypeError(f"dtype {np_dtype} not supported by GDAL.")
    return gdal_code


def gdal_to_numpy_type(gdal_type: str | int) -> np.dtype:
    """Convert gdal type to numpy type."""
    from osgeo import gdal, gdal_array

    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


@contextmanager
def scratch_directory(
    dir_: PathOrStr | None = None, *, delete: bool = True
) -> Generator[Path, None, None]:
    """Context manager that creates a (possibly temporary) file system directory.

    If `dir_` is a path-like object, a directory will be created at the specified
    file system path if it did not already exist. Otherwise, if `dir_` is None, a
    temporary directory will instead be created as though by
    ``tempfile.TemporaryDirectory()``.

    The directory may be automatically removed from the file system upon exiting the
    context manager.

    Parameters
    ----------
    dir_ : PathOrStr or None, optional
        Scratch directory path. If None, a temporary directory will be created. Defaults
        to None.
    delete : bool, optional
        If True and `dir_` didn't exist, the directory and its contents are
        recursively removed from the file system upon exiting the context manager.
        Note that if `dir_` already exists, this option is ignored.
        Defaults to True.

    Yields
    ------
    pathlib.Path
        Scratch directory path. If `delete` was True, the directory will be removed from
        the file system upon exiting the context manager scope.
    """
    if dir_ is None:
        scratchdir = Path(tempfile.mkdtemp())
        dir_already_existed = False
    else:
        scratchdir = Path(dir_)
        dir_already_existed = scratchdir.exists()
        scratchdir.mkdir(parents=True, exist_ok=True)

    yield scratchdir

    if delete and not dir_already_existed:
        shutil.rmtree(scratchdir)


def get_snwe(epsg: int, bounds: Bbox) -> Bbox:
    """Convert bounds to SNWE in lat/lon (WSEN).

    Parameters
    ----------
    epsg : int
        EPSG code.
    bounds : tuple[float, float, float, float]
        Bounds in WSEN format.

    Returns
    -------
    tuple[float, float, float, float]
        Bounds in SNWE (lat/lon) format.
    """
    if epsg != 4326:
        # x, y to Lat/Lon
        srs_src = osr.SpatialReference()
        srs_src.ImportFromEPSG(epsg)

        srs_wgs84 = osr.SpatialReference()
        srs_wgs84.ImportFromEPSG(4326)

        # Transform the xy to lat/lon
        transformer_xy_to_latlon = osr.CoordinateTransformation(srs_src, srs_wgs84)

        # Stack the x and y
        x_y_pnts_radar = np.stack(
            ([bounds[0], bounds[2]], [bounds[1], bounds[3]]), axis=-1
        )

        # Transform to lat/lon
        lat_lon_radar = np.array(
            transformer_xy_to_latlon.TransformPoints(x_y_pnts_radar)
        )

        snwe = (
            lat_lon_radar[0, 0],
            lat_lon_radar[1, 0],
            lat_lon_radar[0, 1],
            lat_lon_radar[1, 1],
        )
    else:
        snwe = (bounds[1], bounds[3], bounds[0], bounds[2])

    return snwe


def create_yx_arrays(
    gt: list[float], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Create the x and y coordinate datasets.

    Parameters
    ----------
    gt : List[float]
        Geotransform list.
    shape : tuple[int, int]
        Shape of the dataset (ysize, xsize).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x and y coordinate arrays.
    """
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt
    y_end = y_origin + y_res * ysize
    x_end = x_origin + x_res * xsize

    # Make the x/y arrays
    y = np.arange(y_origin, y_end - 500, -500)
    x = np.arange(x_origin, x_end + 500, 500)
    return y, x


def transform_xy_to_latlon(
    epsg: int, x: ArrayLike, y: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """Convert the x, y coordinates in the source projection to WGS84 lat/lon.

    Parameters
    ----------
    epsg : int
        EPSG code.
    x : ArrayLike
        x coordinates.
    y : ArrayLike
        y coordinates.

    Returns
    -------
    tuple[ArrayLike, ArrayLike]
        Latitude and longitude arrays.
    """
    # x, y to Lat/Lon
    srs_src = osr.SpatialReference()
    srs_src.ImportFromEPSG(epsg)

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    if epsg != 4326:
        # Transform the xy to lat/lon
        transformer_xy_to_latlon = osr.CoordinateTransformation(srs_src, srs_wgs84)

        # Stack the x and y
        x_y_pnts_radar = np.stack((x.flatten(), y.flatten()), axis=-1)

        # Transform to lat/lon
        lat_lon_radar = np.array(
            transformer_xy_to_latlon.TransformPoints(x_y_pnts_radar)
        )

        # Lat lon of data cube
        lat_datacube = lat_lon_radar[:, 0].reshape(x.shape)
        lon_datacube = lat_lon_radar[:, 1].reshape(x.shape)
    else:
        lat_datacube = y.copy()
        lon_datacube = x.copy()

    ## Extent of the data cube
    # cube_extent = (np.nanmin(lat_datacube) - margin, np.nanmax(lat_datacube) + margin,
    #               np.nanmin(lon_datacube) - margin, np.nanmax(lon_datacube) + margin)

    return lat_datacube, lon_datacube  # , cube_extent


def compute_2d_delay(
    tropo_delay_cube: dict, grid: dict, geo_files: dict[str, Path]
) -> dict:
    """Compute 2D delay.

    Parameters
    ----------
    tropo_delay_cube : dict
        Dictionary containing tropospheric delay data.
    grid : dict
        Dictionary containing grid information.
    geo_files : dict[str, Path]
        Dictionary containing paths to geospatial files.

    Returns
    -------
    dict
        Dictionary containing computed 2D delay.
    """
    dem_file = geo_files["height"]

    ysize, xsize = grid["shape"]
    x_origin, x_res, _, y_origin, _, y_res = grid["geotransform"]

    gt = grid["geotransform"]

    x = 0
    y = 0
    left = gt[0] + x * gt[1] + y * gt[2]
    top = gt[3] + x * gt[4] + y * gt[5]

    x = xsize
    y = ysize

    right = gt[0] + x * gt[1] + y * gt[2]
    bottom = gt[3] + x * gt[4] + y * gt[5]

    bounds = (left, bottom, right, top)

    options = gdal.WarpOptions(
        dstSRS=grid["crs"],
        format="MEM",
        xRes=x_res,
        yRes=y_res,
        outputBounds=bounds,
        outputBoundsSRS=grid["crs"],
        resampleAlg="near",
    )
    target_ds = gdal.Warp(
        os.path.abspath(fspath(dem_file) + ".temp"),
        os.path.abspath(fspath(dem_file)),
        options=options,
    )

    dem = target_ds.ReadAsArray()

    los_east = load_gdal(geo_files["los_east"])
    los_north = load_gdal(geo_files["los_north"])
    los_up = 1 - los_east**2 - los_north**2

    mask = los_east > 0

    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    y = np.arange(y_origin, y_origin + y_res * ysize, y_res)
    x = np.arange(x_origin, x_origin + x_res * xsize, x_res)

    yv, xv = np.meshgrid(y, x, indexing="ij")

    delay_2d = {}
    for delay_type in tropo_delay_cube.keys():
        if delay_type not in ["x", "y", "z"]:
            # tropo_delay_datacube_masked = np.ma.masked_invalid(tropo_delay_cube[delay_type])

            tropo_delay_interpolator = RegularGridInterpolator(
                (grid["height_levels"], grid["ycoord"], grid["xcoord"]),
                tropo_delay_cube[delay_type],
                method="linear",
                bounds_error=False,
            )

            tropo_delay_2d = np.zeros(dem.shape, dtype=np.float32)

            nline = 100
            for i in range(0, dem.shape[1], 100):
                if i + 100 > dem.shape[0]:
                    nline = dem.shape[0] - i
                pnts = np.stack(
                    (
                        dem[i : i + 100, :].flatten(),
                        yv[i : i + 100, :].flatten(),
                        xv[i : i + 100, :].flatten(),
                    ),
                    axis=-1,
                )
                tropo_delay_2d[i : i + 100, :] = tropo_delay_interpolator(pnts).reshape(
                    nline, dem.shape[1]
                )

            out_delay_type = delay_type.replace("Zenith", "LOS")
            delay_2d[out_delay_type] = (tropo_delay_2d / los_up) * mask

    return delay_2d
