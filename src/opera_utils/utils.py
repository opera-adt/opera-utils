from __future__ import annotations

import os
from os import fspath
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from osgeo import gdal, osr
from scipy.interpolate import RegularGridInterpolator

from ._io import load_gdal
from ._types import Bbox


def get_snwe(epsg: int, bounds: Bbox) -> Bbox:
    """Convert bounds to SNWE in lat/lon (WSEN).

    Parameters
    ----------
    epsg : int
        EPSG code.
    bounds : Tuple[float, float, float, float]
        Bounds in WSEN format.

    Returns
    -------
    Tuple[float, float, float, float]
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
    shape : Tuple[int, int]
        Shape of the dataset (ysize, xsize).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
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
) -> Tuple[ArrayLike, ArrayLike]:
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
    Tuple[ArrayLike, ArrayLike]
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
