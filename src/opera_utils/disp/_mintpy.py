"""ALL THE STUFF THAT CAN BE HARD CODED CUZ MINTPY JUST NEEDS APPROX VALUES.

In [25]: p = disp.DispProduct.from_filename(
    "data/copy-subsets-nyc-f08622/OPERA_L3_DISP-S1_IW_F08622_VV_20170313T225042Z_20170723T225049Z_v1.0_20250414T034650Z.nc"
)

In [26]: p.epsg
Out[26]: 32618

In [24]: c = p.crs
In [27]: c
Out[27]:
<Projected CRS: EPSG:32618>
Name: WGS 84 / UTM zone 18N
...
- Prime Meridian: Greenwich

In [32]: c.utm_zone
Out[32]: '18N'

In [28]: p.shape
Out[28]: (7959, 9587)

In [29]: p.get_rasterio_profile()
Out[29]:
{'driver': 'GTiff',
...
 'width': 9587,
 'height': 7959,
 'transform': Affine(30.0, 0.0, 464040.0,
        0.0, -30.0, 4637340.0),
 'crs': 'EPSG:32618'}

"""

from pathlib import Path
from typing import Literal

import numpy as np
from mintpy import readfile
from numpy.typing import ArrayLike


def calc_azimuth_from_east_north_obs(east, north):
    """Calculate the azimuth angle of the given horizontal observation.

    Calculates azimuth from east and north components.

    Parameters: east     - float,  eastward motion
                north    - float, northward motion
    Returns:    az_angle - float, azimuth angle in degree
                           measured from the north with anti-clockwise as positive
    """
    az_angle = -1 * np.rad2deg(np.arctan2(east, north)) % 360
    return az_angle


def azimuth2heading_angle(
    az_angle: ArrayLike, look_direction: Literal["right", "left"] = "right"
):
    """Convert azimuth angle from ISCE los.rdr band2 into satellite orbit heading angle.

    ISCE-2 los.* file band2 is azimuth angle of LOS vector from ground target
    to the satellite measured from the north in anti-clockwise as positive

    Below are typical values in deg for satellites with near-polar orbit:
        ascending  orbit: heading angle of -12  and azimuth angle of 102
        descending orbit: heading angle of -168 and azimuth angle of -102
    """
    if look_direction == "right":
        head_angle = (np.asarray(az_angle) - 90) * -1
    else:
        head_angle = (np.asarray(az_angle) + 90) * -1
    head_angle -= np.round(head_angle / 360.0) * 360.0
    return head_angle


def east_north_to_azimuth(east: ArrayLike, north: ArrayLike):
    return -1 * np.rad2deg(np.arctan2(east, north)) % 360


def prepare_metadata(_meta_file, _int_file, geom_dir):
    """Get the metadata from the GSLC metadata file and the unwrapped interferogram."""
    print("-" * 50)

    # TODO: Implement actual metadata extraction
    # cols, rows = get_raster_xysize(int_file)
    # geotransform = get_raster_gt(int_file)
    # crs = get_raster_crs(int_file)

    meta = {}
    # Placeholder values - need actual implementation
    rows, cols = 1000, 1000  # placeholder
    geotransform = [0, 30, 0, 0, 0, -30]  # placeholder

    meta["LENGTH"] = rows
    meta["WIDTH"] = cols

    meta["X_FIRST"] = geotransform[0]
    meta["Y_FIRST"] = geotransform[3]
    meta["X_STEP"] = geotransform[1]
    meta["Y_STEP"] = geotransform[5]
    meta["X_UNIT"] = meta["Y_UNIT"] = "meters"

    # h5py.File(meta_file, "r")
    # TODO: USE rasterio
    # crs = get_raster_crs(int_file)
    # meta["EPSG"] = crs.to_epsg()
    meta["EPSG"] = 32618  # placeholder

    if str(meta["EPSG"]).startswith("326"):
        meta["UTM_ZONE"] = str(meta["EPSG"])[3:] + "N"
    else:
        meta["UTM_ZONE"] = str(meta["EPSG"])[3:] + "S"

    meta["WAVELENGTH"] = 0.0554657
    meta["EARTH_RADIUS"] = 6371000.0

    # get heading from azimuth angle
    geom_path = Path(geom_dir)
    file_to_path = {
        "los_east": geom_path / "los_east.tif",
        "los_north": geom_path / "los_north.tif",
    }
    dsDict = {}
    for dsName, fname in file_to_path.items():
        data = readfile.read(fname, datasetName=dsName)[0]
        data[data == 0] = np.nan
        dsDict[dsName] = data
    azimuth_angle = east_north_to_azimuth(dsDict["los_east"], dsDict["los_north"])
    azimuth_angle = np.nanmean(azimuth_angle)
    heading = azimuth2heading_angle(azimuth_angle)
    meta["HEADING"] = heading

    # t0 = dt.strptime(
    #     meta_compass[f"{burst_ds}/sensing_start][()].decode("utf-8"),
    #     "%Y-%m-%d %H:%M:%S.%f",
    # )
    # t1 = dt.strptime(
    #     meta_compass[f"{burst_ds}/sensing_stop"][()].decode("utf-8"),
    #     "%Y-%m-%d %H:%M:%S.%f",
    # )
    # t_mid = t0 + (t1 - t0) / 2.0
    # center_line_utc = (
    #     t_mid - dt(t_mid.year, t_mid.month, t_mid.day)
    # ).total_seconds()
    # TODO: Get actual center line UTC from product
    # center_line_utc = product.secondary_datetime + timedelta(seconds=15)
    center_line_utc = 0.0  # placeholder
    meta["CENTER_LINE_UTC"] = center_line_utc
    # Approximate height of the satellite,
    # "used in dem_error, incidence_angle, convert2mat"
    meta["HEIGHT"] = 750000.0
    # HARD CODE
    # STARTING_RANGE = Distance from satellite to first ground pixel in meters,
    # used in incidence_angle calculation (SO WE DONT NEED THAT)
    # meta["STARTING_RANGE"] = meta_compass[f"{burst_ds}/starting_range"][()]
    meta["PLATFORM"] = "Sentinel-1"
    # orbit_direction is in the netcdf product, in /identification/orbit_pass_direction
    #
    # TODO: Get actual orbit direction from product
    # meta["ORBIT_DIRECTION"] = hf_product.orbit_direction
    meta["ORBIT_DIRECTION"] = "ASCENDING"  # placeholder
    # RANGE_PIXEL_SIZE: "used in dem_error, incidence_angle, multilook, transect.""
    # DONT NEED FOR GEO
    # https://github.com/insarlab/MintPy/blob/86a07edb946eda33c16f4e36aaf37c0c4001a20e/src/mintpy/multi_transect.py#L603-L613
    # meta["RANGE_PIXEL_SIZE"] = meta_compass[f"{burst_ds}/range_pixel_spacing"][()]
    # meta["AZIMUTH_PIXEL_SIZE"] = 14.1
    # ALOOKS/RLOOKS = multilook number in azimuth/range direction,
    # used in weighted network inversion.
    # We dont need inversion things
    # meta["ALOOKS"] = 1
    # meta["RLOOKS"] = 1
    # # used in dem_error, incidence_angle, multilook, transect.
    # meta["RANGE_PIXEL_SIZE"] = str(float(meta["RANGE_PIXEL_SIZE"]) * nlks_x)
    # meta["RLOOKS"] = str(float(meta["RLOOKS"]) * nlks_x)

    # meta["AZIMUTH_PIXEL_SIZE"] = str(float(meta["AZIMUTH_PIXEL_SIZE"]) * nlks_y)
    # meta["ALOOKS"] = str(float(meta["ALOOKS"]) * nlks_y)

    # NEED UNITS:
    meta["FILE_TYPE"] = "timeseries"
    meta["UNIT"] = "m"

    return meta


def mintpy_prepare_geometry(
    outfile, _int_file, geom_dir, metadata, water_mask_file=None
):
    """Prepare the geometry file."""
    print("-" * 50)
    print(f"preparing geometry file: {outfile}")

    geom_path = Path(geom_dir)
    # copy metadata to meta
    meta = dict(metadata.items())
    meta["FILE_TYPE"] = "geometry"

    file_to_path = {
        "los_enu": next(iter(geom_path.glob("*los_enu.tif"))),
        "height": next(iter(geom_path.glob("*dem.tif"))),
        "shadowMask": next(iter(geom_path.glob("*layover_shadow_mask.tif"))),
    }

    if water_mask_file:
        file_to_path["waterMask"] = water_mask_file

    dsDict = {}
    for dsName, fname in file_to_path.items():
        try:
            data = readfile.read(fname, datasetName=dsName)[0]
            if dsName not in ["shadowMask", "waterMask"]:
                data[data == 0] = np.nan
            dsDict[dsName] = data

            # write data to HDF5 file
        except FileNotFoundError as e:  # https://github.com/insarlab/MintPy/issues/1081
            print(f"Skipping {fname}: {e}")

    # Compute the azimuth and incidence angles from east/north coefficients
    azimuth_angle = east_north_to_azimuth(dsDict["los_east"], dsDict["los_north"])
    dsDict["azimuthAngle"] = azimuth_angle

    # up = np.sqrt(1 - east**2 - north**2)
    # Up is the 3rd band of the LOS image

    # TODO: Calculate incidence angle from LOS up component
    # up = calculate_up_component(dsDict)
    # incidence_angle = np.rad2deg(np.arccos(up))
    incidence_angle = np.full_like(azimuth_angle, 35.0)  # placeholder
    dsDict["incidenceAngle"] = incidence_angle

    # write out slant range distance
    # slant_range = rio.open(int_file).get_rasterio_profile()
    # from opera_utils.geometry import get_slant_range

    # TODO: Get actual slant range from file
    # slant_range = get_slant_range(int_file)
    # slant_range = float(slant_range.groups["metadata"]["slant_range_mid_swath"][0])
    slant_range_val = 850000.0  # placeholder
    slant_range = np.full_like(
        incidence_angle, fill_value=slant_range_val, dtype=np.float32
    )
    dsDict["slantRangeDistance"] = slant_range

    # TODO: Import writefile
    # from mintpy import writefile
    # writefile.write(dsDict, outfile, metadata=meta)
    print(f"Would write to {outfile} with metadata")
    return outfile
