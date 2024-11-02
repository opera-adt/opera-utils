from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import rasterio as rio
from osgeo import gdal

from opera_utils import get_burst_ids_for_frame, stitching
from opera_utils._types import PathOrStr
from opera_utils._utils import format_nc_filename, scratch_directory
from opera_utils.download import download_cslc_static_layers

logger = logging.getLogger(__name__)

EXTRA_COMPRESSED_TIFF_OPTIONS = (
    "COMPRESS=DEFLATE",
    "ZLEVEL=4",
    "TILED=YES",
    "BLOCKXSIZE=128",
    "BLOCKYSIZE=128",
    # Note: we're dropping mantissa bits before we do not
    # need prevision for LOS rasters (or incidence)
    "NBITS=16",
    "PREDICTOR=2",
)


class Layer(Enum):
    """Names of available datasets in CSLC static layers HDF5 files."""

    LOS_EAST = "los_east"
    LOS_NORTH = "los_north"
    LAYOVER_SHADOW_MASK = "layover_shadow_mask"
    LOCAL_INCIDENCE_ANGLE = "local_incidence_angle"
    SLANT_RANGE_DISTANCE = "slant_range_distance"
    INCIDENCE_ANGLE = "incidence_angle"


# Layover shadow mask. 0=no layover, no shadow; 1=shadow; 2=layover; 3=shadow and layover.
DEFAULT_LAYERS = list(Layer)[:3]  # Skip the local incidence, much less compressible
DEFAULT_STRIDES = {"x": 6, "y": 3}
LAYER_TO_NODATA = {
    Layer.LOS_EAST: 0,
    Layer.LOS_NORTH: 0,
    Layer.LOCAL_INCIDENCE_ANGLE: 0,
    # layover_shadow_mask is Int8 with 127 meaning nodata
    Layer.LAYOVER_SHADOW_MASK: 127,
    Layer.SLANT_RANGE_DISTANCE: 0,
    Layer.INCIDENCE_ANGLE: 0,
}


def create_geometry_files(
    *,
    frame_id: int | None = None,
    burst_ids: Sequence[str] | None = None,
    output_dir: PathOrStr = Path("."),
    download_dir: PathOrStr | None = None,
    save_hdf5_files: bool = False,
    layers: Sequence[Layer | str] = DEFAULT_LAYERS,
    strides: Mapping[str, int] = DEFAULT_STRIDES,
    max_download_jobs: int = 3,
) -> list[Path]:
    """Create merged geometry files for a frame of list of burst IDs.

    Parameters
    ----------
    frame_id : int | None, optional
        DISP frame ID to create, by default None
    burst_ids : Sequence[str] | None, optional
        Alternative to `frame_id`, manually specify CSLC burst IDs.
    output_dir : PathOrStr, optional
        Directory to store output geotiffs, by default Path(".")
    download_dir : PathOrStr | None, optional
        Directory to save files, by default None
    save_hdf5_files : bool, optional
        Flag to retain HDF5 files. Defaults to False.
    layers : Sequence[Layer | str], optional
        Layers to process. Defaults to DEFAULT_LAYERS.
    strides : Mapping[str, int], optional
        Stride values for merging images. Defaults to DEFAULT_STRIDES.
    max_download_jobs : int, optional
        Maximum number of download jobs to run in parallel. Defaults to 3.

    Returns
    -------
    list[Path]
        List of output files with paths.

    Raises
    ------
    ValueError
        If neither `frame_id` nor `burst_ids` are provided.
    """
    if frame_id is not None:
        burst_ids = get_burst_ids_for_frame(frame_id=frame_id)

    if not burst_ids:
        raise ValueError("Must provide frame_id or burst_ids")
    logger.debug("Using burst IDs: %s", burst_ids)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    if download_dir is None:
        download_dir = output_path / "hdf5"

    with scratch_directory(download_dir, delete=not save_hdf5_files) as sd:
        local_hdf5_files = download_cslc_static_layers(
            burst_ids=burst_ids, output_dir=sd, max_jobs=max_download_jobs
        )
        stitch_geometry_layers(
            local_hdf5_files=local_hdf5_files,
            layers=layers,
            strides=strides,
            output_dir=output_path,
        )

    _make_los_up(output_path)
    output_files = _make_3band_los(output_path)
    return output_files


def _make_los_up(output_path: Path):
    """Create los_up.tif from east/north."""
    los_east_path = output_path / "los_east.tif"
    los_north_path = output_path / "los_north.tif"
    los_up_path = output_path / "los_up.tif"

    with rio.open(los_east_path) as src_east, rio.open(los_north_path) as src_north:
        profile = src_east.profile
        los_east = src_east.read(1)
        los_north = src_north.read(1)

        los_up = np.sqrt(1 - los_east**2 - los_north**2)

        profile.update(
            dtype=rio.float32,
            count=1,
            compress="deflate",
            zlevel=4,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            predictor=2,
        )

        with rio.open(los_up_path, "w", **profile) as dst:
            dst.write(los_up.astype(rio.float32), 1)


def _make_3band_los(output_path: Path):
    """Combine the 3 TIFFs into one, 3-band TIFF."""
    los_east_path = output_path / "los_east.tif"
    los_north_path = output_path / "los_north.tif"
    los_up_path = output_path / "los_up.tif"
    combined_los_path = output_path / "los_combined.tif"

    with (
        rio.open(los_east_path) as src_east,
        rio.open(los_north_path) as src_north,
        rio.open(los_up_path) as src_up,
    ):
        profile = src_east.profile
        profile.update(
            count=3,
            compress="deflate",
            zlevel=4,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            predictor=2,
            interleave="band",
        )
        desc_base = "{} component of line of sight unit vector (ground to satellite)"
        with rio.open(combined_los_path, "w", **profile) as dst:
            dst.write(src_east.read(1), 1)
            dst.set_band_description(1, desc_base.format("East"))

            dst.write(src_north.read(1), 2)
            dst.set_band_description(2, desc_base.format("North"))

            dst.write(src_up.read(1), 3)
            dst.set_band_description(3, desc_base.format("Up"))

    # Delete old single-band TIFFs
    Path(los_east_path).unlink()
    Path(los_north_path).unlink()
    Path(los_up_path).unlink()

    # Create VRT files for each band
    for band, name in enumerate(["los_east", "los_north", "los_up"], start=1):
        vrt_path = output_path / f"{name}.vrt"
        gdal.BuildVRT(str(vrt_path), [str(combined_los_path)], bandList=[band])

    return [combined_los_path] + [
        output_path / f"{name}.vrt" for name in ["los_east", "los_north", "los_up"]
    ]


def stitch_geometry_layers(
    local_hdf5_files: list[Path],
    layers: Sequence[Layer | str] = DEFAULT_LAYERS,
    strides: Mapping[str, int] = DEFAULT_STRIDES,
    output_dir: PathOrStr = Path("."),
) -> list[Path]:
    """Stitch geometry layers from downloaded HDF5 files.

    Parameters
    ----------
    local_hdf5_files : list[Path]
        List of paths to the downloaded HDF5 files.
    layers : Sequence[Layer | str]
        Layers to be processed.
    strides : Mapping[str, int]
        Stride values for merging images.
    output_dir : PathOrStr
        Directory to store output GeoTIFFs.

    Returns
    -------
    list[Path]
        List of output files with paths.
    """
    output_files: list[Path] = []

    for layer in layers:
        layer_enum = Layer(layer)
        name = layer_enum.value
        gdal_strings = [
            format_nc_filename(f, ds_name=f"data/{name}") for f in local_hdf5_files
        ]
        nodata = LAYER_TO_NODATA[Layer(layer)]
        cur_outfile = Path(output_dir) / f"{name}.tif"
        output_files.append(cur_outfile)
        logger.info(f"Merging images for {name}")
        stitching.merge_images(
            file_list=gdal_strings,
            outfile=cur_outfile,
            strides=strides,
            driver="GTIff",
            options=EXTRA_COMPRESSED_TIFF_OPTIONS,
            resample_alg="nearest",
            in_nodata=nodata,
            out_nodata=nodata,
        )
    return output_files


def get_incidence_angles(static_h5file: PathOrStr, subsample_factor: int = 10):
    """Calculate incidence angles from Line Of Sight (LOS) east and north components.

    This function reads the LOS east and north components from the HDF5 file,
    downsamples them, and then calculates the incidence angle based on
    the LOS vectors.

    Parameters
    ----------
    static_h5file : PathOrStr
        Path to the HDF5 file containing the static data.
    subsample_factor : int, optional
        Factor by which to subsample the data, by default 10.

    Returns
    -------
    np.ndarray
        Array of incidence angles in degrees.

    """
    with h5py.File(static_h5file) as hf:
        ds_east = hf[f"data/{Layer.LOS_EAST.value}"]
        ds_north = hf[f"data/{Layer.LOS_NORTH.value}"]
        los_east = ds_east[::subsample_factor, ::subsample_factor]
        los_north = ds_north[::subsample_factor, ::subsample_factor]

        inc_angle_rad = np.arccos(np.sqrt(1 - los_east**2 - los_north**2))
        return np.degrees(inc_angle_rad)

    # los_east_raster = format_nc_filename(
    #     static_h5file, ds_name=f"data/{Layer.LOS_EAST.value}"
    # )
    # los_north_raster = format_nc_filename(
    #     static_h5file, ds_name=f"data/{Layer.LOS_NORTH.value}"
    # )
    # from osgeo_utils import gdal_calc

    # return gdal_calc.Calc(
    #     NoDataValue=0,
    #     format="MEM",
    #     outfile="",
    #     # type=output_type,
    #     # quiet=True,
    #     # overwrite=True,
    #     # creation_options=io.DEFAULT_TIFF_OPTIONS,
    #     E=los_east_raster,
    #     N=los_north_raster,
    #     calc="degrees(arccos(sqrt(1 - E**2 - N**2)))",
    # ).ReadAsArray()


def get_slant_range(
    cslc_h5file: PathOrStr, static_h5file: PathOrStr, subsample: int = 100
):
    """Calculate the approximate slant range for CSLC products.

    Parameters
    ----------
    cslc_h5file : PathOrStr
        Path to the HDF5 file containing the CSLC data.
    static_h5file : PathOrStr
        Path to the HDF5 file containing the static data.
    subsample : int, optional
        Factor by which to subsample the incidence data, by default 100.

    Returns
    -------
    np.ndarray
        Array of slant range values.

    Notes
    -----
    This function reads the orbit data from the CSLC HDF5 file, calculates
    the incidence angles from the static HDF5 file, and then computes the
    slant range using geometric relationships.
    """
    from opera_utils._gslc import get_orbit_arrays

    _t, x, _v, _t0 = get_orbit_arrays(cslc_h5file)
    R = np.linalg.norm(x, axis=1).mean()
    # sat_altitude = R - radius_of_earth

    # See here for other implementation
    # https://github.com/insarlab/MintPy/blob/2012127edbe81b6b0817cc6a27283eb33dfca466/src/mintpy/utils/utils0.py#L175

    incidence = get_incidence_angles(static_h5file, subsample_factor=subsample)
    incidence_rad = np.radians(incidence)
    earth_radius = 6371.0088e3

    # calculate 2R based on the law of sines
    two_times_circ = R / np.sin(np.pi - incidence_rad)

    look_angle_rad = np.arcsin(earth_radius / two_times_circ)
    range_angle_rad = incidence_rad - look_angle_rad
    return two_times_circ * np.sin(range_angle_rad)
