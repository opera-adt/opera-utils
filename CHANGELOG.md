# [Unreleased](https://github.com/opera-adt/opera-utils/compare/v0.2.1...HEAD)

# [v0.2.1](https://github.com/opera-adt/opera-utils/compare/v0.2.0...v0.2.1)

**Fixed**
- Export `sort_files_by_date`
- Stop considering missing data options after 10,000 in `get_missing_data_options`

**Requirements**
- Remove scipy


# [v0.2.0](https://github.com/opera-adt/opera-utils/compare/v0.1.5...v0.2.0)

**Added**

- functions for ionosphere correction, used in `dolphin`
- Convenience functions for downloading OPERA CSLCs from ASF

**Fixed**
- changed `reproject_bounds` to use `rasterio's transform_bounds` instead of only warping 2 corners.


# [v0.1.5](https://github.com/opera-adt/opera-utils/compare/v0.1.4...v0.1.5)

**Added**
- Creates first version of the `_gslc.py` for functions working with geocoded SLCs.
  - Includes reading dataset and attributes, wavelength and zero doppler time.
- Adds methods to `_dates.py` for reformatting, sorting and grouping based on date
- Adds method to `_helpers.py` for reprojecting coordinates
- Adds methods to `_io.py` to read data with gdal and change datatype from gdal to numpy
- Adds a method to `_utils.py` to reproject and reformat the bounds from WSEN to SNWE to be compatible with some softwares like RAiDER
  - A method to create an array of x and y coordinates based on shape, geotransform and pixel size
  - A method to transform coordinates from any projection to lat/lon
- Adds test for reformatting bounds


**Requirements**
- scipy>=1.5


# [v0.1.4](https://github.com/opera-adt/opera-utils/compare/v0.1.0...v0.1.4)

**Added**
- Based raster metadata convenience functions as `_io.get_raster_*`
- Ability to download CSLC static layers and stitch into line-of-sight rasters


**Requirements**
Minimum python version is 3.9

- click>=7.0
- h5py>=1.10
- numpy>=1.20
- pooch>=1.7
- pyproj>=3.3
- shapely>=1.8
- typing_extensions>=4

For other interactions with geospatial rasters (i.e. reading metadata, creating geometries from Static Layers):
- asf_search>=6.7.2
- gdal>=3.5
- rasterio>=1.3
