# [Unreleased](https://github.com/opera-adt/opera-utils/compare/v0.1.4...main)

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
