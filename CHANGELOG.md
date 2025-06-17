# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased](https://github.com/opera-adt/opera-utils/compare/v0.23.0...HEAD)

## [0.23.0](https://github.com/opera-adt/opera-utils/compare/v0.22.1...v0.23.0) - 2025-06-16

### Added

- Add spatial reference to `disp-s1-reformat` by @scottstanie in https://github.com/opera-adt/opera-utils/pull/147
- Add `NanPolicy`, `shards` to `disp-s1-reformat` output by @scottstanie in https://github.com/opera-adt/opera-utils/pull/144
- `disp-s1-reformat`: Add quality masking options by @scottstanie in https://github.com/opera-adt/opera-utils/pull/145
- `plot-disp-s1-intersecting-frames.py`: Add `--wkt` and `--no-ascending`/`--no-descending` filters by @scottstanie in https://github.com/opera-adt/opera-utils/pull/146

## [0.22.1](https://github.com/opera-adt/opera-utils/compare/v0.22.0...v0.22.1) - 2025-06-05

### Added
- Write out `short_wavelength_displacement` unless dropped by @scottstanie in https://github.com/opera-adt/opera-utils/pull/142

### Fixed
- Update disp download to allow `s3://` urls by @scottstanie in https://github.com/opera-adt/opera-utils/pull/141

## [0.22.0](https://github.com/opera-adt/opera-utils/compare/v0.21.0...v0.22.0) - 2025-06-04

### Added
- Add alternative DISP stack reformatters by @scottstanie in https://github.com/opera-adt/opera-utils/pull/140
- Add `disp-s1-download` for a subsetted download by @scottstanie in https://github.com/opera-adt/opera-utils/pull/137
- `plot-disp-s1-frame.py`: Add multiple frame option by @scottstanie in https://github.com/opera-adt/opera-utils/pull/134
- `download`: add `search-l2` to CLI interface by @scottstanie in https://github.com/opera-adt/opera-utils/pull/136
- `disp-s1-download`: Add `/metadata` and `/identification` to outputs by @scottstanie in https://github.com/opera-adt/opera-utils/pull/139

### Changed
- `cslc`: Add Compressed SLC file regex to `parse_filename` by @scottstanie in https://github.com/opera-adt/opera-utils/pull/135

## [0.21.0](https://github.com/opera-adt/opera-utils/compare/v0.20.0...v0.21.0) - 2025-05-19

### Added
- Add optional `disp-s1-search` to cli by @scottstanie in https://github.com/opera-adt/opera-utils/pull/130
- Add `reference_lonlat` option to `rebase_reference` by @scottstanie in https://github.com/opera-adt/opera-utils/pull/131

## [0.20.0](https://github.com/opera-adt/opera-utils/compare/v0.19.0...v0.20.0) - 2025-05-09

### Added

- Add DISP-S1 product handling by @scottstanie in https://github.com/opera-adt/opera-utils/pull/106
- Add `plot-disp-s1-frame.py` and `plot-intersecting-frames.py` scripts for visualization DISP-S1 Frames by @scottstanie in https://github.com/opera-adt/opera-utils/pull/110
- Add `credentials` and `disp._remote` module with `open_h5` for https/s3 access by @scottstanie in https://github.com/opera-adt/opera-utils/pull/108

### Changed
- Make `disp-s1` and `disp-nisar` sub-commands, add `intersects` by @scottstanie in https://github.com/opera-adt/opera-utils/pull/104
- Use `tyro` for CLI instead of `click`
- Make one single conda environment, let users `pip` install for smaller dependency footprint

### Fixed

- Add missing tests for `burst_frame_db` and `stitching` modules by @scottstanie in https://github.com/opera-adt/opera-utils/pull/105
- Restrict `str` usage for `ASFCredentialEndpoints` by @scottstanie in https://github.com/opera-adt/opera-utils/pull/109
- Use `Enum` for `UrlType` instead of `Literal["s3", "https"]` by @scottstanie in https://github.com/opera-adt/opera-utils/pull/111


## [0.19.0](https://github.com/opera-adt/opera-utils/compare/v0.18.1...v0.19.0) - 2025-04-11

### Added
- Add DISP-S1 product handling: search with `opera_utils.disp.search`, read and parse files with `opera_utils.disp.DispProduct` and `DispProductStack`
- Add `credentials` and `disp._remote` module with `open_h5` for https/s3 access
- Add `plot-disp-s1-frame.py` and `plot-intersecting-frames.py` scripts for visualization DISP-S1 Frames

### Changed
- Make `disp-s1` and `disp-nisar` sub-commands, add `intersects`

### Fixed
- Add missing tests for `stitching` and `burst_frame_db` modules
- Use `Enum` for `UrlType` instead of `Literal["s3", "https"]` by @scottstanie in https://github.com/opera-adt/opera-utils/pull/111
- Restrict `str` usage for `ASFCredentialEndpoints`


## [0.18.1](https://github.com/opera-adt/opera-utils/compare/v0.18.0...v0.18.1) - 2025-03-25

### Changed
- Use `pyproject` sections instead of `*/requirements.txt` by @scottstanie ([#102](https://github.com/opera-adt/opera-utils/pull/102))
- Check for project name instead of sensor by @mirzaees ([#103](https://github.com/opera-adt/opera-utils/pull/103))

## [0.18.0](https://github.com/opera-adt/opera-utils/compare/v0.17.0...v0.18.0) - 2025-03-24

### Fixed
- Fix `stitching` for strided outputs by @scottstanie ([#98](https://github.com/opera-adt/opera-utils/pull/98))

### Changed
- Use only static layers to get `slant_range` and allow manual `incidence_angles` by @scottstanie ([#99](https://github.com/opera-adt/opera-utils/pull/99))

### Added
- Add ability to read NISAR subdatasets in `create_nodata_mask` and `get_zero_doppler_time` by @mirzaees ([#100](https://github.com/opera-adt/opera-utils/pull/100))

## [0.17.0](https://github.com/opera-adt/opera-utils/compare/v0.16.0...v0.17.0) - 2025-03-18

### Changed
- Update Burst DB data to v0.9.0 by @scottstanie ([#91](https://github.com/opera-adt/opera-utils/pull/91))

### Fixed
- Fix output types for `date | datetime` by @scottstanie ([#93](https://github.com/opera-adt/opera-utils/pull/93))

### Added
- In `download.py`: Add `download_rtcs` and `download_rtc_static_layers` by @scottstanie ([#95](https://github.com/opera-adt/opera-utils/pull/95))

---

## [0.16.0](https://github.com/opera-adt/opera-utils/compare/v0.15.0...v0.16.0) - 2025-02-12

### Added
- Add bounds options to `stitch_geometry_layers` by @scottstanie ([#88](https://github.com/opera-adt/opera-utils/pull/88))

### Changed
- Update Burst DB data to v0.8.0 by @scottstanie ([#90](https://github.com/opera-adt/opera-utils/pull/90))

---

## [0.15.0](https://github.com/opera-adt/opera-utils/compare/v0.14.0...v0.15.0) - 2025-01-14

### Fixed
- Fix second datetime in CSLC name to be `generation_datetime` by @scottstanie ([#82](https://github.com/opera-adt/opera-utils/pull/82))

### Changed
- Refactor filtering to work with/without geopandas by @scottstanie ([#81](https://github.com/opera-adt/opera-utils/pull/81))
- Bump `burst_db` version to 0.7.0 by @scottstanie ([#84](https://github.com/opera-adt/opera-utils/pull/84))

---

## [0.14.0](https://github.com/opera-adt/opera-utils/compare/v0.13.0...v0.14.0) - 2024-11-02

### Added
- Use `search_cslcs` to also search for Static Layers by @scottstanie ([#73](https://github.com/opera-adt/opera-utils/pull/73))
- Add geometry functions for incidence and approximate slant range by @scottstanie ([#40](https://github.com/opera-adt/opera-utils/pull/40))

### Changed
- Normalize burst ID string before download by @scottstanie ([#77](https://github.com/opera-adt/opera-utils/pull/77))

---

## [0.13.0](https://github.com/opera-adt/opera-utils/compare/v0.12.1...v0.13.0) - 2024-10-01

### Changed
- Combine LOS east, north, up in one file in `geometry` by @scottstanie ([#68](https://github.com/opera-adt/opera-utils/pull/68))
- Add better named alias `create_nodata_mask` by @scottstanie ([#66](https://github.com/opera-adt/opera-utils/pull/66))

### Added
- Implement `get_urls(type_="s3")` to fetch CSLC S3 URLs from `asf_search` by @scottstanie ([#71](https://github.com/opera-adt/opera-utils/pull/71))

---

## [0.12.1](https://github.com/opera-adt/opera-utils/compare/v0.12.0...v0.12.1) - 2024-08-29

### Fixed
- Fix grouping by `startTime` to not skip bursts by @scottstanie ([#62](https://github.com/opera-adt/opera-utils/pull/62))
- Filter CSLCs by PGE version and processing date by @scottstanie ([#64](https://github.com/opera-adt/opera-utils/pull/64))

---

## [0.12.0](https://github.com/opera-adt/opera-utils/compare/v0.11.0...v0.12.0) - 2024-08-12

### Added
- Add `.inputs` to the missing data options to track selections by @scottstanie ([#59](https://github.com/opera-adt/opera-utils/pull/59))

---

## [0.11.0](https://github.com/opera-adt/opera-utils/compare/v0.10.0...v0.11.0) - 2024-08-01

### Added
- Load/make thumbnail for any 2D image by @scottstanie ([#56](https://github.com/opera-adt/opera-utils/pull/56))

### Fixed
- Parse EPSG from `attrs["spatial_ref"]` by @scottstanie ([#57](https://github.com/opera-adt/opera-utils/pull/57))

---

## [0.10.0](https://github.com/opera-adt/opera-utils/compare/v0.9.0...v0.10.0) - 2024-07-29

### Added
- Add `normalize_burst_id` by @scottstanie ([#54](https://github.com/opera-adt/opera-utils/pull/54))

### Fixed
- Fix parsing to allow COMPASS filenames by @scottstanie ([#54](https://github.com/opera-adt/opera-utils/pull/54))

---

## [0.9.0](https://github.com/opera-adt/opera-utils/compare/v0.8.0...v0.9.0) - 2024-07-29

### Added
- Add `parse_filename` to avoid hardcoding `VV` and use Python `Rasterize` bindings by @scottstanie ([#52](https://github.com/opera-adt/opera-utils/pull/52))

---

## [0.8.0](https://github.com/opera-adt/opera-utils/compare/v0.7.0...v0.8.0) - 2024-07-15

### Added
- Add a maximum number of options to `missing-data-options` by @scottstanie ([#43](https://github.com/opera-adt/opera-utils/pull/43))
- Add a `search_cslcs` check for results/missing data before download by @scottstanie ([#48](https://github.com/opera-adt/opera-utils/pull/48))

### Fixed
- Fix CSLC static layers filtering/stitching and burst-to-frame mapping by @scottstanie ([#50](https://github.com/opera-adt/opera-utils/pull/50))

### Changed
- Use `NBITS=16` when saving LOS files by @scottstanie ([#51](https://github.com/opera-adt/opera-utils/pull/51))

---

## [0.7.0](https://github.com/opera-adt/opera-utils/compare/v0.6.0...v0.7.0) - 2024-07-08

### Changed
- Update burst DB datasets to 0.5.0 by @scottstanie ([#47](https://github.com/opera-adt/opera-utils/pull/47))

---

## [0.6.0](https://github.com/opera-adt/opera-utils/compare/v0.5.0...v0.6.0) - 2024-06-21

### Added
- Add `missing-data` CLI tool for subsetting acquisitions by @scottstanie ([#42](https://github.com/opera-adt/opera-utils/pull/42))

---

## [0.5.0](https://github.com/opera-adt/opera-utils/compare/v0.4.1...v0.5.0) - 2024-06-18

### Removed
- Remove duplicate CSLCs with different PGE versions by @scottstanie ([#41](https://github.com/opera-adt/opera-utils/pull/41))

---

## [0.4.1](https://github.com/opera-adt/opera-utils/compare/v0.4.0...v0.4.1) - 2025-06-17

### Changed
- Return degrees for `get_lonlat_grid` (not radians) by @scottstanie ([#38](https://github.com/opera-adt/opera-utils/pull/38))

---

## [0.4.0](https://github.com/opera-adt/opera-utils/compare/v0.3.2...v0.4.0) - 2025-06-07

### Added
- Add compressed GSLC and tests by @scottstanie ([#35](https://github.com/opera-adt/opera-utils/pull/35))
- Add coordinate grid and ISCE3 orbit helpers for CSLC product by @scottstanie ([#36](https://github.com/opera-adt/opera-utils/pull/36))

---

## [0.3.2](https://github.com/opera-adt/opera-utils/compare/v0.3.1...v0.3.2) - 2025-05-29

### Added
- Add `get_orbit_arrays` to parse orbit info from GSLC by @scottstanie ([#34](https://github.com/opera-adt/opera-utils/pull/34))

---

## [0.3.1](https://github.com/opera-adt/opera-utils/compare/v0.3.0...v0.3.1) - 2025-05-29

### Added
- Add demo plots for the burst incidence matrix by @scottstanie ([#26](https://github.com/opera-adt/opera-utils/pull/26))

### Fixed
- Add check for duplicated burst_id/date pairs in `missing_data` by @scottstanie ([#25](https://github.com/opera-adt/opera-utils/pull/25))

### Changed
- In `geometry.py`, split the download from local file stitching by @scottstanie ([#32](https://github.com/opera-adt/opera-utils/pull/32))

### Removed
- Remove length check from `sort_files_by_date` by @scottstanie ([#33](https://github.com/opera-adt/opera-utils/pull/33))

---

## [0.3.0](https://github.com/opera-adt/opera-utils/compare/v0.2.1...v0.3.0) - 2025-02-12

### Added
- Add tests for `sort_files_by_date` by @scottstanie ([#27](https://github.com/opera-adt/opera-utils/pull/27))
- Add a `date_idx` option to `group_by_date` by @scottstanie ([#29](https://github.com/opera-adt/opera-utils/pull/29))


https://github.com/scottstanie/s1-reader/commit/d6d8a2f7920d7b54b85faf741a1e3fd366c6fd86

# [v0.2.1](https://github.com/opera-adt/opera-utils/compare/v0.2.0...v0.2.1) - 2024-02-06

**Fixed**
- Export `sort_files_by_date`
- Stop considering missing data options after 10,000 in `get_missing_data_options`

**Requirements**
- Remove scipy


# [v0.2.0](https://github.com/opera-adt/opera-utils/compare/v0.1.5...v0.2.0) - 2024-01-16

**Added**

- functions for ionosphere correction, used in `dolphin`
- Convenience functions for downloading OPERA CSLCs from ASF

**Fixed**
- changed `reproject_bounds` to use `rasterio's transform_bounds` instead of only warping 2 corners.


# [v0.1.5](https://github.com/opera-adt/opera-utils/compare/v0.1.4...v0.1.5) - 2023-12-20

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
