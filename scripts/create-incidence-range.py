#!/usr/bin/env python
# /// script
# dependencies = ['rasterio', 'tyro']
# ///
"""Create incidence angle and approximate slant range from line-of-sight rasters.

DISP-S1 static layers include the line-of-sight (LOS) east, north, up (ENU) unit
vectors.  From this, we can get the incidence `arccos(up)`, and an approximation of
the slant range distance based on Sentinel-1 orbit altitude.
"""

import tyro

from opera_utils.disp import create_incidence_range

if __name__ == "__main__":
    tyro.cli(create_incidence_range.run)
