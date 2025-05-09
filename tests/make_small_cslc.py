import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

if __name__ == "__main__":
    # List of all datasets to zero out
    datasets_to_zero = [
        "/data/VV",
        "/data/azimuth_carrier_phase",
        "/data/flattening_phase",
        "/metadata/processing_information/timing_corrections/azimuth_fm_rate_mismatch",
        "/metadata/processing_information/timing_corrections/azimuth_solid_earth_tides",
        "/metadata/processing_information/timing_corrections/bistatic_delay",
        "/metadata/processing_information/timing_corrections/geometry_steering_doppler",
        "/metadata/processing_information/timing_corrections/los_ionospheric_delay",
        "/metadata/processing_information/timing_corrections/los_solid_earth_tides",
        "/metadata/calibration_information/dn",
        "/metadata/calibration_information/gamma",
        "/metadata/calibration_information/sigma_naught",
        "/metadata/calibration_information/x_coordinates",
        "/metadata/calibration_information/y_coordinates",
        "/metadata/noise_information/thermal_noise_lut",
    ]
    # Datasets that need special handling
    data_datasets = [
        "/data/VV",
        "/data/azimuth_carrier_phase",
        "/data/flattening_phase",
    ]

    # Modify the file
    fn = sys.argv[1]
    out = fn.replace("*.h5", "_compressed.h5")
    if not Path(fn).exists():
        msg = f"{fn} does not exist"
        raise ValueError(msg)

    with h5py.File(fn, "a") as hf:
        # Store x and y coordinates
        x_ds = hf["/data/x_coordinates"]
        y_ds = hf["/data/y_coordinates"]

        for dn in datasets_to_zero:
            if dn in hf:
                shape = hf[dn].shape
                dtype = hf[dn].dtype
                del hf[dn]
                hf[dn] = np.zeros(shape, dtype=dtype)

                # Add grid_mapping attribute to /data datasets
                if dn in data_datasets:
                    ds = hf[dn]
                    ds.attrs["grid_mapping"] = np.bytes_("projection")
                    ds.dims[1].attach_scale(x_ds)
                    ds.dims[0].attach_scale(y_ds)

    # Prepare h5repack command
    repack_command = ["h5repack"]
    for dataset in datasets_to_zero:
        repack_command.extend(["-f", f"{dataset}:SHUF", "-f", f"{dataset}:GZIP=5"])
    repack_command.extend([fn, out])

    # Run h5repack
    subprocess.run(repack_command, check=True)

    print("File processing and compression completed.")
