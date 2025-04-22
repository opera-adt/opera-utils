from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np


def rebase_timeseries(
    raw_data: np.ndarray, reference_dates: Sequence[datetime]
) -> np.ndarray:
    """Adjust for moving reference dates to create a continuous time series.

    DISP-S1 products have a reference date which changes over time. For example,
    shortening to YYYY-MM-DD notation, the products may have (reference, secondary) dates

        (2020-01-01, 2020-01-13)
        (2020-01-01, 2020-01-25)
        ...
        (2020-01-01, 2020-06-17)
        (2020-06-17, 2020-06-29)
        ...


    This function sums up the "crossover" values (the displacement values where the
    reference date moves forward) so that the output is referenced to the first input time.

    Parameters
    ----------
    raw_data : np.ndarray
        3D array of displacement values with moving reference dates
        shape = (time, rows, cols)
    reference_dates : Sequence[datetime]
        Reference dates for each time step

    Returns
    -------
    np.ndarray
        Continuous displacement time series with consistent reference date
    """
    if len(set(reference_dates)) == 1:
        return raw_data.copy()

    shape2d = raw_data.shape[1:]
    cumulative_offset = np.zeros(shape2d, dtype=np.float32)
    last_displacement = np.zeros(shape2d, dtype=np.float32)

    # Set initial reference date
    current_reference_date = reference_dates[0]

    output = np.zeros_like(raw_data)
    # Process each time step
    for cur_ref_date, current_displacement, out_layer in zip(
        reference_dates, raw_data, output
    ):
        # Check for shift in temporal reference date
        if cur_ref_date != current_reference_date:
            # When reference date changes, accumulate the previous displacement
            cumulative_offset += last_displacement
            current_reference_date = cur_ref_date

        # Store current displacement for next iteration
        last_displacement = current_displacement.copy()

        # Add cumulative offset to get consistent reference
        out_layer[:] = current_displacement + cumulative_offset

    return output
