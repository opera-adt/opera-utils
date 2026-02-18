"""Example: Download NISAR GSLC data and create a multi-looked interferogram.

This script demonstrates:
1. Searching/downloading NISAR GSLC data by bounding box
2. Loading the HH polarization data
3. Creating a multi-looked interferogram from two acquisitions

Example files (frame 8, orbit 172, ascending):
- NISAR_L2_PR_GSLC_006_172_A_008_..._20251204T024618_...
- NISAR_L2_PR_GSLC_005_172_A_008_..._20251122T024618_...
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from opera_utils.nisar import run_download, search

# Area of interest (west, south, east, north)
# This bbox is in Ethiopia, covered by NISAR orbit 172, frame 8, ascending
BBOX = (40.62, 13.56, 40.72, 13.64)


def multilook(arr: np.ndarray, looks_row: int, looks_col: int) -> np.ndarray:
    """Apply spatial multilooking (averaging) to reduce noise."""
    nrows, ncols = arr.shape
    new_rows = nrows // looks_row
    new_cols = ncols // looks_col
    # Trim to exact multiple
    arr = arr[: new_rows * looks_row, : new_cols * looks_col]
    # Reshape and average
    return arr.reshape(new_rows, looks_row, new_cols, looks_col).mean(axis=(1, 3))


def load_gslc_hh(filepath: Path, frequency: str = "A") -> np.ndarray:
    """Load HH polarization data from a GSLC HDF5 file."""
    with h5py.File(filepath, "r") as f:
        dset_path = f"/science/LSAR/GSLC/grids/frequency{frequency}/HH"
        return f[dset_path][:]


def get_pixel_spacing(filepath: Path, frequency: str = "A") -> tuple[float, float]:
    """Get the x and y pixel spacing from a GSLC file."""
    with h5py.File(filepath, "r") as f:
        freq_path = f"/science/LSAR/GSLC/grids/frequency{frequency}"
        x_spacing = abs(float(f[freq_path]["xCoordinateSpacing"][()]))
        y_spacing = abs(float(f[freq_path]["yCoordinateSpacing"][()]))
    return x_spacing, y_spacing


def calculate_square_looks(
    x_spacing: float, y_spacing: float, target_resolution: float = 100.0
) -> tuple[int, int]:
    """Calculate looks that produce approximately square ground pixels.

    Parameters
    ----------
    x_spacing : float
        Pixel spacing in x (range) direction in meters.
    y_spacing : float
        Pixel spacing in y (azimuth) direction in meters.
    target_resolution : float
        Target ground resolution in meters. Default is 100m.

    Returns
    -------
    tuple[int, int]
        Number of looks in (row, col) directions.

    """
    looks_col = max(1, round(target_resolution / x_spacing))
    looks_row = max(1, round(target_resolution / y_spacing))
    return looks_row, looks_col


def main():
    """Download GSLC data and create a multi-looked interferogram."""
    # Step 1: Search for available products
    # Just bbox is enough - CMR finds all products that intersect
    print("Searching for NISAR GSLC products...")
    products = search(bbox=BBOX)
    print(f"Found {len(products)} products:")
    for p in products:
        print(f"  - {Path(p.filename).name}")

    if len(products) < 2:
        print("Need at least 2 products to form an interferogram.")
        print("Downloading with bbox subsetting...")

    # Step 2: Download and subset to bbox
    output_dir = Path("./gslc_subsets")
    downloaded = run_download(
        bbox=BBOX,
        polarizations=["HH"],
        output_dir=output_dir,
        num_workers=1,  # Use 1 worker for clearer progress output
    )

    if len(downloaded) < 2:
        print(f"Only {len(downloaded)} files downloaded. Need 2 for interferogram.")
        return

    # Sort by date
    downloaded = sorted(downloaded)
    print(f"\nDownloaded {len(downloaded)} files:")
    for f in downloaded:
        print(f"  - {f.name}")

    # Step 3: Load the SLC data
    print("\nLoading HH data from first two acquisitions...")
    slc1 = load_gslc_hh(downloaded[0])
    slc2 = load_gslc_hh(downloaded[1])
    print(f"  SLC1 shape: {slc1.shape}")
    print(f"  SLC2 shape: {slc2.shape}")

    # Step 4: Form interferogram (conjugate multiply)
    print("\nForming interferogram...")
    ifg = slc1 * np.conj(slc2)

    # Step 5: Multilook to reduce noise and get square ground pixels
    x_spacing, y_spacing = get_pixel_spacing(downloaded[0])
    print(f"  Pixel spacing: x={x_spacing:.1f}m, y={y_spacing:.1f}m")
    looks_row, looks_col = calculate_square_looks(
        x_spacing, y_spacing, target_resolution=60
    )
    print(f"Multilooking with {looks_row}x{looks_col} looks for ~60m resolution...")
    ifg_ml = multilook(ifg, looks_row, looks_col)
    print(f"  Multilooked shape: {ifg_ml.shape}")

    # Step 6: Visualize
    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Amplitude of first SLC
    amp1 = np.abs(multilook(slc1, looks_row, looks_col))
    vmax = np.nanpercentile(amp1, 98)
    axes[0].imshow(amp1, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title(f"Amplitude: {downloaded[0].name[:40]}...")

    # Interferometric phase
    phase = np.angle(ifg_ml)
    axes[1].imshow(phase, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Interferometric Phase")

    # Coherence (normalized cross-correlation magnitude)
    coh = np.abs(ifg_ml) / (
        np.sqrt(multilook(np.abs(slc1) ** 2, looks_row, looks_col))
        * np.sqrt(multilook(np.abs(slc2) ** 2, looks_row, looks_col))
    )
    axes[2].imshow(coh, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Coherence")

    for ax in axes:
        ax.set_xlabel("Range")
        ax.set_ylabel("Azimuth")

    plt.tight_layout()
    plt.savefig("nisar_interferogram.png", dpi=150)
    print("\nSaved figure to nisar_interferogram.png")
    plt.show()


if __name__ == "__main__":
    main()
