"""Module for exploring HDF5 files interactively in Jupyter."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from ._types import PathOrStr


class HDF5Explorer:
    """Class which maps an HDF5 file and allows tab-completion to explore datasets.

    Useful for interactive exploration of large HDF5 files in IPython/Jupyter.
    """

    def __init__(  # noqa: D107
        self, hdf5_filepath: PathOrStr, load_less_than: float = 1e3
    ):
        self.hdf5_filepath = hdf5_filepath
        self._hf = h5py.File(hdf5_filepath, "r")
        self._root_group = _HDF5GroupExplorer(
            self._hf["/"], load_less_than=load_less_than
        )

    def close(self):
        self._hf.close()

    def __getattr__(self, name):
        return getattr(self._root_group, name)

    def __dir__(self):
        return self._root_group.__dir__()

    def __repr__(self):
        return f"HDF5Explorer({self.hdf5_filepath})"

    def __del__(self):
        self.close()


class _HDF5GroupExplorer:
    def __init__(self, group: h5py.Group, load_less_than: float = 1e3):
        self._group = group
        self._attr_cache: dict[str, Any] = {}
        self._populate_attr_cache(load_less_than)

    @property
    def group_path(self) -> str:
        return self._group.name

    def _populate_attr_cache(self, load_less_than: float = 1e3):
        for name, item in self._group.items():
            if isinstance(item, h5py.Group):
                self._attr_cache[name] = _HDF5GroupExplorer(item)
            elif isinstance(item, h5py.Dataset):
                if item.size < load_less_than:
                    self._attr_cache[name] = item[()]
                else:
                    self._attr_cache[name] = item
            else:
                self._attr_cache[name] = item

    def __getattr__(self, name):
        if name not in self._attr_cache:
            msg = f"'{name}' not found in the group '{self.group_path}'."
            raise AttributeError(msg)
        return self._attr_cache[name]

    def __dir__(self):
        return list(self._attr_cache.keys())


def create_explorer_widget(
    hf: h5py.File,
    load_less_than: float = 1e3,
    subsample: tuple[int, int] = (10, 10),
    cmap: str = "gray",
    interpolation: str = "nearest",
):
    """Make a widget in Jupyter to explore a h5py file.

    Requires `ipywidgets` and `matplotlib` to be installed.
    Must run `%matplotlib widget` in Jupyter before calling this function.

    Examples
    --------
    >>> hf = h5py.File("file.h5", "r") # doctest: +SKIP
    >>> create_explorer_widget(hf) # doctest: +SKIP

    """
    from io import BytesIO

    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    sub_row, sub_col = subsample

    def _make_thumbnail(image):
        # Create a thumbnail of the dataset
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(
            image,
            cmap=cmap,
            interpolation=interpolation,
            vmax=np.nanpercentile(image, 99),
        )
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        # Display the thumbnail in an Image widget
        return widgets.Image(value=buf.read(), format="png")

    def _add_widgets(item, level: int = 0):
        """Recursively add widgets to the accordion widget."""
        if isinstance(item, h5py.Group):
            # Add a new accordion widget for the group
            accordion = widgets.Accordion(selected_index=None)
            for key, value in item.items():
                widget = _add_widgets(value, level + 1)
                accordion.children += (widget,)
                accordion.set_title(len(accordion.children) - 1, key)
            return accordion

        # Once we're at a leaf node, add a widget for the dataset
        elif isinstance(item, h5py.Dataset):
            attributes = [f"<b>{k}:</b> {v}" for k, v in item.attrs.items()]
            content = f"Type: {item.dtype}<br>Shape: {item.shape}<br>"
            content += "<br>".join(attributes)
            if item.size < load_less_than:
                content += f"<br>Value: {item[()]}"
            html_widget = widgets.HTML(content)

            if item.ndim != 2:
                return html_widget
            # If the dataset is a 2D array, make a thumbnail
            # Handle the real or complex the same
            data = np.abs(item[::sub_row, ::sub_col])
            image_widget = _make_thumbnail(data)
            return widgets.VBox([image_widget, html_widget])

        else:
            # Other types of items
            return widgets.HTML(f"{item}")

    # Now add everything starting at the root
    return _add_widgets(hf, 0)
