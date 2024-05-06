import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import h5py
import mrc
import napari
import nd2
import numpy as np
import pandas as pd
import tifffile
from napari.layers import Image, Labels, Points, Tracks
from napari.types import LayerData
from napari_gemscape.widgets import Parameter

if TYPE_CHECKING:
    from napari_gemscape.gemscape_widget import EasyGEMsWidget

PathLike = Union[str, Path]
ReaderFunction = Callable[[Path], List[LayerData]]

FILE_FORMATS = [".tif", ".tiff", ".nd2", ".dv", ".ims"]


def get_reader(path: PathLike):
    """returns reader with layer properties
    this function is invoked when the image file is opened with 'napari'
    """

    def _reader(path):
        image_reader = gemscape_get_reader(path)
        image_data = image_reader(path)
        clo, chi = np.percentile(image_data, (0.01, 99.9))
        return [
            (
                image_data,
                {"colormap": "viridis", "contrast_limits": (clo, chi)},
            )
        ]

    return _reader


def gemscape_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """this function returns a simpler reader that returns only image data

    used internally in napari-gemscape
    """
    if isinstance(path, str):
        path = Path(path)

    if path.suffix in FILE_FORMATS:
        if path.suffix == ".nd2":
            return nd2.imread
        if path.suffix == ".dv":
            return mrc.imread
        if path.suffix == ".tif" or path.suffix == ".tiff":
            return tifffile.imread
        if path.suffix == ".ims":
            print("Imaris file detected")
            return read_imaris_timelapse
        else:
            return None


def read_imaris_timelapse(path: Path) -> LayerData:

    with h5py.File(path) as fhd:

        image_metadata = {}

        for key, value in fhd["DataSetInfo"]["Image"].attrs.items():
            decoded_val = "".join(map(lambda x: x.decode(), value))
            image_metadata[key] = decoded_val

        Nz = int(image_metadata["Z"])
        Ny = int(image_metadata["Y"])
        Nx = int(image_metadata["X"])

        em_wvlen = b"".join(
            fhd["DataSetInfo"]["Channel 0"].attrs["LSMEmissionWavelength"]
        ).decode("utf-8")

        psfinfo_str = b"".join(
            fhd["DataSetInfo"]["CustomData"].attrs[
                "PSF Settings Configuration V2"
            ]
        ).decode("utf-8")

        Nt = int(
            b"".join(
                fhd["DataSetInfo"]["CustomData"].attrs["NumberOfTimePoints"]
            )
        )

        # psfinfo contains voxel information
        psfinfo = json.loads(psfinfo_str)

        timepoints = []

        for t in range(Nt):
            _attrstr = f"TimePoint{t+1:d}"
            _timestr = b"".join(
                fhd["DataSetInfo"]["TimeInfo"].attrs[_attrstr]
            ).decode("utf-8")
            _timestamp = datetime.strptime(_timestr, "%Y-%m-%d %H:%M:%S.%f")
            timepoints.append(_timestamp)

        timedeltas = [0.0]
        timedeltas.extend(
            [
                (timepoints[i + 1] - timepoints[i]).total_seconds()
                for i in range(Nt - 1)
            ]
        )
        timedeltas = np.array(timedeltas)

        # image data
        data = np.empty((Nt, Ny, Nx), dtype=np.uint16)

        for t in range(Nt):
            data[t, :, :] = np.array(
                fhd["DataSet"]["ResolutionLevel 0"][f"TimePoint {t}"][
                    "Channel 0"
                ]["Data"]
            )[0]

    pixsize = psfinfo["ResolutionX"]
    dt = float(np.median(timedeltas))
    print(f"Time interval : {dt:.3f}")
    print(f"dxy : {pixsize:0.4f}")
    fps = 1 / dt

    return data


def save_parameters_to_hdf5(file_handle: h5py.File, parameters: dict):
    """create an HDF5 group 'parameters'
    to save the multi-step analyses parameters.

    Currently there are only 3 steps of analysis.
    - "spot_finding"
    - "linking"
    - "analysis"

    The input dictionary `parameters` needs to have these keys.  Each one of
    these will have its own HDF5 'group' with its own corresponding attributes
    which contain the parameter value.

    """
    parameters_group = file_handle.require_group("parameters")

    for step in ["spot_finding", "linking", "analysis"]:
        step_group = parameters_group.require_group(step)
        save_step_parameters(step_group, parameters[step])


def save_step_parameters(group, params):
    """Helper function to save parameters of a specific analysis step to a group."""
    for parname, par in params.items():
        group.attrs[parname] = get_parameter_value(par)


def get_parameter_value(par):
    """Return the appropriate representation of the parameter for HDF5 attributes."""
    if isinstance(par, Parameter):
        return (
            par.value.name
            if isinstance(par.value, (Image, Labels, Points, Tracks))
            else ("None" if par.value is None else par.value)
        )
    else:
        return "None" if par is None else par


def save_masks_to_hdf5(
    file_handle: h5py.File, layers: "napari.components.layerlist.LayerList"
):
    """create an HDF5 group for 'masks' dataset"""

    masks = [layer for layer in layers if isinstance(layer, Labels)]

    if len(masks) == 0:
        return

    layers_group = file_handle.require_group("masks")

    for mask in masks:
        name = mask.name
        layers_group.create_dataset(f"{name}", data=mask.data.astype(np.uint8))


def save_dict_to_hdf5(hdf_file: h5py.File, d: dict, parent_group="/"):
    """recursive function to save a dictionary into a HDF5 file group

    The dictionary may contain scalar quantities like int, float, bool, or str.
    DataFrames with columns containing string is treated like a generic Python
    object, so the column is converted into a fixed-length string of 16
    characters.

    """
    for key, value in d.items():
        if isinstance(value, dict):
            group_name = parent_group + key
            hdf_file.require_group(group_name)
            save_dict_to_hdf5(hdf_file, value, group_name + "/")
        elif isinstance(value, (int, float, str, bool)):
            hdf_file[parent_group].attrs[key] = value
        elif isinstance(value, pd.DataFrame):
            object_columns = value.columns[value.dtypes == "object"]
            if len(object_columns) > 0:
                for column in object_columns:
                    value[column] = value[column].astype("S16")
            hdf_file[parent_group + key] = value.to_records(index=False)
        elif isinstance(value, np.ndarray):
            hdf_file[parent_group + key] = value
        else:
            raise ValueError(f"Unsupported data type: {type(value)}")


def extract_parameter_values(d):
    # import here to avoid import clashes with top-level modules
    if isinstance(d, dict):
        return {
            key: extract_parameter_values(value) for key, value in d.items()
        }
    elif isinstance(d, Parameter):
        return d.value
    else:
        return d


def print_dict(d: dict, indent=0):
    for key, value in d.items():
        print("  " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print()  # Print a newline before recursing
            print_dict(value, indent + 1)
        else:
            print(value)


def save_state_to_hdf5(
    output_path: Path,
    napari_viewer: napari.Viewer,
    widget: "EasyGEMsWidget",
):  # noqa: F821
    with h5py.File(output_path, "w") as fhd:
        if widget.shared_parameters != {}:
            save_parameters_to_hdf5(fhd, widget.shared_parameters)
        save_masks_to_hdf5(fhd, napari_viewer.layers)
        if widget.shared_data["analyses"] != {}:
            save_dict_to_hdf5(fhd, widget.shared_data)


def load_dict_from_hdf5(hdf_file: h5py.File, parent_group="/") -> dict:
    """Recursive function to load a dictionary from an HDF5 file group.

    The function assumes that the HDF5 file was saved using the
    `save_dict_to_hdf5` function, which handles scalar quantities,
    DataFrames, and NumPy arrays.
    """
    d = {}
    for key, value in hdf_file[parent_group].items():
        if key == "masks":
            masks = {}
            for mask_name, mask_data in value.items():
                masks[mask_name] = np.array(mask_data)
            d[key] = masks

        if isinstance(value, h5py.Group):
            d[key] = load_dict_from_hdf5(hdf_file, parent_group + key + "/")

        elif isinstance(value, h5py.Dataset):
            if value.dtype.kind == "V":
                # V for void, indicating structured data
                # get column names from numpy.rec.recarray
                column_names = value.dtype.names
                _df = pd.DataFrame.from_records(value, columns=column_names)
                # decode the string bytes back to string
                for colname, series in _df.items():
                    if series.dtype == "object":
                        _df[colname] = _df[colname].str.decode("utf-8")
                d[key] = _df
            else:
                d[key] = np.array(value)

    for key, value in hdf_file[parent_group].attrs.items():
        d[key] = value

    return d


def load_state(
    state_dict: dict, napari_viewer: napari.Viewer, w: "EasyGEMsWidget"
):  # noqa: F821
    """loads the state for a new image"""

    if "masks" in state_dict.keys():
        masks = state_dict["masks"]
        w.shared_data["masks"] = {}  # clear current masks
        for key, value in masks.items():
            napari_viewer.add_labels(value, name=key, opacity=0.45)

    # load analysis results back to GUI
    if "analyses" in state_dict.keys():
        # restoring the 'state' is easy
        analyses = state_dict["analyses"]
        w.shared_data["analyses"] = {}  # clear current analyses
        w.shared_data["analyses"].update(analyses)
        # but then we need to load relevant data back to napari_viewer

        # go through each 'analysis'
        for key, value in analyses.items():
            mask_name = value["mask_name"]
            if mask_name != "":
                spacer = 1
            else:
                spacer = 0
            if "tracks" in value.keys():
                track_columns = ["particle", "frame", "y", "x"]
                tracks = value["tracks"]
                napari_viewer.add_tracks(
                    tracks[track_columns],
                    features=tracks,
                    name=f"{mask_name + ' ' * spacer}tracks",
                )
            if "points" in value.keys():
                napari_viewer.add_points(
                    value["points"][["frame", "y", "x"]],
                    features=value["points"],
                    name=f"{mask_name + ' ' * spacer}GEMs",
                    symbol="s",
                    edge_color="yellow",
                    face_color="transparent",
                    size=state_dict["parameters"]["spot_finding"]["boxsize"],
                )

    # parameters can't be updated via dictionary update
    # because a two-way communication is needed for each `Parameter` object
    if "parameters" in state_dict.keys():
        parameters = state_dict["parameters"]
        # we don't want to 'clear' parameters in the widget because it
        # holds 'Parameter' objects, mean to only be updated.
        # update 'shared_parameters' in widget
        for outer_key in parameters:
            for inner_key in parameters[outer_key]:
                if inner_key not in ["image", "mask", "points", "tracks"]:
                    w.shared_parameters[outer_key][inner_key].value = (
                        parameters[outer_key][inner_key]
                    )
