"""
this file contains functions that carry out 'batch' processing from the
widget's list of files. Progress on the batch processing is monitored via
stdout by printing "PROGRESS:<int>" where the integer has to be a value from 0
to 100.
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import analysis
except ModuleNotFoundError:
    from . import analysis

import h5py
import numpy as np
import pandas as pd
import tifffile
from spotfitlm.utils import find_spots_in_timelapse

from napari_gemscape.core.utils import (
    gemscape_get_reader,
    save_dict_to_hdf5,
    save_parameters_to_hdf5,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process files with configurable tasks."
    )
    parser.add_argument("task", type=str, help="The task to perform.")
    parser.add_argument(
        "files", nargs="+", help="List of input files to process."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional path to JSON configuration file.",
        default=None,
    )
    return parser.parse_args()


def load_configuration(file_path):
    if file_path:
        with open(file_path, "r") as f:
            return json.load(f)
    return {}


def _collect_h5_files(file_list):
    """Return existing .h5 paths corresponding to the given file list."""
    return [p for fn in file_list if (p := Path(fn).with_suffix(".h5")).exists()]


def _for_each_analysis(hdf5_files, extract_fn):
    """Iterate over analysis types in each HDF5 file, calling extract_fn.

    extract_fn(file, analysis_type, h5_handle) -> row or None

    Returns {analysis_type: [row, ...]} collecting non-None results.
    """
    result_dict = {}
    n_files = len(hdf5_files)
    for i, file in enumerate(hdf5_files, start=1):
        print(f"PROGRESS:{int(i / n_files * 100):d}")
        sys.stdout.flush()
        try:
            with h5py.File(file, "r") as f:
                if "analyses" not in f:
                    print(f"No 'analyses' group in {file}.")
                    sys.stdout.flush()
                    continue
                for analysis_type in f["analyses"]:
                    row = extract_fn(file, analysis_type, f)
                    if row is not None:
                        result_dict.setdefault(analysis_type, []).append(row)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            sys.stdout.flush()
    return result_dict


def compile_tracks(hdf5_files):
    def extract(file, analysis_type, f):
        path = f"analyses/{analysis_type}/tracks"
        if path not in f:
            print(f"-----> Can't find {file}[{path}]")
            sys.stdout.flush()
            return None
        dataset = f[path]
        df = pd.DataFrame.from_records(dataset, columns=dataset.dtype.names)
        df.insert(0, "source_file", file.stem)
        df["particle"] = pd.factorize(df["particle"])[0] + 1
        return df

    return _for_each_analysis(hdf5_files, extract)


def compile_dataframes(hdf5_files, dataset_name="mobile ensemble"):
    def extract(file, analysis_type, f):
        path = f"analyses/{analysis_type}/MSD analysis/{dataset_name}"
        if path not in f:
            print(f"-----> Can't find {file}[{path}]")
            sys.stdout.flush()
            return None
        dataset = f[path]
        df = pd.DataFrame.from_records(dataset, columns=dataset.dtype.names)
        df.insert(0, "source_file", file.stem)
        return df

    return _for_each_analysis(hdf5_files, extract)


def compile_summaries(hdf5_files):
    def extract(file, analysis_type, f):
        analysis_grp = f[f"analyses/{analysis_type}"]
        msd_path = f"analyses/{analysis_type}/MSD analysis"
        if msd_path not in f:
            print(f"-----> Can't find {file}[{msd_path}]")
            sys.stdout.flush()
            return None
        msd_grp = f[msd_path]
        return {
            "source_file": file.stem,
            "frac_mobile": analysis_grp.attrs.get("frac_mobile"),
            "mobile_D": msd_grp.attrs.get("mobile D"),
            "mobile_D_std": msd_grp.attrs.get("mobile D std"),
            "stationary_D": msd_grp.attrs.get("stationary D"),
            "stationary_D_std": msd_grp.attrs.get("stationary D std"),
        }

    return _for_each_analysis(hdf5_files, extract)


def analyze_single_timelapse(file_in, timelapse, parameters, mask_file=None):
    p1 = parameters["spot_finding"]

    if mask_file is not None:
        mask = tifffile.imread(mask_file)
        mask_name = f'{Path(mask_file).stem.split("_")[-1]} '
    else:
        mask = None
        mask_name = ""

    spots_df = find_spots_in_timelapse(
        timelapse,
        mask=mask,
        start_frame=p1["start_frame"],
        end_frame=p1["end_frame"],
        sigma=p1["sigma"],
        significance=p1["significance"],
        boxsize=p1["boxsize"],
        itermax=p1["itermax"],
        use_filter=p1["use_filter"],
        min_sigma=p1["min_sigma"],
        max_sigma=p1["max_sigma"],
        min_amplitude=p1["min_amplitude"],
        max_amplitude=p1["max_amplitude"],
    )

    p2 = parameters["linking"]
    link_result = analysis.link_trajectory(
        points=spots_df,
        minimum_track_length=p2["minimum_track_length"],
        maximum_displacement=p2["maximum_displacement"],
        alpha_cutoff=p2["alpha_cutoff"],
        drift_corr_smooth=p2["drift_corr_smooth"],
    )

    p3 = parameters["analysis"]
    msdfitres = analysis.fit_msd(
        link_result["tracks_df"],
        dxy=p3["dxy"],
        dt=p3["dt"],
        n_pts_to_fit=p3["n_pts_to_fit"],
    )

    msd_group = {}
    for key, result in msdfitres.items():
        motion_str = key.split("_")[0]  # 'mobile' or 'stationary'
        if result["msd_ens"] is not None:
            msd_group[f"{motion_str} ensemble"] = result["msd_ens"]
        if result["msd_ta"] is not None:
            msd_group[f"{motion_str} time-averaged"] = result["msd_ta"]
        if result["D_eff"] is not None:
            msd_group[f"{motion_str} D"] = result["D_eff"]
        if result["D_eff_sd"] is not None:
            msd_group[f"{motion_str} D std"] = result["D_eff_sd"]

    group_name = f"{mask_name}analysis"
    shared_data = {
        "analyses": {
            group_name: {
                "points": spots_df,
                "tracks": link_result["tracks_df"],
                "drift": link_result["drift"],
                "frac_mobile": link_result["frac_mobile"],
                "mask_name": mask_name,
                "MSD analysis": msd_group,
            }
        }
    }

    hdf5_outpath = Path(file_in).with_suffix(".h5")
    with h5py.File(hdf5_outpath, "w") as fhd:
        save_parameters_to_hdf5(fhd, parameters)
        save_dict_to_hdf5(fhd, shared_data)

    if mask is not None:
        with h5py.File(hdf5_outpath, "a") as fhd:
            mask_group = fhd.require_group("masks")
            mask_group.create_dataset(
                mask_name, data=mask.astype(np.uint8), dtype="uint8"
            )


def process_files(task, file_list, config):
    total_files = len(file_list)

    if task == "test":
        for i, fn in enumerate(file_list, 1):
            time.sleep(0.5)
            print(f"PROGRESS:{int(i / total_files * 100)}")
            sys.stdout.flush()
            print(f"Processing {fn}")
            sys.stdout.flush()

    elif task == "analyse GEMs":
        print("analyzing GEMs -- batch mode")
        print("using parameters: ")
        print(config)
        sys.stdout.flush()

        for i, fn in enumerate(file_list, 1):
            print(f"Working on {fn}...")
            fp = Path(fn)
            mask_path = fp.parent / f"{fp.stem}_mask.tif"
            mask_filename = str(mask_path) if mask_path.exists() else None

            try:
                print(f"PROGRESS:{int(i / total_files * 100)}")
                sys.stdout.flush()
                file_reader = gemscape_get_reader(fn)
                image = file_reader(fn)
                analyze_single_timelapse(fn, image, config, mask_file=mask_filename)
                sys.stdout.flush()
            except Exception as e:
                print(f"for loop error: {str(e)}", file=sys.stderr)
                sys.stdout.flush()

    elif task == "compile tracks":
        print("compiling tracks -- batch mode")
        sys.stdout.flush()
        parentdir = Path(file_list[0]).parent
        compiled_tracks = compile_tracks(_collect_h5_files(file_list))

        for key, value in compiled_tracks.items():
            coldf = pd.concat(value, ignore_index=True)
            particle_max = coldf.groupby("source_file")["particle"].max()
            offset = particle_max.cumsum().shift(fill_value=0)
            coldf = coldf.merge(
                offset.rename("offset"), on="source_file", how="left"
            )
            coldf["particle"] += coldf["offset"]
            coldf.drop(columns=["offset"], inplace=True)
            coldf.to_csv(
                parentdir / f"{key.replace(' ', '_')}_tracks.csv", index=False
            )

    elif task == "compile MSDs":
        print("compiling MSDs -- batch mode")
        sys.stdout.flush()
        parentdir = Path(file_list[0]).parent
        h5flist = _collect_h5_files(file_list)

        for key, value in compile_dataframes(h5flist, "mobile ensemble").items():
            pd.concat(value, ignore_index=True).to_csv(
                parentdir / f"{key.replace(' ', '_')}_mobile_ensemble.csv",
                index=False,
            )

        for key, value in compile_dataframes(h5flist, "mobile time-averaged").items():
            pd.concat(value, ignore_index=True).to_csv(
                parentdir / f"{key.replace(' ', '_')}_time-averaged.csv",
                index=False,
            )

    elif task == "compile summaries":
        print("compiling summaries -- batch mode")
        sys.stdout.flush()
        parentdir = Path(file_list[0]).parent

        for key, value in compile_summaries(_collect_h5_files(file_list)).items():
            pd.DataFrame(value).to_csv(
                parentdir / f"{key.replace(' ', '_')}_summary.csv", index=False
            )


if __name__ == "__main__":
    args = parse_args()
    config = load_configuration(args.config)
    process_files(args.task, args.files, config)
