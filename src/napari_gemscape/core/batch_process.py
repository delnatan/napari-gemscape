"""
this file contains functions that carry out 'batch' processing from the
widget's list of files. Progress on the batch processing is monitored via
stdout by printing "PROGRESS:<int>" where the integer has to be a value from 0
to 100.

This is done by parsing the string and checking to see if it's starts with
'PROGRESS:', then taking the second element of the results after splitting it
with ':'

"""

import argparse
import json
import sys
import time
from pathlib import Path

import analysis
import h5py
import pandas as pd
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


def compile_dataframes(hdf5_files, dataset_name="mobile ensemble"):
    analysis_dict = {}

    for i, file in enumerate(hdf5_files, start=1):
        print(f"PROGRESS:{i}")
        sys.stdout.flush()

        try:
            with h5py.File(file, "r") as f:
                analyses_group = f["analyses"]

                if analyses_group is None:
                    print(f"No 'analyses' has been done for {file}.")
                    sys.stdout.flush()
                    continue

                for analysis_type in analyses_group:
                    try:
                        datagroupstr = f"analyses/{analysis_type}/MSD analysis/{dataset_name}"
                        dataset = f[datagroupstr]
                    except Exception as e:
                        print(f"Error : {e}")
                        sys.stdout.flush()
                        print(f"-----> Can't find {file}[{datagroupstr}]")
                        sys.stdout.flush()
                        continue

                    if dataset is None:
                        print(f"Can't find {file}.../{dataset_name}")
                        sys.stdout.flush()
                        continue

                    df = pd.DataFrame.from_records(
                        dataset, columns=dataset.dtype.names
                    )
                    df.insert(0, "source_file", file.stem)

                    if analysis_type not in analysis_dict:
                        analysis_dict[analysis_type] = []
                    analysis_dict[analysis_type].append(df)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    return analysis_dict


def analyze_single_timelapse(file_in, timelapse, parameters):
    # detect spots
    p1 = parameters["spot_finding"]

    spots_df = find_spots_in_timelapse(
        timelapse,
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

    # link spots
    p2 = parameters["linking"]

    link_result = analysis.link_trajectory(
        points=spots_df,
        minimum_track_length=p2["minimum_track_length"],
        maximum_displacement=p2["maximum_displacement"],
        prob_mobile_cutoff=p2["prob_mobile_cutoff"],
    )

    # analyze MSDs
    p3 = parameters["analysis"]

    msdfitres = analysis.fit_msd(
        link_result["tracks_df"],
        dxy=p3["dxy"],
        dt=p3["dt"],
        n_pts_to_fit=p3["n_pts_to_fit"],
    )

    m_D, m_D_std = msdfitres["m_D"]
    s_D, s_D_std = msdfitres["s_D"]

    analysis_data = {
        "MSD analysis": {
            "mobile time-averaged": msdfitres["m_msd_ta"],
            "stationary time-averaged": msdfitres["s_msd_ta"],
            "mobile ensemble": msdfitres["m_msd_ens"],
            "stationary ensemble": msdfitres["s_msd_ens"],
            "mobile D": m_D,
            "mobile D std": m_D_std,
            "stationary D": s_D,
            "stationary D std": s_D_std,
        }
    }

    # compile data results
    shared_data = {"analyses": {}}
    group_name = "analysis"
    shared_data["analyses"].update({group_name: {}})
    shared_data["analyses"][group_name].update({"points": spots_df})
    shared_data["analyses"][group_name].update(
        {"tracks": link_result["tracks_df"]}
    )
    # at the moment, no mask is supported for batch-mode
    shared_data["analyses"][group_name].update(
        {"frac_mobile": link_result["frac_mobile"], "mask_name": ""}
    )
    shared_data["analyses"][group_name].update(analysis_data)

    # save to HDF5 file
    hdf5_outpath = Path(file_in).with_suffix(".h5")
    with h5py.File(hdf5_outpath, "w") as fhd:
        save_parameters_to_hdf5(fhd, parameters)
        save_dict_to_hdf5(fhd, shared_data)


def process_files(task, file_list, config):
    total_files = len(file_list)

    if task == "test":
        for i, fn in enumerate(file_list, 1):
            time.sleep(0.5)
            progress = int((i / total_files) * 100)
            print(f"PROGRESS:{progress}")
            sys.stdout.flush()

            print(f"Processing {fn}")
            sys.stdout.flush()

    if task == "analyse GEMs":
        print("analyzing GEMs -- batch mode")
        sys.stdout.flush()
        print("using parameters: ")
        print(config)
        sys.stdout.flush()

        for i, fn in enumerate(file_list, 1):
            print(f"Working on {fn}...")
            sys.stdout.flush()
            try:
                progress = int((i / total_files) * 100)
                print(f"PROGRESS:{progress}")
                sys.stdout.flush()

                # read image file
                file_reader = gemscape_get_reader(fn)
                image = file_reader(fn)
                analyze_single_timelapse(fn, image, config)
                sys.stdout.flush()

            except Exception as e:
                print(f"for loop error: {str(e)}", file=sys.stderr)
                sys.stdout.flush()

    if task == "compile MSDs":
        print("compiling MSDs -- batch mode")
        sys.stdout.flush()

        parentdir = Path(file_list[0]).parent

        h5flist = []

        for i, fn in enumerate(file_list, 1):
            # convert file string to path
            fp = Path(fn)
            h5path = fp.with_suffix(".h5")

            if h5path.exists():
                h5flist.append(h5path)

        compiled_ensemble = compile_dataframes(
            h5flist, dataset_name="mobile ensemble"
        )
        compiled_timeavg = compile_dataframes(
            h5flist, dataset_name="mobile time-averaged"
        )

        for key, value in compiled_ensemble.items():
            coldf = pd.concat(value, ignore_index=True)
            coldf.to_csv(
                parentdir / f"{key.replace(' ', '_')}_mobile_ensemble.csv",
                index=False,
            )

        for key, value in compiled_timeavg.items():
            coldf = pd.concat(value, ignore_index=True)
            coldf.to_csv(
                parentdir / f"{key.replace(' ', '_')}_time-averaged.csv",
                index=False,
            )

        sys.stdout.flush()


if __name__ == "__main__":
    args = parse_args()
    config = load_configuration(args.config)
    process_files(args.task, args.files, config)
