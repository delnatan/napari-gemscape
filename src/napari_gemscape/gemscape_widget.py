"""
This file contains the 'main' GEMscape widget for analyzing GEMs data

"""

import json
import os
from pathlib import Path
from typing import List

import dask.array as da
import h5py
import napari
import numpy as np
import pandas as pd
import tifffile
from qtpy.QtCore import QProcess
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from spotfitlm.utils import find_spots_in_timelapse

from napari_gemscape.core import analysis
from napari_gemscape.core import utils as u
from napari_gemscape.core.utils import (
    convert_numpy_types,
    extract_parameter_values,
    gemscape_get_reader,
    load_dict_from_hdf5,
    load_state,
    save_state_to_hdf5,
    viz,
)

from .widgets import (
    CodeEditor,
    FilepathItem,
    InputFileList,
    Parameter,
    StackedPlotWidget,
    create_widget_from_params,
)

__modulepath__ = Path(__file__).parent


class SharedState:
    """convenient class to allow QWidget share a set of class attributes

    This allows one access to some 'common' data that can be used across
    multiple QWidget instances.
    """

    shared_data = {"analyses": {}}
    shared_parameters = {}
    plot_widget = None

    def __init__(self):
        pass

    def add_shared_parameters(self, parameters: List[Parameter], group: str):
        """adds a list of <Parameter> object to class attribute (a dictionary)
        'shared_parameters'.

        This convenience function is intended to make it easier to add analysis
        parameter (a `Parameter` object) to its own 'group' so that it's
        easier to reference it later using the `shared_parameters` dictionary.

        """
        dict_pars = {p.name: p for p in parameters}
        self.shared_parameters[group] = dict_pars


class EasyGEMsWidget(QWidget, SharedState):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.layout = QVBoxLayout()

        self.tab_widget = QTabWidget()

        self.setup_UI()

        self.plot_widget = StackedPlotWidget()

        self.viewer.window.add_dock_widget(
            self.plot_widget, name="Plot widget (napari-gemscape)"
        )

    def setup_UI(self):
        """setup widgets for EasyGEMs GUI"""

        ########################################################################
        # INPUT FILE WIDGET                                                    #
        ########################################################################

        # input widget as separate tab widget
        inputWidget = QWidget()
        inputLayout = QVBoxLayout()

        self.fileListWidget = InputFileList()
        inputLayout.addWidget(self.fileListWidget)

        self.batchTaskChoices = QComboBox()
        self.batchTaskChoices.addItem("test")
        self.batchTaskChoices.addItem("analyse GEMs")
        self.batchTaskChoices.addItem("compile MSDs")
        self.batchTaskChoices.addItem("compile summaries")
        self.batchTaskChoices.addItem("compile tracks")

        inputLayout.addWidget(self.batchTaskChoices)

        self.batchProcessButton = QPushButton("Do it!")
        inputLayout.addWidget(self.batchProcessButton)

        self.batchProgressBar = QProgressBar()
        inputLayout.addWidget(self.batchProgressBar)

        inputWidget.setLayout(inputLayout)

        self.tab_widget.addTab(inputWidget, "input file")

        # hook up input file widget behavior
        self.fileListWidget.currentItemChanged.connect(self.load_image)

        self.batchProcessButton.clicked.connect(self.start_batch_process)

        ########################################################################
        # SPOT FINDING WIDGET                                                  #
        # - button on top to create 'mask' to match the input 'image' shape    #
        ########################################################################

        spotFindingWidget = QWidget()
        spotFindingLayout = QVBoxLayout()

        # add 'Create mask' button
        createMaskButton = QPushButton("Create mask")
        spotFindingLayout.addWidget(createMaskButton)
        saveMaskButton = QPushButton("Save mask")
        spotFindingLayout.addWidget(saveMaskButton)

        spot_finding_parameters = [
            Parameter("image", None, annotation=napari.layers.Image),
            Parameter("mask", None, annotation=napari.layers.Labels),
            Parameter("start_frame", 0, min=0),
            Parameter("end_frame", 19, min=1),
            Parameter("sigma", 1.4, min=0.6, step=0.1),
            Parameter("significance", 0.01, min=0.001, max=1.0),
            Parameter("boxsize", 11, min=5, step=2),
            Parameter("itermax", 50, min=5),
            Parameter("use_filter", False),
            Parameter("min_sigma", 1.0, step=0.1),
            Parameter("max_sigma", 2.2, step=0.1),
            Parameter("min_amplitude", 10.0, step=50, max=65535),
            Parameter("max_amplitude", 500, step=50, max=65535),
        ]

        # dynamically generate widget group from parameters
        self.coreSpotFindWidget = create_widget_from_params(
            spot_finding_parameters,
            find_spots_in_timelapse,
            self.handle_spot_finding_result,
            napari_viewer=self.viewer,
            use_threading=True,
        )

        # keep reference to each Parameter object for 'spot_finding'
        self.add_shared_parameters(spot_finding_parameters, "spot_finding")

        # add 'spot-finding' widget
        spotFindingLayout.addWidget(self.coreSpotFindWidget.native)

        spotFindingLayout.addStretch()

        spotFindingWidget.setLayout(spotFindingLayout)

        # connect button click to create mask
        createMaskButton.clicked.connect(self.add_mask)
        # connect button click to save mask
        saveMaskButton.clicked.connect(self.save_mask)

        self.tab_widget.addTab(spotFindingWidget, "Find spots")

        ########################################################################
        # ANALYSIS WIDGET                                                      #
        ########################################################################
        analysisWidget = QWidget()
        analysisLayout = QVBoxLayout()

        coordLinkingGroup = QGroupBox("Coordinate linking")
        coordLinkingLayout = QVBoxLayout()

        # add track linking
        coord_linking_parameters = [
            Parameter("points", None, annotation=napari.layers.Points),
            Parameter("minimum_track_length", 3, min=2),
            Parameter("maximum_displacement", 4.2, min=0.1, step=1.0),
            Parameter("alpha_cutoff", 0.05, min=0.0, max=1.0, step=0.01),
            Parameter("drift_corr_smooth", -1, min=-1, step=1),
        ]

        # keep references to linking parameters
        self.add_shared_parameters(coord_linking_parameters, "linking")

        self.coordLinkingWidget = create_widget_from_params(
            coord_linking_parameters,
            analysis.link_trajectory,
            self.handle_trajectory_linking_result,
            napari_viewer=self.viewer,
            use_threading=False,
        )

        coordLinkingLayout.addWidget(self.coordLinkingWidget.native)
        coordLinkingGroup.setLayout(coordLinkingLayout)

        analysisLayout.addWidget(coordLinkingGroup)

        msdfitGroup = QGroupBox("MSD fitting")
        msdfitLayout = QVBoxLayout()

        analysis_parameters = [
            Parameter("tracks", None, annotation=napari.layers.Tracks),
            Parameter("dxy", 0.065, min=0.01, step=0.001),
            Parameter("dt", 0.010, min=0.001, step=0.005),
            Parameter("n_pts_to_fit", 3, min=2),
            Parameter("separate_immobile", True),
        ]

        self.add_shared_parameters(analysis_parameters, "analysis")

        self.MSDanalysisWidget = create_widget_from_params(
            analysis_parameters,
            analysis.fit_msd,
            self.handle_msd_fit_result,
            self.viewer,
            use_threading=False,
        )

        msdfitLayout.addWidget(self.MSDanalysisWidget.native)
        msdfitGroup.setLayout(msdfitLayout)

        analysisLayout.addWidget(msdfitGroup)

        saveStateButton = QPushButton("Save state")
        saveStateButton.clicked.connect(self.save_state)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)

        analysisLayout.addWidget(divider)
        analysisLayout.addWidget(saveStateButton)

        analysisLayout.addStretch()
        analysisWidget.setLayout(analysisLayout)

        self.tab_widget.addTab(analysisWidget, "Analysis")

        ########################################################################
        # SCRIPTING WIDGET                                                     #
        # - uses 'exec()' with local variables providing access to widget      #
        #   attributes                                                         #
        # The following variables are available in the namespace :             #
        # - 'np' for NumPy                                                     #
        # - 'plt' for the plot widget. Calling plt.add_plot(name) returns a    #
        #   matplotlib 'Axes' object.                                          #
        # - 'viewer' for napari 'Viewer' object.                               #
        # - 'w' for the Easy GEMs widget, so its class attributes are          #
        #   accessible                                                         #
        ########################################################################
        scriptWidget = QWidget()
        scriptLayout = QVBoxLayout()
        self.script_widget = CodeEditor()

        runScriptButton = QPushButton("Execute!")
        scriptLayout.addWidget(self.script_widget)
        scriptLayout.addWidget(runScriptButton)
        scriptWidget.setLayout(scriptLayout)
        self.tab_widget.addTab(scriptWidget, "Xscript")
        runScriptButton.clicked.connect(self.execute_script)

        # finally add the tab to the main widget layout
        self.layout.addWidget(self.tab_widget)
        self.setLayout(self.layout)

    def handle_spot_finding_result(self, result):
        if result is None:
            return

        coords = result[["frame", "y", "x"]]
        mask = self.shared_parameters["spot_finding"]["mask"].value

        if mask is None:
            mask_name = ""
            spacer = 0
        else:
            spacer = 1
            mask_name = mask.name

        points_layer_name = f"{mask_name + ' ' * spacer}GEMs"
        group_name = f"{mask_name + ' ' * spacer}analysis"

        # if 'analysis' group is not in 'analyses', create it for this
        # particular 'mask'
        if group_name not in self.shared_data["analyses"].keys():
            self.shared_data["analyses"].update({group_name: {}})

        self.shared_data["analyses"][group_name].update(
            {"mask_name": mask_name}
        )

        # add 'points' to analysis group
        self.shared_data["analyses"][group_name].update({"points": result})

        if points_layer_name in self.viewer.layers:
            # replace data if already there
            self.viewer.layers[points_layer_name].data = coords
            self.viewer.layers[points_layer_name].features = result
            self.viewer.layers[
                points_layer_name
            ].size = self.shared_parameters["spot_finding"]["boxsize"].value
        else:
            self.viewer.add_points(
                coords,
                symbol="s",
                size=self.shared_parameters["spot_finding"]["boxsize"].value,
                face_color="transparent",
                border_color="yellow",
                name=points_layer_name,
                features=result,
            )

    def handle_trajectory_linking_result(self, result):
        """function to handle output 'result' from trajectory linking

        The result (output) is return from `analysis.link_trajectory()`
        It's a dictionary containing:
         - 'frac_mobile': <float>, fraction of 'mobile' particles
         - 'tracks_df' : <DataFrame>

        """
        points_input = self.shared_parameters["linking"]["points"]

        if points_input.value is None:
            return

        layer_name = points_input.value.name

        # extract mask name (if any)
        layer_str = layer_name.split(" ")
        if len(layer_str) > 1:
            mask_name = layer_str[0]
            spacer = 1
        else:
            mask_name = ""
            spacer = 0

        track_layer_name = f"{mask_name + ' ' * spacer}tracks"
        group_name = f"{mask_name + ' ' * spacer}analysis"

        tracks = result["tracks_df"]
        drift = result["drift"]

        # update "results" in shared data
        # form a dictionary to represent an 'analysis' group
        if group_name not in self.shared_data["analyses"].keys():
            self.shared_data["analyses"].update({group_name: {}})

        # update mask name (in case it had been renamed)
        self.shared_data["analyses"][group_name].update(
            {"mask_name": mask_name}
        )
        self.shared_data["analyses"][group_name].update(
            {"frac_mobile": result["frac_mobile"]}
        )

        self.shared_data["analyses"][group_name].update(
            {"tracks": tracks, "drift": drift}
        )

        napari_track_columns = ["particle", "frame", "y", "x"]
        tracks_df = tracks[napari_track_columns]

        if track_layer_name in self.viewer.layers:
            # replace data if already there
            self.viewer.layers[track_layer_name].data = tracks_df
            self.viewer.layers[track_layer_name].features = tracks
        else:
            self.viewer.add_tracks(
                tracks_df,
                name=track_layer_name,
                tail_length=5,
                features=tracks,
            )

        # overlay mobile/immobile
        mobile_mask = tracks["motion"] == "mobile"
        mobile_tracks = tracks[mobile_mask]
        static_tracks = tracks[~mobile_mask]
        mobile_track_layer_name = f"{mask_name + ' ' * spacer} mobile"
        static_track_layer_name = f"{mask_name + ' ' * spacer} static"

        if mobile_track_layer_name in self.viewer.layers:
            self.viewer.layers[mobile_track_layer_name].data = mobile_tracks[
                ["frame", "y", "x"]
            ]
            self.viewer.layers[
                mobile_track_layer_name
            ].features = mobile_tracks
        else:
            self.viewer.add_points(
                mobile_tracks[["frame", "y", "x"]],
                symbol="o",
                face_color="transparent",
                border_color="#00aaff",
                size=11,
                name=mobile_track_layer_name,
                features=mobile_tracks,
            )

        if static_track_layer_name in self.viewer.layers:
            self.viewer.layers[static_track_layer_name].data = static_tracks[
                ["frame", "y", "x"]
            ]
            self.viewer.layers[
                static_track_layer_name
            ].features = static_tracks
        else:
            self.viewer.add_points(
                static_tracks[["frame", "y", "x"]],
                symbol="x",
                face_color="transparent",
                border_color="#ff55ff",
                size=11,
                name=static_track_layer_name,
                features=static_tracks,
            )

    def handle_msd_fit_result(self, result):
        """handle the conventional MSD fitting result

        This generates two plots (if 'separate_immobile' is checked/True):
        - 'mobile' ensemble MSD plot
        - 'stationary' ensemble MSD plot

        The plots should go into a separate plotting widget (a matplotlib
        `Axes` object returned by `StackedPlotWidget.add_subplot()`)

        """
        # get input track name
        track_input = self.shared_parameters["analysis"]["tracks"]

        if track_input.value is None:
            return

        track_name = track_input.value.name

        # extract mask name from track name
        layer_str = track_name.split(" ")
        if len(layer_str) > 1:
            mask_name = layer_str[0]
            spacer = 1
        else:
            mask_name = ""
            spacer = 0

        analysis_group_name = f"{mask_name + ' ' * spacer}analysis"

        analysis_data = {"MSD analysis": result}

        self.shared_data["analyses"][analysis_group_name].update(analysis_data)

        npts = self.shared_parameters["analysis"]["n_pts_to_fit"].value

        for key, msdres in result.items():
            motion_str = key.split("_")[0]
            ax = self.plot_widget.add_subplot(
                name=f"{mask_name + ' ' * spacer}MSD ({motion_str})"
            )
            viz.plot_ensemble_MSD(
                msdres["msd_ens"],
                ax,
                msdres["coefs"],
                msdres["D_eff"],
                msdres["D_eff_sd"],
                npts,
            )

    def execute_script(self):
        contents = self.script_widget.toPlainText()
        # execute 'code' given some local variable context
        exec(
            contents,
            {
                "plt": self.plot_widget,
                "np": np,
                "viewer": self.viewer,
                "pd": pd,
                "w": self,
                "u": u,
            },
        )

    def save_state(self):
        current_item = self.fileListWidget.currentItem()

        if current_item is None:
            return

        current_file = current_item.file_path
        outfn = current_file.with_suffix(".h5")
        save_state_to_hdf5(outfn, self.viewer, self)

    def load_image(self, item: FilepathItem):
        if item is None:
            return
        filepath = item.get_file_path()
        filestatus = item.get_status()
        file_reader = gemscape_get_reader(filepath)
        image = file_reader(filepath)

        # get contrast limits
        clo, chi = np.percentile(image.ravel(), (0.01, 99.9))

        if isinstance(clo, da.Array):
            clo = clo.compute()
        if isinstance(chi, da.Array):
            chi = chi.compute()

        self.viewer.layers.clear()
        self.viewer.add_image(
            image,
            name=f"{filepath.name}",
            colormap="gray",
            contrast_limits=(clo, chi),
        )

        # park at t=0
        self.viewer.dims.current_step = (0, 0, 0)

        if filestatus == "complete":
            state_file = filepath.with_suffix(".h5")
            with h5py.File(state_file, "r") as fhd:
                state_dict = load_dict_from_hdf5(fhd)
            load_state(state_dict, self.viewer, self)
        elif filestatus == "in progress":
            mask_file = filepath.parent / f"{filepath.stem}_mask.tif"
            try:
                mask = tifffile.imread(mask_file)
                self.viewer.add_labels(mask, name="mask")
            except Exception as e:
                print(f"Loading mask error: {e}")
                raise
        else:
            # erase current state
            self.shared_data = {"analyses": {}}
            # clear previous plots
            self.plot_widget.clear()

    def add_mask(self):
        _image = self.shared_parameters["spot_finding"]["image"].value
        if _image:
            image_name = str(_image)
            image_shape = self.viewer.layers[image_name].data.shape
            new_mask = np.zeros(image_shape[-2:], dtype=np.uint8)
            self.viewer.add_labels(new_mask, name="mask")

    def save_mask(self):
        _mask_layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

        if len(_mask_layers) > 0:
            # get root folder
            outpath = self.fileListWidget.folder_path
            current_item = self.fileListWidget.currentItem()

            for label_name in _mask_layers:
                mask_data = self.viewer.layers[label_name].data
                current_path = current_item.file_path
                outfn = outpath / f"{current_path.stem}_{label_name}.tif"
                tifffile.imwrite(outfn, mask_data, compression="lzw")

    def start_batch_process(self):
        task = self.batchTaskChoices.currentText()

        if task == "analyse GEMs":
            # skip ones that are already done
            flist = [
                str(item.file_path.resolve())
                for item in self.fileListWidget.items()
                if item.get_status() not in ["skip", "complete"]
            ]
        elif task in ["compile MSDs", "compile summaries"]:
            # only compile MSDs from completed files
            flist = [
                str(item.file_path.resolve())
                for item in self.fileListWidget.items()
                if item.status == "complete"
            ]
        elif task == "test":
            flist = [
                str(item.file_path.resolve())
                for item in self.fileListWidget.items()
                if item.get_status() not in ["skip", "complete"]
            ]
        else:
            # otherwise just give the entire list (the checking should
            # be done by the batch processing function)
            flist = [
                str(item.file_path.resolve())
                for item in self.fileListWidget.items()
            ]

        # debug
        print(f"task: {task}")
        for f in flist:
            print(f)

        # get analysis parameters
        pars = self.shared_parameters

        # extract parameter values from 'shared_parameters' (set by GUI)
        batch_parameters = extract_parameter_values(pars)

        # write parameters to batch directory
        batch_directory = self.fileListWidget.folder_path

        with open(batch_directory / "analysis_parameters.json", "w") as f:
            json.dump(
                batch_parameters, f, indent=4, default=convert_numpy_types
            )

        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_batch_stdout)
        self.process.readyReadStandardError.connect(self.handle_batch_stderr)
        self.process.started.connect(self.batch_process_started)
        self.process.finished.connect(self.batch_processing_finished)

        # execute batch processing
        script_path = os.path.join(__modulepath__, "core/batch_process.py")
        config_path = batch_directory / "analysis_parameters.json"

        exec_str = [script_path, task] + flist + ["--config", str(config_path)]

        self.process.start("python", exec_str)

        # wait 10 seconds for timeout
        started = self.process.waitForStarted(10_000)

        if not started:
            print("Batch process failed to start")

    def stop_batch_process(self):
        self.process.kill()

    def batch_process_started(self):
        self.batchProcessButton.setText("Stop it!")
        self.batchProcessButton.disconnect()
        self.batchProcessButton.clicked.connect(self.stop_batch_process)

    def handle_batch_stdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        lines = data.strip().split("\n")
        for line in lines:
            if line.startswith("PROGRESS:"):
                progress = int(line.split(":")[1])
                self.batchProgressBar.setValue(progress)
            else:
                print(line)

    def batch_processing_finished(self):
        self.batchProgressBar.setValue(100)
        self.batchProcessButton.setText("Do it!")
        self.batchProcessButton.disconnect()
        self.batchProcessButton.clicked.connect(self.start_batch_process)

    def handle_batch_stderr(self):
        error_data = self.process.readAllStandardError().data().decode()
        print("Batch script error:")
        print(error_data)


class SimpleMovieRecorderWidget(QWidget):
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer


if __name__ == "__main__":
    # run this script by going into root 'easy_gems'
    # then 'python -m easy_gems.easy_gems'

    # sys.path.insert(
    #     0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # )

    viewer = napari.Viewer()

    myWidget = EasyGEMsWidget(viewer)
    plot_widget = StackedPlotWidget()
    # assign plot_widget to 'myWidget'
    myWidget.plot_widget = plot_widget
    viewer.window.add_plugin_dock_widget(myWidget, name="Easy GEMs")
    viewer.window.add_plugin_dock_widget(
        plot_widget, name="Plots", tabify=True
    )

    napari.run()
