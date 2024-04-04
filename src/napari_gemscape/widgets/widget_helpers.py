from functools import wraps
from typing import Callable, List

import napari
from magicgui.widgets import Container, ProgressBar, create_widget
from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtWidgets import QPushButton


class Parameter(QObject):
    valueChanged = Signal(object)  # Can emit any type of value

    def __init__(self, name, value, **kwargs):
        super().__init__()
        self.name = name
        self._value = value
        self.kwargs = kwargs

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        """sets a Parameter's value and emit the new value if it's changed

        Note that when the value is a napari Layer instance, the value is not
        updated because we can't set the widget's value to be a napari Layer
        """
        if hasattr(new_value, "name") and isinstance(
            new_value, napari.layers.Layer
        ):
            # processed_value = new_value.name
            # do nothing because we don't want to emit an actual Layer
            pass

        if self._value != new_value:
            self._value = new_value
            self.valueChanged.emit(new_value)  # Emit signal on value change

    def __repr__(self):
        return f"<Parameter name={self.name} value={self.value}>"


class TaskSignal(QObject):
    progress = Signal(int)  # signal for progress bar update
    finished = Signal()  # signal to indicate the task is finished
    result = Signal(object)


class FunctionTask(QRunnable):
    def __init__(self, function, parameters, napari_viewer):
        """wraps input 'function' to run on thread

        Input function must take 'progress_callback' argument to update
        the progress bar. The callback function should take an `int` argument
        with range [0,100].

        here's an example function:

        def your_function_here(parameters, progress_callback=None):
            total_steps = 100  # Example total steps
                for step in range(total_steps):
                    # Your processing logic here
                    if progress_callback:
                        progress_callback(int((step / total_steps) * 100))


        """
        super().__init__()
        self.function = function
        self.parameters = parameters
        self.viewer = napari_viewer
        self.kwargs = {}
        self.signals = TaskSignal()

    def run(self):
        """run function on thread"""

        for param in self.parameters:

            if "annotation" in param.kwargs:
                if issubclass(param.kwargs["annotation"], napari.layers.Layer):
                    if param.value is not None:
                        self.kwargs[param.name] = param.value.data
            else:
                self.kwargs[param.name] = param.value
                if "progress_callback" in self.function.__code__.co_varnames:
                    self.kwargs["progress_callback"] = (
                        self.signals.progress.emit
                    )
        try:
            result = self.function(**self.kwargs)
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()  # emit finished signal


def parameters_to_arguments(function, napari_viewer, progress_callback=None):
    """decorator function to convert list of Parameters to actual function
    parameters
    """

    @wraps(function)
    def wrapper(params_list):
        # Convert the list of Parameter objects to kwargs for func
        kwargs = {}

        for param in params_list:
            if hasattr(param, "annotation") and issubclass(
                param.annotation, napari.layers.Layer
            ):
                layer = napari_viewer.layers.get(param.value, None)
                if layer is not None:
                    kwargs[param.name] = layer.data
                else:
                    ValueError(f"Layer named {param.value} not found.")
            else:
                kwargs[param.name] = param.value

        if "progress_callback" in function.__code__.co_varnames:
            kwargs["progress_callback"] = progress_callback

        return function(**kwargs)

    return wrapper


def create_widget_from_params(
    parameters: List[Parameter],
    function: Callable,
    result_handler: Callable,
    napari_viewer: napari.Viewer,
    use_threading: bool = False,
):
    """create a widget from a list of Parameter"""
    container = Container(layout="vertical")

    for param in parameters:

        if "annotation" in param.kwargs:
            widget = create_widget(
                value=param.value,
                label=param.name,
                annotation=param.kwargs["annotation"],
            )
        else:
            widget = create_widget(
                value=param.value,
                label=param.name,
                options=param.kwargs,
            )

        # Update Parameter value when widget's value changed
        widget.changed.connect(
            lambda value, p=param: setattr(p, "value", value)
        )

        # Update widget's value when parameter's value changed
        param.valueChanged.connect(
            lambda new_value, w=widget: setattr(w, "value", new_value)
        )

        # if the widget is a napari layer, connect to the inserted/removed
        # events to update widget choices
        if widget.annotation is not None:
            if issubclass(widget.annotation, napari.layers.Layer):
                if napari_viewer is None:
                    raise ValueError(
                        "widget for napari layers data must also pass"
                        "napari.Viewer instance to `napari_viewer` argument. "
                    )

                # also monitor for name changes
                # from Talley Lambert's answer in
                # https://forum.image.sc/t/event-handling-in-napari/46539/6

                # we have to do this because layers.events do not monitor
                # for name changes. So we need to connect 'reset_choices()' to
                # layer.events.name for every new layer that's inserted

                # update layer choices when layers are added/removed
                @napari_viewer.layers.events.inserted.connect
                def _on_insert(event, widget=widget):
                    """
                    The 'widget' argument is needed to bind the variable
                    'widget' to the current widget. This is needed because
                    the variable 'widget' gets reassigned to a different
                    widget type throughout the loop. We need it to be
                    the specific widget within the 'if' statement scope
                    """
                    # get inserted layer
                    layer = event.value

                    if hasattr(widget, "reset_choices"):
                        widget.reset_choices()

                    # bind layer name changes to 'reset_choices()' also
                    layer.events.name.connect(widget.reset_choices)

                napari_viewer.layers.events.removed.connect(
                    widget.reset_choices
                )

        container.append(widget)

    # Conditionally add progress bar and run button if threading is used
    if use_threading:
        progress_bar = ProgressBar(value=0, max=100, label="Progress")
        container.append(progress_bar)

        run_button = QPushButton("Do it!")
        container.native.layout().addWidget(run_button)

        thread_pool = QThreadPool()

        def start_task():
            # disable run button to prevent multiple clicks
            run_button.setDisabled(True)

            task = FunctionTask(function, parameters, napari_viewer)
            task.signals.progress.connect(
                lambda val: setattr(progress_bar, "value", val)
            )

            task.signals.result.connect(result_handler)
            # re-enable button when process is finished
            task.signals.finished.connect(lambda: run_button.setEnabled(True))
            # start task
            thread_pool.start(task)

        run_button.clicked.connect(start_task)

    else:
        # For synchronous tasks, add a simple run button without progress bar
        run_button = QPushButton("Do it!")
        container.native.layout().addWidget(run_button)

        def run_function(result_handler):
            _func = parameters_to_arguments(function, napari_viewer)
            result = _func(parameters)
            result_handler(result)

        run_button.clicked.connect(lambda: run_function(result_handler))

    container.native.layout().addStretch()

    return container


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()

    # Adjust these parameters as needed for your actual tasks
    parameters_long = [
        Parameter("input", None, annotation=napari.layers.Labels),
        Parameter("a", 0.0),
        Parameter("b", 5),
        Parameter("filter", False),
    ]

    parameters_short = [
        Parameter("input", None, annotation=napari.layers.Image),
        Parameter("ϕ", 7.237),
        Parameter("π", 3.1425),
    ]

    # Asynchronous task with progress bar
    async_container = create_widget_from_params(
        parameters_long, long_running_task, use_threading=True
    )
    short_container = create_widget_from_params(parameters_short, quick_task)

    w01 = viewer.window.add_dock_widget(async_container, name="w01")
    w02 = viewer.window.add_dock_widget(short_container, name="w02")

    napari.run()
