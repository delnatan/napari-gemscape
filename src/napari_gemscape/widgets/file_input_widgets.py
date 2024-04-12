from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

SUPPORTED_FORMATS = [".tiff", ".nd2", ".tif", ".ims"]


class FilepathItem(QListWidgetItem):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = Path(file_path)
        self.setText(self.file_path.name)  # Display only the filename
        self.setToolTip(str(self.file_path))  # Show full path on hover

    def set_status(self, status="untouched"):
        """set status of item: "untouched", "in progress", or "complete"

        this method needs to be called after instantiation for the display
        text to have a status indicator
        """
        self.status = status
        self.update_display_text()

    def get_status(self):
        return self.status

    def update_display_text(self):
        # default is "untouched"
        status_indicator = {
            "untouched": "•",
            "in progress": "—",
            "complete": "✓",
        }.get(self.status, "•")
        self.setText(f"{status_indicator} {self.file_path.name}")

    def get_file_path(self):
        return self.file_path


class InputFileList(QListWidget):
    # custom list widget that supports drag-n-drop & delete
    # these signals are implemented to communicate of the number of items
    # present in the list
    itemDeleted = Signal(int)
    itemAdded = Signal(int)

    def __init__(self):
        """customized ListWidget to work with files within a folder

        embed in another widget and connect events using the signals
        `itemDeleted` and `itemAdded` to `<InputFileList
        instance>._update_count`

        when selection is changed, this widget also emits `currentItemChanged`
        which can be connected to a loading function.

        """
        super().__init__()
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.InternalMove)
        # image directory
        self.folder_path = None
        self.new_status = "·"

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            current_index = self.currentRow()
            self.takeItem(current_index)
            self.itemDeleted.emit(self.count())

        if e.key() == Qt.Key_Up:
            current_index = self.currentRow()
            moved_index = max(current_index, 0)
            self.setCurrentRow(moved_index)

        if e.key() == Qt.Key_Down:
            n_items = self.count()
            current_index = self.currentRow()
            moved_index = min(current_index, n_items - 1)
            self.setCurrentRow(moved_index)

        if e.key() == Qt.Key_R:
            if self.folder_path is not None:
                self.load_folder(self.folder_path)

        super().keyPressEvent(e)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def add_item(self, filename):
        """overload parent method"""
        item = FilepathItem(filename)
        item.set_status()
        self.addItem(item)
        return item

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_dir():
                self.load_folder(path)

    def _open_file_dialog(self):
        """triggers opening file dialog"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", "~"
        )
        if folder_path:
            self.load_folder(Path(folder_path))

    def load_folder(self, folder_path):
        """populates file list from input folder"""
        self.clear()
        self.folder_path = folder_path

        for file_path in self.folder_path.iterdir():
            if file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() in SUPPORTED_FORMATS:
                _item = self.add_item(file_path)
                if file_path.with_suffix(".h5").exists():
                    # do some checks here
                    _item.set_status("complete")

            # '0' can be added to the list
            self.itemAdded.emit(self.count())

    def items(self):
        """convenient method to return the list of items in the widget"""
        return [self.item(index) for index in range(self.count())]


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication, QPushButton

    class TestApp(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Input File List Test")
            self.resize(400, 300)

            # Initialize the custom widget
            self.file_list = InputFileList()

            # Layout setup
            layout = QVBoxLayout()
            layout.addWidget(self.file_list)

            btn01 = QPushButton("toggle complete")
            layout.addWidget(btn01)

            btn02 = QPushButton("toggle in-progress")
            layout.addWidget(btn02)

            self.setLayout(layout)

            btn01.clicked.connect(self.toggle_state_complete)
            btn02.clicked.connect(self.toggle_state_in_progress)

        def toggle_state_complete(self):
            for item in self.file_list.items():
                item.set_status(status="complete")

        def toggle_state_in_progress(self):
            for item in self.file_list.items():
                item.set_status(status="in progress")

    app = QApplication(sys.argv)
    window = TestApp()
    window.show()
    sys.exit(app.exec_())
