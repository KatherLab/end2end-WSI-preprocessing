#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog
from pathlib import Path
import argparse


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WSI Normalization")
        self.setGeometry(100, 100, 500, 300)

        # Create input fields
        self.output_path_label = QLabel("Output Path", self)
        self.output_path_label.move(50, 50)
        self.output_path_input = QLineEdit(self)
        self.output_path_input.move(150, 50)

        self.wsi_dir_label = QLabel("WSI Directory", self)
        self.wsi_dir_label.move(50, 90)
        self.wsi_dir_input = QLineEdit(self)
        self.wsi_dir_input.move(150, 90)

        self.model_label = QLabel("Model Path", self)
        self.model_label.move(50, 130)
        self.model_input = QLineEdit(self)
        self.model_input.move(150, 130)

        self.cache_dir_label = QLabel("Cache Directory", self)
        self.cache_dir_label.move(50, 170)
        self.cache_dir_input = QLineEdit(self)
        self.cache_dir_input.move(150, 170)

        # Create browse buttons
        self.output_path_button = QPushButton("Browse", self)
        self.output_path_button.move(350, 50)
        self.output_path_button.clicked.connect(self.browse_output_path)

        self.wsi_dir_button = QPushButton("Browse", self)
        self.wsi_dir_button.move(350, 90)
        self.wsi_dir_button.clicked.connect(self.browse_wsi_dir)

        self.model_button = QPushButton("Browse", self)
        self.model_button.move(350, 130)
        self.model_button.clicked.connect(self.browse_model)

        self.cache_dir_button = QPushButton("Browse", self)
        self.cache_dir_button.move(350, 170)
        self.cache_dir_button.clicked.connect(self.browse_cache_dir)

        # Create run button
        self.run_button = QPushButton("Run", self)
        self.run_button.move(200, 220)
        self.run_button.clicked.connect(self.run_script)

    def browse_output_path(self):
        output_path = QFileDialog.getExistingDirectory(self, "Select Output Path")
        if output_path:
            self.output_path_input.setText(output_path)

    def browse_wsi_dir(self):
        wsi_dir = QFileDialog.getExistingDirectory(self, "Select WSI Directory")
        if wsi_dir:
            self.wsi_dir_input.setText(wsi_dir)

    def browse_model(self):
        model = QFileDialog.getOpenFileName(self, "Select Model Path")
        if model:
            self.model_input.setText(model[0])

    def browse_cache_dir(self):
        cache_dir = QFileDialog.getExistingDirectory(self, "Select Cache Directory")
        if cache_dir:
            self.cache_dir_input.setText(cache_dir)

        def run_script(self):
            # Get input values
            output_path = Path(self.output_path_input.text())
            wsi_dir = Path(self.wsi_dir_input.text())
            model = Path(self.model_input.text())
            cache_dir = Path(self.cache_dir_input.text()) if self.cache_dir_input.text() else None

            # Create argparse arguments
            parser = argparse.ArgumentParser(description='Normalize WSI directly.')
            parser.add_argument('-o', '--output-path', type=Path, required=True,
                                help='Path to save results to.')
            parser.add_argument('--wsi-dir', metavar='DIR', type=Path, required=True,
                                help='Path of where the whole-slide images are.')
            parser.add_argument('-m', '--model', metavar='DIR', type=Path, required=True,
                                help='Path of where the whole-slide images are.')
            parser.add_argument('--cache-dir', type=Path, default=None,
                                help='Directory to cache extracted features etc. in.')
            # Parse arguments
            args = parser.parse_args(
                ['--output-path', str(output_path), '--wsi-dir', str(wsi_dir), '--model', str(model)])
            if cache_dir:
                args.cache_dir = str(cache_dir)

            # Run script with arguments
            # TODO: call the script here with the argparse arguments
if __name__ == 'main':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())