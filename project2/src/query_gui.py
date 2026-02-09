#!/usr/bin/env python3
"""
CS5330 CBIR - Final Fixed GUI
- Fixed fullpath parsing
- CSV persistence across task switches
- All features working
"""

import re
import sys
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from PyQt6.QtCore import Qt, QSize, QStringListModel, QTimer, QSettings
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSpinBox,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QComboBox,
    QLineEdit,
    QGroupBox,
    QMessageBox,
    QScrollArea,
    QTextEdit,
    QCompleter,
    QProgressDialog,
)

BUILD_CANDIDATES = [
    Path("build"),
    Path("cmake-build-debug"),
    Path("cmake-build-release"),
]

TASK_CONFIG = {
    1: ("query_db", True, 147, True),
    2: ("query_db", True, 256, True),
    3: ("query_db", True, 512, True),
    4: ("query_db", True, 290, True),
    5: ("query_task5", True, 512, False),
    7: ("query_task7_grass", True, 512, False),
}


@dataclass
class Match:
    rank: int
    filename: str
    dist: float
    fullpath: str


# Regex patterns to parse output lines
LINE_PATTERNS = [
    # Match "1) pic.jpg dist=0.5 fullpath=/path/to/pic.jpg"
    re.compile(
        r"^\s*(\d+)\)\s+(\S+)\s+.*?dist\s*=\s*([0-9.eE+-]+)\s+fullpath\s*=\s*(.+)$",
        re.IGNORECASE,
    ),
    # Match "1) pic.jpg dist=0.5" (no fullpath)
    re.compile(
        r"^\s*(\d+)\)\s+(\S+)\s+.*?dist\s*=\s*([0-9.eE+-]+)",
        re.IGNORECASE,
    ),
    # Match "1. pic.jpg (distance: 0.5)"
    re.compile(
        r"^\s*(\d+)[\).]\s+(\S+)\s+.*?(?:distance|dist)\s*[:\s=]\s*([0-9.eE+-]+)",
        re.IGNORECASE,
    ),
]


def parse_matches(text: str, image_dir: Path) -> List[Match]:
    matches = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("Top") or line.startswith("Task"):
            continue

        for i, pattern in enumerate(LINE_PATTERNS):
            m = pattern.match(line)
            if m:
                rank = int(m.group(1))
                fname = m.group(2)
                dist = float(m.group(3))

                # Check if we captured fullpath
                if len(m.groups()) >= 4 and m.group(4):
                    fullpath = m.group(4).strip()
                else:
                    # Construct it ourselves
                    fullpath = str(image_dir / fname)

                matches.append(Match(rank, fname, dist, fullpath))
                break
    return matches


def find_build_dir() -> Optional[Path]:
    for base in [Path.cwd(), Path.cwd().parent]:
        for cand in BUILD_CANDIDATES:
            p = base / cand
            if p.exists() and p.is_dir():
                return p
    return None


def find_exe(build_dir: Path, exe_name: str) -> Optional[Path]:
    for p in [
        build_dir / exe_name,
        build_dir / "Debug" / exe_name,
        build_dir / "Release" / exe_name,
    ]:
        if p.exists():
            return p
    return None


def list_images(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm"}
    files = [p for p in image_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    files.sort(key=lambda x: x.name)
    return files


def load_pixmap(path: Path, max_w: int, max_h: int) -> QPixmap:
    pm = QPixmap(str(path))
    if pm.isNull():
        print(f"[WARN] Cannot load: {path}")
        return QPixmap()
    return pm.scaled(
        max_w,
        max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def validate_csv(csv_path: Path, expected_dims: int) -> Tuple[bool, str, int]:
    try:
        with open(csv_path, "r") as f:
            first_line = f.readline().strip()
            if not first_line:
                return False, "Empty CSV", 0
            parts = first_line.split(",")
            actual_dims = len(parts) - 1
            if actual_dims != expected_dims:
                return (
                    False,
                    f"Expected {expected_dims}D, got {actual_dims}D",
                    actual_dims,
                )
            return True, f"✓ Valid ({actual_dims} features)", actual_dims
    except Exception as e:
        return False, f"Error: {str(e)}", 0


def build_features(
    build_dir: Path, image_dir: Path, task_id: int
) -> Tuple[bool, str, Optional[Path]]:
    build_db = find_exe(build_dir, "build_db")
    if not build_db:
        return False, "build_db not found", None
    output_csv = image_dir.parent / f"features_task{task_id}.csv"
    cmd = [str(build_db), str(image_dir), str(output_csv), str(task_id)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            return False, f"Failed:\n{proc.stderr}", None
        return True, f"✓ Built: {output_csv.name}", output_csv
    except Exception as e:
        return False, str(e), None


def run_query(
    build_dir: Path,
    task_id: int,
    target: Path,
    image_dir: Path,
    csv_path: Path,
    topk: int,
    show_bottom: bool = False,
) -> Tuple[str, List[Match]]:
    exe_name, _, _, _ = TASK_CONFIG[task_id]
    exe = find_exe(build_dir, exe_name)
    if not exe:
        raise RuntimeError(f"{exe_name} not found")

    if task_id in [1, 2, 3, 4]:
        cmd = [
            str(exe),
            str(target),
            str(image_dir),
            str(csv_path),
            str(topk),
            str(task_id),
        ]
    elif task_id == 5:
        cmd = [str(exe), target.name, str(csv_path), str(topk)]
    else:
        cmd = [str(exe), str(target), str(image_dir), str(csv_path), str(topk)]
        if task_id == 7 and show_bottom:
            cmd.append("--bottom")

    print(f"[CMD] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout + proc.stderr, parse_matches(proc.stdout, image_dir)


class ImageTile(QWidget):
    def __init__(
        self, title: str, subtitle: str, img_path: Optional[Path], size: QSize
    ):
        super().__init__()

        # Fixed size for the whole tile
        self.setFixedSize(size.width(), size.height() + 50)  # +50 for text

        self.img_label = QLabel()
        self.img_label.setFixedSize(size)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet(
            "background: #f5f5f5; border: 2px solid #ddd; border-radius: 4px;"
        )

        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.title_label.setWordWrap(True)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #333; padding: 3px;")

        self.sub_label = QLabel(subtitle)
        self.sub_label.setFont(QFont("Arial", 9))
        self.sub_label.setWordWrap(True)
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_label.setStyleSheet("color: #666; padding: 2px;")

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)
        layout.addWidget(self.img_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.sub_label)
        layout.addStretch()  # Push content to top
        self.setLayout(layout)

        self.setStyleSheet(
            "background: white; border: 1px solid #e0e0e0; border-radius: 6px;"
        )

        if img_path and img_path.exists():
            pm = load_pixmap(img_path, size.width(), size.height())
            if not pm.isNull():
                self.img_label.setPixmap(pm)
            else:
                self.img_label.setText("❌")
                self.img_label.setStyleSheet(
                    "background: #ffebee; color: #c62828; font-size: 24pt;"
                )
        else:
            self.img_label.setText("Not found")
            self.img_label.setStyleSheet(
                "background: #fff3e0; color: #f57c00; font-size: 10pt; font-family: Arial;"
            )


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CS5330 CBIR - Query Interface")

        self.build_dir = find_build_dir()
        self.image_dir: Optional[Path] = None
        self.csv_path: Optional[Path] = None
        self.csv_valid = False
        self.dataset: List[Path] = []
        self.target_idx = 0
        self.page = 0
        self.page_size = 9  # Will be updated by adjust_zoom()

        # Zoom levels: [tile_width, tile_height, cols, rows_per_page]
        # Page size calculation: row0 has (cols-1), subsequent rows have cols
        # For 2 rows: page_size = (cols-1) + cols
        self.zoom_levels = [
            (160, 120, 6, 2),  # Small: 5+6 = 11/page
            (200, 150, 5, 2),  # Medium: 4+5 = 9/page (default)
            (230, 175, 4, 2),  # Large: 3+4 = 7/page
        ]
        self.zoom_index = 1  # Start at Medium

        # CSV cache: task_id -> csv_path
        self.csv_cache: Dict[int, Path] = {}

        # Load saved settings
        self.settings = QSettings("CS5330", "CBIR")
        self.load_settings()

        self.init_ui()
        self.apply_styles()
        self.update_ui_state()

    def load_settings(self):
        """Load CSV paths from previous session."""
        for task_id in TASK_CONFIG.keys():
            key = f"csv_task{task_id}"
            csv_str = self.settings.value(key, "")
            if csv_str:
                csv_path = Path(csv_str)
                if csv_path.exists():
                    self.csv_cache[task_id] = csv_path
                    print(f"[LOAD] Cached CSV for Task {task_id}: {csv_path.name}")

    def save_settings(self):
        """Save CSV paths for future sessions."""
        for task_id, csv_path in self.csv_cache.items():
            self.settings.setValue(f"csv_task{task_id}", str(csv_path))
        print(f"[SAVE] Saved {len(self.csv_cache)} CSV paths")

    def init_ui(self):
        # Toggle button
        self.btn_toggle_ctrl = QPushButton("▼ Hide Controls")
        self.btn_toggle_ctrl.setCheckable(True)
        self.btn_toggle_ctrl.setChecked(False)
        self.btn_toggle_ctrl.clicked.connect(self.on_toggle_controls)
        self.btn_toggle_ctrl.setStyleSheet(
            """
            QPushButton {
                background: #e3f2fd;
                color: #1565c0;
                text-align: left;
                padding: 6px 12px;
                border: 1px solid #90caf9;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background: #bbdefb; }
        """
        )

        # Controls
        self.ctrl_group = QGroupBox("Controls")
        ctrl_layout = QGridLayout()

        ctrl_layout.addWidget(QLabel("Image Dir:"), 0, 0)
        self.dir_edit = QLineEdit()
        self.dir_edit.setReadOnly(True)
        ctrl_layout.addWidget(self.dir_edit, 0, 1, 1, 2)
        self.btn_pick_dir = QPushButton("Browse")
        self.btn_pick_dir.clicked.connect(self.on_pick_dir)
        ctrl_layout.addWidget(self.btn_pick_dir, 0, 3)

        self.csv_label = QLabel("CSV:")
        ctrl_layout.addWidget(self.csv_label, 1, 0)
        self.csv_edit = QLineEdit()
        self.csv_edit.setReadOnly(True)
        ctrl_layout.addWidget(self.csv_edit, 1, 1)

        csv_btns = QHBoxLayout()
        self.btn_pick_csv = QPushButton("Browse")
        self.btn_pick_csv.clicked.connect(self.on_pick_csv)
        csv_btns.addWidget(self.btn_pick_csv)
        self.btn_build_csv = QPushButton("Build")
        self.btn_build_csv.setToolTip("Auto-generate (Task 1-4)")
        self.btn_build_csv.clicked.connect(self.on_build_csv)
        csv_btns.addWidget(self.btn_build_csv)
        ctrl_layout.addLayout(csv_btns, 1, 2, 1, 2)

        self.csv_status = QLabel("")
        self.csv_status.setWordWrap(True)
        ctrl_layout.addWidget(self.csv_status, 2, 1, 1, 3)

        # Compact row
        compact_layout = QHBoxLayout()
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        self.target_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.target_combo.setMinimumHeight(32)
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)

        # Handle text edits
        self.target_combo.lineEdit().editingFinished.connect(
            self.on_target_text_changed
        )

        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.target_combo.setCompleter(self.completer)
        compact_layout.addWidget(self.target_combo, 2)

        self.btn_prev_target = QPushButton("◀")
        self.btn_prev_target.setFixedSize(32, 32)
        self.btn_prev_target.clicked.connect(lambda: self.shift_target(-1))
        compact_layout.addWidget(self.btn_prev_target)

        self.btn_next_target = QPushButton("▶")
        self.btn_next_target.setFixedSize(32, 32)
        self.btn_next_target.clicked.connect(lambda: self.shift_target(1))
        compact_layout.addWidget(self.btn_next_target)

        compact_layout.addWidget(QLabel("Task:"))
        self.task_box = QComboBox()
        self.task_box.setMinimumWidth(110)
        self.task_box.setMinimumHeight(32)
        for tid in sorted(TASK_CONFIG.keys()):
            self.task_box.addItem(f"Task {tid}", tid)
        self.task_box.currentIndexChanged.connect(self.on_task_changed)
        compact_layout.addWidget(self.task_box)

        self.topk_label = QLabel("Top K:")
        compact_layout.addWidget(self.topk_label)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 200)
        self.topk_spin.setValue(10)
        self.topk_spin.setFixedWidth(70)
        self.topk_spin.setMinimumHeight(32)
        compact_layout.addWidget(self.topk_spin)

        # Toggle: Top ↔ Bottom
        self.btn_top_bottom = QPushButton("⇅")
        self.btn_top_bottom.setCheckable(True)
        self.btn_top_bottom.setChecked(False)  # False=Top, True=Bottom
        self.btn_top_bottom.setFixedSize(32, 32)
        self.btn_top_bottom.setToolTip("Toggle: Top K (best) ↔ Bottom K (worst)")
        self.btn_top_bottom.clicked.connect(self.on_toggle_top_bottom)
        self.btn_top_bottom.setStyleSheet(
            """
            QPushButton {
                background: #f5f5f5;
                color: #666;
                font-weight: bold;
                font-size: 14pt;
                border: 2px solid #ddd;
                border-radius: 4px;
            }
            QPushButton:hover { background: #e0e0e0; }
            QPushButton:checked {
                background: #ff5722;
                color: white;
                border-color: #d32f2f;
            }
        """
        )
        compact_layout.addWidget(self.btn_top_bottom)

        compact_layout.addSpacing(10)

        self.btn_run = QPushButton("▶ Run (R)")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.setStyleSheet(
            """
            QPushButton { background: #4CAF50; color: white; font-weight: bold; padding: 6px 16px; }
            QPushButton:hover { background: #45a049; }
            QPushButton:disabled { background: #ccc; }
        """
        )
        self.btn_run.clicked.connect(self.on_run_query)
        compact_layout.addWidget(self.btn_run)

        ctrl_layout.addLayout(compact_layout, 3, 0, 1, 4)
        self.ctrl_group.setLayout(ctrl_layout)

        # Status
        self.status = QLabel("Ready. Select directory and CSV.")
        self.status.setStyleSheet(
            "padding: 6px; background: #e3f2fd; border: 1px solid #90caf9; border-radius: 4px; color: #1565c0;"
        )

        # Results (NO scrollbars - strict pagination)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: #fafafa; }")
        self.result_container = QWidget()
        self.grid = QGridLayout()
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(8)
        self.result_container.setLayout(self.grid)
        self.scroll.setWidget(self.result_container)

        # Page + Zoom controls
        page_layout = QHBoxLayout()

        # Zoom controls (left side)
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedSize(32, 32)
        self.btn_zoom_out.setToolTip("Zoom out (smaller tiles)")
        self.btn_zoom_out.clicked.connect(lambda: self.adjust_zoom(-1))
        page_layout.addWidget(self.btn_zoom_out)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("font-weight: bold; padding: 0 8px;")
        page_layout.addWidget(self.zoom_label)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(32, 32)
        self.btn_zoom_in.setToolTip("Zoom in (larger tiles)")
        self.btn_zoom_in.clicked.connect(lambda: self.adjust_zoom(1))
        page_layout.addWidget(self.btn_zoom_in)

        page_layout.addStretch()

        # Page controls (center)
        self.btn_prev_page = QPushButton("◀ Prev")
        self.btn_prev_page.clicked.connect(lambda: self.shift_page(-1))
        page_layout.addWidget(self.btn_prev_page)
        self.page_label = QLabel("Page 1")
        self.page_label.setStyleSheet("font-weight: bold; padding: 0 10px;")
        page_layout.addWidget(self.page_label)
        self.btn_next_page = QPushButton("Next ▶")
        self.btn_next_page.clicked.connect(lambda: self.shift_page(1))
        page_layout.addWidget(self.btn_next_page)

        page_layout.addStretch()

        # Debug toggle
        self.btn_toggle_debug = QPushButton("▼ Show Program Output")
        self.btn_toggle_debug.setCheckable(True)
        self.btn_toggle_debug.setChecked(False)
        self.btn_toggle_debug.clicked.connect(self.on_toggle_debug)
        self.btn_toggle_debug.setStyleSheet(
            """
            QPushButton {
                background: #f5f5f5;
                color: #333;
                text-align: left;
                padding: 6px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QPushButton:checked { background: #e3f2fd; }
        """
        )

        self.raw_output = QTextEdit()
        self.raw_output.setReadOnly(True)
        self.raw_output.setFont(QFont("Courier", 9))
        self.raw_output.setMaximumHeight(120)
        self.raw_output.setVisible(False)

        # Main
        main_layout = QVBoxLayout()
        main_layout.setSpacing(6)
        main_layout.addWidget(self.btn_toggle_ctrl)
        main_layout.addWidget(self.ctrl_group)
        main_layout.addWidget(self.status)
        main_layout.addWidget(self.scroll, 1)
        main_layout.addLayout(page_layout)
        main_layout.addWidget(self.btn_toggle_debug)
        main_layout.addWidget(self.raw_output)

        self.setLayout(main_layout)
        self.resize(1400, 900)

    def apply_styles(self):
        self.setStyleSheet(
            """
            QWidget { 
                font-family: Arial, 'Helvetica Neue', sans-serif; 
                font-size: 11pt; 
            }
            QGroupBox { 
                font-family: Arial;
                font-size: 11pt;
                font-weight: bold; 
                border: 2px solid #ddd; 
                border-radius: 6px; 
                margin-top: 4px; 
                padding-top: 8px; 
            }
            QGroupBox::title { 
                color: #333; 
                padding: 0 10px; 
                background: white; 
                font-family: Arial;
                font-size: 11pt;
            }
            QPushButton { 
                font-family: Arial;
                font-size: 11pt;
                background: #2196F3; 
                color: white; 
                padding: 6px 12px; 
                border: none; 
                border-radius: 4px; 
            }
            QPushButton:hover { background: #1976D2; }
            QPushButton:disabled { background: #BDBDBD; color: #757575; }
            QLineEdit, QComboBox { 
                font-family: Arial;
                font-size: 11pt;
                padding: 5px; 
                border: 1px solid #ccc; 
                border-radius: 4px; 
            }
            QSpinBox { 
                font-family: Arial;
                font-size: 11pt;
                padding: 4px; 
                border: 1px solid #ccc; 
                border-radius: 4px; 
            }
            QLabel {
                font-family: Arial;
                font-size: 11pt;
                color: #333;
            }
        """
        )
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        self.setPalette(palette)

    def closeEvent(self, event):
        """Save settings on close."""
        self.save_settings()
        event.accept()

    def on_toggle_controls(self, checked: bool):
        self.ctrl_group.setVisible(not checked)
        self.btn_toggle_ctrl.setText(
            "▶ Show Controls" if checked else "▼ Hide Controls"
        )

    def on_toggle_debug(self, checked: bool):
        self.raw_output.setVisible(checked)
        self.btn_toggle_debug.setText("▲ Hide Output" if checked else "▼ Show Output")

    def on_pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Image Directory")
        if not d:
            return
        QTimer.singleShot(0, lambda: self._process_directory(d))

    def _process_directory(self, d: str):
        try:
            self.image_dir = Path(d)
            self.dir_edit.setText(str(self.image_dir))
            self.dataset = list_images(self.image_dir)
            if not self.dataset:
                QMessageBox.warning(self, "No Images", "No images found.")
                return
            self.target_combo.clear()
            for img in self.dataset:
                self.target_combo.addItem(img.name)
            self.completer.setModel(
                QStringListModel([img.name for img in self.dataset])
            )
            self.target_idx = 0
            self.page = 0
            self.update_ui_state()
            self.status.setText(f"✓ Loaded {len(self.dataset)} images")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_pick_csv(self):
        f, _ = QFileDialog.getOpenFileName(self, "Choose CSV", "", "CSV (*.csv)")
        if not f:
            return
        QTimer.singleShot(0, lambda: self._process_csv(f))

    def _process_csv(self, f: str):
        try:
            self.csv_path = Path(f)
            self.csv_edit.setText(str(self.csv_path.name))

            # Cache this CSV for current task
            task_id = self.task_box.currentData()
            self.csv_cache[task_id] = self.csv_path

            self.validate_current_csv()
            self.update_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_build_csv(self):
        if not self.image_dir or not self.build_dir:
            QMessageBox.warning(self, "Not Ready", "Select directory first.")
            return

        task_id = self.task_box.currentData()
        _, _, _, can_build = TASK_CONFIG[task_id]
        if not can_build:
            QMessageBox.information(
                self, "Cannot Build", f"Task {task_id} needs pre-computed embeddings."
            )
            return

        progress = QProgressDialog("Building features...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()

        success, message, csv_path = build_features(
            self.build_dir, self.image_dir, task_id
        )
        progress.close()

        if success and csv_path:
            self.csv_path = csv_path
            self.csv_edit.setText(csv_path.name)

            # Cache this CSV
            self.csv_cache[task_id] = csv_path

            self.validate_current_csv()
            self.update_ui_state()

            # Custom success message (prettier than default QMessageBox)
            msg = QMessageBox(self)
            msg.setWindowTitle("Build Complete")
            msg.setText(f"<b>Feature database built successfully!</b>")
            msg.setInformativeText(
                f"<p style='font-size: 11pt; font-family: Arial;'>"
                f"File: <b>{csv_path.name}</b><br>"
                f"Location: {csv_path.parent}<br>"
                f"Task {task_id} features ready to use.</p>"
            )
            msg.setIcon(QMessageBox.Icon.NoIcon)  # No icon!
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setStyleSheet(
                """
                QMessageBox {
                    background: white;
                    font-family: Arial;
                }
                QLabel {
                    font-family: Arial;
                    font-size: 12pt;
                    color: #333;
                }
                QPushButton {
                    background: #4CAF50;
                    color: white;
                    padding: 8px 24px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11pt;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background: #45a049;
                }
            """
            )
            msg.exec()
        else:
            QMessageBox.critical(self, "Failed", message)

    def validate_current_csv(self):
        if not self.csv_path:
            self.csv_valid = False
            self.csv_status.setText("")
            return
        task_id = self.task_box.currentData()
        _, _, expected_dims, _ = TASK_CONFIG[task_id]
        is_valid, message, _ = validate_csv(self.csv_path, expected_dims)
        self.csv_valid = is_valid
        if is_valid:
            self.csv_status.setText(message)
            self.csv_status.setStyleSheet(
                "padding: 3px 8px; background: #e8f5e9; border: 1px solid #4caf50; border-radius: 3px; color: #1b5e20; font-size: 9pt;"
            )
        else:
            self.csv_status.setText(f"⚠ {message}")
            self.csv_status.setStyleSheet(
                "padding: 3px 8px; background: #fff3e0; border: 1px solid #ff9800; border-radius: 3px; color: #e65100; font-size: 9pt;"
            )

    def on_target_changed(self, index: int):
        if 0 <= index < len(self.dataset):
            self.target_idx = index
            self.page = 0

    def on_target_text_changed(self):
        """Handle manual text input in target combo box."""
        text = self.target_combo.currentText().strip()
        if not text or not self.dataset:
            return

        print(f"[TARGET] User typed: '{text}'")

        # Strategy 1: Exact match
        for i, img in enumerate(self.dataset):
            if img.name == text:
                if i != self.target_idx:
                    self.target_idx = i
                    self.target_combo.blockSignals(True)
                    self.target_combo.setCurrentIndex(i)
                    self.target_combo.blockSignals(False)
                    print(f"[TARGET] ✓ Exact match: {img.name}")
                return

        # Strategy 2: Case-insensitive exact match
        text_lower = text.lower()
        for i, img in enumerate(self.dataset):
            if img.name.lower() == text_lower:
                self.target_idx = i
                self.target_combo.blockSignals(True)
                self.target_combo.setCurrentIndex(i)
                self.target_combo.blockSignals(False)
                print(f"[TARGET] ✓ Case-insensitive match: {img.name}")
                return

        # Strategy 3: Match without extension (for partial typing like "pic.0022.j")
        # Remove common extensions and compare
        text_no_ext = text
        for ext in [".jpg", ".jpeg", ".png", ".j", ".jp", ".pn"]:
            if text_lower.endswith(ext):
                text_no_ext = text[: -len(ext)]
                break

        for i, img in enumerate(self.dataset):
            img_no_ext = img.stem  # filename without extension
            if img_no_ext.lower() == text_no_ext.lower():
                self.target_idx = i
                self.target_combo.blockSignals(True)
                self.target_combo.setCurrentIndex(i)
                self.target_combo.blockSignals(False)
                print(f"[TARGET] ✓ Stem match: '{text}' -> {img.name}")
                return

        # Strategy 4: Starts with match
        for i, img in enumerate(self.dataset):
            if img.name.lower().startswith(text_lower):
                self.target_idx = i
                self.target_combo.blockSignals(True)
                self.target_combo.setCurrentIndex(i)
                self.target_combo.blockSignals(False)
                print(f"[TARGET] ✓ Starts with: '{text}' -> {img.name}")
                return

        # Strategy 5: Contains match (fallback)
        for i, img in enumerate(self.dataset):
            if text_lower in img.name.lower():
                self.target_idx = i
                self.target_combo.blockSignals(True)
                self.target_combo.setCurrentIndex(i)
                self.target_combo.blockSignals(False)
                print(f"[TARGET] ✓ Contains: '{text}' -> {img.name}")
                return

        print(f"[TARGET] ⚠ No match found for: '{text}'")

    def on_task_changed(self):
        """Handle task change - restore cached CSV if available."""
        self.page = 0

        # Try to restore cached CSV for this task
        task_id = self.task_box.currentData()
        if task_id in self.csv_cache:
            cached_csv = self.csv_cache[task_id]
            if cached_csv.exists():
                self.csv_path = cached_csv
                self.csv_edit.setText(cached_csv.name)
                print(
                    f"[RESTORE] Using cached CSV for Task {task_id}: {cached_csv.name}"
                )
            else:
                # Cache is stale
                del self.csv_cache[task_id]
                self.csv_path = None
                self.csv_edit.setText("")
        else:
            # No cache for this task
            self.csv_path = None
            self.csv_edit.setText("")

        self.update_csv_visibility()
        if self.csv_path:
            self.validate_current_csv()
        self.update_ui_state()

    def on_toggle_top_bottom(self, checked: bool):
        """Toggle between Top K and Bottom K."""
        if checked:
            # switch to Bottom K
            self.topk_label.setText("Bottom K:")
            self.btn_top_bottom.setToolTip("Showing worst matches (click to show best)")
        else:
            # switch to Top K
            self.topk_label.setText("Top K:")
            self.btn_top_bottom.setToolTip("Showing best matches (click to show worst)")
        self.status.setText("Mode changed. Click Run to refresh results.")

    def on_topk_changed(self):
        self.page = 0

    def shift_target(self, delta: int):
        if not self.dataset:
            return
        self.target_idx = (self.target_idx + delta) % len(self.dataset)
        self.target_combo.setCurrentIndex(self.target_idx)

    def shift_page(self, delta: int):
        self.page = max(0, self.page + delta)
        self.refresh_results()

    def adjust_zoom(self, delta: int):
        """Adjust tile size (zoom in/out)."""
        new_index = self.zoom_index + delta
        if 0 <= new_index < len(self.zoom_levels):
            self.zoom_index = new_index

            # Get zoom config
            tile_w, tile_h, cols, rows = self.zoom_levels[self.zoom_index]

            # Update display
            zoom_names = ["Small", "Medium", "Large"]
            self.zoom_label.setText(f"{zoom_names[self.zoom_index]}")

            # Calculate page_size: first row has (cols-1), rest have cols
            # For 2 rows: (cols-1) + cols = 2*cols - 1
            self.page_size = (cols - 1) + cols if rows == 2 else cols * rows

            print(
                f"[ZOOM] {zoom_names[self.zoom_index]}: {tile_w}×{tile_h}, {cols}cols×{rows}rows → {self.page_size} results/page"
            )

            # Refresh display
            if self.dataset and self.csv_valid:
                self.page = 0
                self.refresh_results()

        # Update button states
        self.btn_zoom_out.setEnabled(self.zoom_index > 0)
        self.btn_zoom_in.setEnabled(self.zoom_index < len(self.zoom_levels) - 1)

    def on_run_query(self):
        self.refresh_results()
        self.btn_toggle_ctrl.setChecked(True)
        self.on_toggle_controls(True)

    def update_csv_visibility(self):
        task_id = self.task_box.currentData()
        _, needs_csv, _, can_build = TASK_CONFIG.get(task_id, (None, False, 0, False))
        self.csv_label.setVisible(needs_csv)
        self.csv_edit.setVisible(needs_csv)
        self.btn_pick_csv.setVisible(needs_csv)
        self.btn_build_csv.setVisible(needs_csv and can_build)
        self.csv_status.setVisible(needs_csv and self.csv_path is not None)

    def update_ui_state(self):
        self.update_csv_visibility()
        task_id = self.task_box.currentData()
        _, needs_csv, _, _ = TASK_CONFIG.get(task_id, (None, False, 0, False))
        has_dir = bool(self.image_dir and self.dataset)
        has_csv = bool((not needs_csv) or (self.csv_path and self.csv_valid))
        can_run = bool(has_dir and has_csv and self.build_dir)
        self.btn_run.setEnabled(can_run)
        self.btn_prev_target.setEnabled(has_dir)
        self.btn_next_target.setEnabled(has_dir)
        self.target_combo.setEnabled(has_dir)
        self.btn_build_csv.setEnabled(bool(has_dir and self.build_dir))

    def clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def refresh_results(self):
        if not self.image_dir or not self.dataset or not self.build_dir:
            return

        # Check if typed text matches current selection
        typed_text = self.target_combo.currentText().strip()
        current_name = self.dataset[self.target_idx].name if self.dataset else ""

        if typed_text != current_name:
            # User typed something different - try to match it
            print(
                f"[CHECK] Typed '{typed_text}' != current '{current_name}', re-matching..."
            )
            self.on_target_text_changed()

            # After re-matching, check if we found a match
            new_current = self.dataset[self.target_idx].name
            new_typed = self.target_combo.currentText().strip()

            if new_typed != new_current:
                # Still no match - show error
                QMessageBox.warning(
                    self,
                    "Invalid Target",
                    f"Cannot find image matching: '{typed_text}'\n\n"
                    f"Please select from the dropdown or type a valid filename.\n"
                    f"Examples: pic.0435.jpg, pic.1016.jpg",
                )
                # Reset to a valid selection
                self.target_combo.setCurrentIndex(self.target_idx)
                return

        task_id = self.task_box.currentData()
        _, needs_csv, _, _ = TASK_CONFIG[task_id]

        if needs_csv and (not self.csv_path or not self.csv_valid):
            QMessageBox.warning(self, "Invalid CSV", "Select or build CSV.")
            return

        topk = self.topk_spin.value()
        target_path = self.dataset[self.target_idx]

        show_bottom = self.btn_top_bottom.isChecked()

        try:
            text, matches = run_query(
                self.build_dir,
                task_id,
                target_path,
                self.image_dir,
                self.csv_path or Path(""),
                topk,
                show_bottom=show_bottom,
            )
        except Exception as ex:
            QMessageBox.critical(self, "Query Failed", str(ex))
            return

        self.raw_output.setText(text[:5000])

        if not matches:
            self.status.setText(f"⚠ No matches")
            self.status.setStyleSheet(
                "padding: 6px; background: #fff3e0; border: 1px solid #ff9800; border-radius: 4px; color: #e65100;"
            )
            page_matches = []
        else:
            if show_bottom:
                # Bottom K
                self.status.setStyleSheet(
                    "padding: 6px; background: #ffebee; border: 1px solid #f44336; ..."
                )
            else:
                # Top K
                self.status.setStyleSheet(
                    "padding: 6px; background: #e8f5e9; border: 1px solid #4caf50; ..."
                )

            total_pages = (len(matches) + self.page_size - 1) // self.page_size
            self.status.setText(
                f"✓ Task {task_id} | {target_path.name} | {len(matches)} matches"
            )
            self.status.setStyleSheet(
                "padding: 6px; background: #e8f5e9; border: 1px solid #4caf50; border-radius: 4px; color: #1b5e20;"
            )
            self.page_label.setText(f"Page {self.page + 1} of {total_pages}")

            start = self.page * self.page_size
            page_matches = matches[start : start + self.page_size]

        self.clear_grid()

        # Use current zoom level
        tile_width, tile_height, cols, rows = self.zoom_levels[self.zoom_index]
        tile_size = QSize(tile_width, tile_height)

        # Target tile (always at 0,0)
        target_tile = ImageTile("🎯 TARGET", target_path.name, target_path, tile_size)
        target_tile.setStyleSheet(
            "background: #fff3e0; border: 3px solid #ff9800; border-radius: 6px;"
        )
        self.grid.addWidget(target_tile, 0, 0)

        # Result tiles - fill row 0 first (starting from column 1), then subsequent rows
        show_bottom = self.btn_top_bottom.isChecked()

        r, c = 0, 1
        for m in page_matches:
            p = Path(m.fullpath)

            print(f"[IMG] #{m.rank}: {m.filename}")
            print(f"      path: {p}, exists: {p.exists()}")

            if show_bottom:
                label = f"Worst #{m.rank}\nd={m.dist:.5f}"
            else:
                label = f"#{m.rank}\nd={m.dist:.5f}"

            tile = ImageTile(label, m.filename, p, tile_size)
            self.grid.addWidget(tile, r, c)
            c += 1
            if c >= cols:
                r, c = r + 1, 0  # Next row, start from column 0
                # (no more TARGET, so use full width)

        self.btn_prev_page.setEnabled(self.page > 0)
        self.btn_next_page.setEnabled((self.page + 1) * self.page_size < len(matches))

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_N:
            self.shift_target(1)
        elif key == Qt.Key.Key_P:
            self.shift_target(-1)
        elif key == Qt.Key.Key_R:
            self.on_run_query()
        elif key == Qt.Key.Key_Left:
            self.shift_page(-1)
        elif key == Qt.Key.Key_Right:
            self.shift_page(1)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):  # + or =
            self.adjust_zoom(1)
        elif key == Qt.Key.Key_Minus:  # -
            self.adjust_zoom(-1)
        elif key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()


def main():
    app = QApplication(sys.argv)
    app.setOrganizationName("CS5330")
    app.setApplicationName("CBIR")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
