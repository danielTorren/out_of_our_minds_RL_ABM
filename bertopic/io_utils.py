# -*- coding: utf-8 -*-
"""
io_utils.py — lightweight I/O helpers for this project
======================================================

What this module provides
-------------------------
- Paths & dialogs
  - choose_open_file()     : File-open dialog (CSV-focused), Qt first, Tk fallback
  - choose_save_dir()      : Folder-select dialog, Qt first, Tk fallback
  - choose_save_as()       : "Save As…" file dialog for CSV/HTML/etc., Qt first, Tk fallback
  - ensure_dir(Path)       : mkdir -p
  - open_folder_in_explorer(Path): open folder in OS file explorer (Win/macOS/Linux)

- Filenames & logging
  - build_default_basename(prefix, suffix="", with_timestamp=True)
  - file_sha256(Path, chunk_mb=4)         : SHA-256 checksum for provenance
  - write_manifest(data: dict, path: Path): Save a JSON sidecar ("run manifest")

- Environment metadata (for manifests)
  - get_env_versions()     : Collect key package versions + Python/platform + CUDA info
  - get_git_info()         : {commit, branch, is_dirty} if running inside a Git repo

Version history
---------------
v1.0 (2025-10-22)  Initial extraction from scripts. Qt-first dialogs, JSON manifest.

Quick usage
-----------
from pathlib import Path
from io_utils import (
    choose_open_file, choose_save_dir, choose_save_as, ensure_dir,
    build_default_basename, open_folder_in_explorer,
    write_manifest, get_env_versions, get_git_info, file_sha256
)

out_dir = ensure_dir(Path("outputs"))
base = build_default_basename("bertopic_run", "mydata_seed42")
manifest_path = out_dir / f"{base}_manifest.json"
write_manifest({"hello": "world"}, manifest_path)

Notes
-----
- Qt dialogs work best inside Spyder. Tk is used as a fallback.
- Keep this file under version control (GitHub). Data/models/outputs should be ignored.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# -----------------------
# Dialogs (Qt-first; Tk fallback)
# -----------------------
def build_timestamped_folder(prefix: str = "Out") -> str:
    """Return a unique folder name like 'Out_YYYYMMDD_HHMMSS'."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def choose_open_file(title: str = "Select input CSV",
                     initial_dir: Optional[Path] = None,
                     name_filter: str = "CSV files (*.csv)") -> Optional[Path]:
    # 1) Qt
    try:
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance()
        owns = False
        if app is None:
            app = QtWidgets.QApplication([])
            owns = True
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilter(name_filter)
        if initial_dir:
            dlg.setDirectory(str(initial_dir))
        dlg.setWindowTitle(title)
        path = None
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            path = dlg.selectedFiles()[0]
        if owns:
            app.quit()
        return Path(path) if path else None
    except Exception:
        pass
    # 2) Tk
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(initial_dir) if initial_dir else None
        )
        try:
            root.destroy()
        except Exception:
            pass
        return Path(path) if path else None
    except Exception:
        return None


def choose_save_dir(initial_dir: Optional[Path] = None,
                    title: str = "Select output folder") -> Optional[Path]:
    # 1) Qt
    try:
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance()
        owns = False
        if app is None:
            app = QtWidgets.QApplication([])
            owns = True
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, title, str(initial_dir) if initial_dir else ""
        )
        if owns:
            app.quit()
        return Path(path) if path else None
    except Exception:
        pass
    # 2) Tk
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askdirectory(
            title=title,
            initialdir=str(initial_dir) if initial_dir else None
        )
        try:
            root.destroy()
        except Exception:
            pass
        return Path(path) if path else None
    except Exception:
        return None


def choose_save_as(default_name: str = "output.csv",
                   initial_dir: Optional[Path] = None,
                   title: str = "Save As…",
                   filter_label: str = "CSV files (*.csv)",
                   default_suffix: str = "csv") -> Optional[Path]:
    # 1) Qt
    try:
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance()
        owns = False
        if app is None:
            app = QtWidgets.QApplication([])
            owns = True
        dlg = QtWidgets.QFileDialog()
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setNameFilter(filter_label)
        dlg.setDefaultSuffix(default_suffix)
        if initial_dir:
            dlg.setDirectory(str(initial_dir))
        if default_name:
            dlg.selectFile(default_name)
        dlg.setWindowTitle(title)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            path = dlg.selectedFiles()[0]
        else:
            path = None
        if owns:
            app.quit()
        return Path(path) if path else None
    except Exception:
        pass
    # 2) Tk
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=f".{default_suffix}",
            filetypes=[(filter_label.split(" (")[0], f"*.{default_suffix}")],
            initialdir=str(initial_dir) if initial_dir else None,
            initialfile=default_name
        )
        try:
            root.destroy()
        except Exception:
            pass
        return Path(path) if path else None
    except Exception:
        return None


# -----------------------
# Paths & filenames
# -----------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def build_default_basename(prefix: str,
                           suffix: str = "",
                           with_timestamp: bool = True) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if with_timestamp else ""
    mid = f"_{suffix}" if suffix else ""
    end = f"_{ts}" if ts else ""
    return f"{prefix}{mid}{end}"

def open_folder_in_explorer(folder: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)], check=False)
        else:
            subprocess.run(["xdg-open", str(folder)], check=False)
    except Exception:
        pass


# -----------------------
# Manifests & environment info
# -----------------------
def file_sha256(path: Path, chunk_mb: int = 4) -> Optional[str]:
    """Compute SHA-256 checksum for a file (for provenance)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_mb * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def get_env_versions() -> Dict[str, Any]:
    """Collect key versions and environment bits."""
    import platform
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import bertopic
        info["bertopic"] = getattr(bertopic, "__version__", "unknown")
    except Exception:
        pass
    for pkg in ["torch", "sentence_transformers", "umap", "hdbscan",
                "pandas", "numpy", "plotly"]:
        try:
            mod = __import__(pkg if pkg != "sentence_transformers" else "sentence_transformers")
            ver = getattr(mod, "__version__", None) or getattr(mod, "version", None)
            info[pkg] = ver
        except Exception:
            info[pkg] = None
    # CUDA info if torch is present
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info

def get_git_info(repo_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Return {commit, branch, is_dirty} if inside a Git repo, else {}."""
    try:
        cwd = str(repo_dir or Path.cwd())
        def run_git(args):
            return subprocess.check_output(["git", "-C", cwd] + args, stderr=subprocess.DEVNULL).decode().strip()
        commit = run_git(["rev-parse", "HEAD"])
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        status = run_git(["status", "--porcelain"])
        return {"commit": commit, "branch": branch, "is_dirty": bool(status)}
    except Exception:
        return {}

def write_manifest(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
