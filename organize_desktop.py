"""Organize Desktop files by extension.

This script scans the user's Desktop directory and moves files into
subfolders based on their extension. Folders are created on-the-fly if
missing. Unknown extensions are moved into an 'Others' folder.

Usage:
    python organize_desktop.py

The script uses only standard libraries (pathlib, shutil) and supports
Python 3.9+.
"""

from pathlib import Path
import shutil
from typing import Dict, Iterable

# Mapping of destination folder names to a set of associated extensions.
EXTENSION_MAP: Dict[str, Iterable[str]] = {
    "Images": {".jpg", ".jpeg", ".png", ".gif"},
    "Documents": {".pdf", ".docx", ".txt", ".xlsx"},
    "Videos": {".mp4", ".avi", ".mov"},
    "Music": {".mp3", ".wav"},
    "Code": {".py", ".sql", ".ipynb"},
}

DEFAULT_FOLDER = "Others"

def get_folder_name(suffix: str) -> str:
    """Return the destination folder for a given file extension.

    Parameters
    ----------
    suffix: str
        File extension (including the leading dot), case-insensitive.
    """
    suffix = suffix.lower()
    for folder, extensions in EXTENSION_MAP.items():
        if suffix in extensions:
            return folder
    return DEFAULT_FOLDER

def organize_desktop(desktop_path: Path | None = None) -> None:
    """Organize files on the desktop into folders by extension.

    Parameters
    ----------
    desktop_path: Path | None
        Optional path to a desktop directory. If not provided, the
        function uses ``Path.home() / 'Desktop'``.
    """
    desktop = desktop_path or Path.home() / "Desktop"

    if not desktop.exists():
        raise FileNotFoundError(f"Desktop path does not exist: {desktop}")

    for item in desktop.iterdir():
        if not item.is_file():
            continue

        folder_name = get_folder_name(item.suffix)
        destination_dir = desktop / folder_name
        destination_dir.mkdir(exist_ok=True)

        destination = destination_dir / item.name
        shutil.move(str(item), str(destination))
        print(f"Moved {item.name} -> {folder_name}/")

if __name__ == "__main__":
    organize_desktop()
