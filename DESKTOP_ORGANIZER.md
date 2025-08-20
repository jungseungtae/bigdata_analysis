# Desktop File Organizer

This utility script organizes files on your Desktop into folders based on
file extensions. Each known extension is mapped to a category such as
**Images** or **Documents**. Unknown extensions are moved into an
`Others` folder.

## Requirements
- Python 3.9 or newer

## Usage
1. Ensure the script `organize_desktop.py` is located somewhere in your
   `PATH` or run it directly from this repository.
2. Run the script:
   ```bash
   python organize_desktop.py
   ```
3. The script scans `~/Desktop`, creating folders as needed and moving
   files into them. Each move is logged to the console.

## Extension Mapping
- **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`
- **Documents**: `.pdf`, `.docx`, `.txt`, `.xlsx`
- **Videos**: `.mp4`, `.avi`, `.mov`
- **Music**: `.mp3`, `.wav`
- **Code**: `.py`, `.sql`, `.ipynb`
- **Others**: any extension not listed above

## Future Enhancements
- **JSON Configuration**: Instead of the hard-coded dictionary, load
  rules from a JSON file so new categories and extensions can be defined
  without modifying the script. Example structure:
  ```json
  {
    "Images": [".jpg", ".png"],
    "CustomCategory": [".xyz"]
  }
  ```
- **Real-time Monitoring**: Use the [`watchdog`](https://pypi.org/project/watchdog/) library to monitor the Desktop and
  organize new files as soon as they appear. The library can trigger the
  `organize_desktop` function whenever a file creation event occurs.

## Notes
- The script moves files, so ensure you have backups if needed.
- Existing files with the same name in target folders will be
  overwritten.
