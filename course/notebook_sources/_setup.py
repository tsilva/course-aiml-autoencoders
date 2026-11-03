import os
from pathlib import Path
import subprocess
import sys

# Find an existing checkout before fetching one in a fresh Colab session.
_start = Path.cwd().resolve()
_course_root = next((
    candidate for candidate in (_start, *_start.parents)
    if (candidate / "course/curriculum.yaml").is_file()
    and (candidate / "src/course_aiml_autoencoders").is_dir()
), None)
if _course_root is None:
    try:
        import google.colab
    except ImportError as error:
        raise RuntimeError("Open this notebook from the course checkout or in Google Colab.") from error
    _course_root = _start / "course-aiml-autoencoders"
    if not _course_root.exists():
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", "main",
            "https://github.com/tsilva/course-aiml-autoencoders.git",
            str(_course_root),
        ], check=True)
    if not (_course_root / "course/curriculum.yaml").is_file():
        raise RuntimeError(f"Incomplete course checkout at {_course_root}; rename it and rerun setup.")
os.chdir(_course_root)
_course_src = str(_course_root / "src")
if _course_src not in sys.path:
    sys.path.insert(0, _course_src)
print("Course ready:", _course_root)
