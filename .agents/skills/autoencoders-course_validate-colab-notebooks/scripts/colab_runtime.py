"""Embedded by prepare_validation.py; executes only in the generated Colab copy."""

import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zlib

from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import show as show_figures


payload = json.loads(zlib.decompress(base64.b64decode(ENCODED_SNAPSHOT)))
snapshot = payload["snapshot"]
fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
if fingerprint != payload["fingerprint"]:
    raise RuntimeError("Snapshot fingerprint mismatch")
root = Path.cwd().resolve()
if root.name != "course-aiml-autoencoders" or not (root / ".git").is_dir():
    raise RuntimeError("Expected the checkout created by the original setup cell")
if (root / "runs").exists():
    raise RuntimeError("Use a fresh checkout: training evidence already exists")
# Remove stale source files from fetched main before restoring the exact snapshot.
shutil.rmtree(root / "src/course_aiml_autoencoders")
shutil.rmtree(root / "recipes")
for relative, content in snapshot["files"].items():
    destination = (root / relative).resolve()
    destination.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
evidence = root / "runs/colab-validation" / fingerprint[:12]
evidence.mkdir(parents=True)
versions = {package: importlib.metadata.version(package)
            for package in ("torch", "torchvision", "matplotlib", "PyYAML", "ipython")}
report = {"fingerprint": fingerprint, "head": payload["head"],
          "started_at": datetime.now(timezone.utc).isoformat(),
          "python": sys.version, "packages": versions,
          "semantic_review": "pending", "lessons": []}
print("Runtime:", sys.version.split()[0], versions, flush=True)
print("Snapshot:", fingerprint, flush=True)


def save_report():
    (evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")


save_report()
for lesson in snapshot["lessons"]:
    os.chdir(root)
    # Isolate notebook variables and course modules; keep matching training evidence.
    for name in list(sys.modules):
        if name == "course_aiml_autoencoders" or name.startswith("course_aiml_autoencoders."):
            del sys.modules[name]
    namespace = {"__name__": "__main__"}
    notebook = lesson["notebook"]
    result = {"id": lesson["id"], "path": lesson["path"],
              "sha256": lesson["sha256"], "expected_code_cells": lesson["code_cells"],
              "completed_code_cells": 0, "status": "running", "cells": []}
    report["lessons"].append(result)
    started = time.monotonic()
    print(f"\n=== Lesson {lesson['id']}: {lesson['path']} ===", flush=True)
    save_report()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        error = None
        elapsed = time.monotonic()
        with capture_output() as captured:
            try:
                exec(compile(cell["source"], f"{lesson['path']}:cell:{index + 1}", "exec"), namespace)
                # exec has no IPython post-cell hook; flush inline figures explicitly.
                show_figures(close=True)
            except Exception:
                error = traceback.format_exc()
        cell["execution_count"] = result["completed_code_cells"] + 1
        cell["outputs"] = []
        for stream in ("stdout", "stderr"):
            text = getattr(captured, stream)
            if text:
                cell["outputs"].append({"output_type": "stream", "name": stream, "text": text})
        for output in captured.outputs:
            cell["outputs"].append({"output_type": "display_data", "data": output.data,
                                    "metadata": output.metadata})
        if error:
            cell["outputs"].append({"output_type": "error", "ename": "CellExecutionError",
                                    "evalue": error.splitlines()[-1], "traceback": error.splitlines()})
        captured.show()
        result["cells"].append({"index": index + 1, "seconds": round(time.monotonic() - elapsed, 2),
                                "status": "fail" if error else "pass"})
        if error:
            result["status"] = "execution_fail"
            result["traceback"] = error
            print(error, flush=True)
        else:
            result["completed_code_cells"] += 1
        (evidence / Path(lesson["path"]).name).write_text(json.dumps(notebook, indent=1) + "\n")
        save_report()
        if error:
            break
    if result["status"] == "running":
        result["status"] = "execution_pass"
    result["seconds"] = round(time.monotonic() - started, 2)
    print(f"{lesson['id']}: {result['status']} "
          f"({result['completed_code_cells']}/{result['expected_code_cells']} cells)", flush=True)
    save_report()
report["finished_at"] = datetime.now(timezone.utc).isoformat()
save_report()
print("\nEvidence:", evidence, flush=True)
print(json.dumps(report, indent=2), flush=True)
if any(item["status"] != "execution_pass" for item in report["lessons"]):
    raise RuntimeError("Some lessons failed; inspect report.json and cell outputs")
print("All selected code cells completed. Expected-result review is still pending.", flush=True)
