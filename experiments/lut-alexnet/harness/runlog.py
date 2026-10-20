"""Experiment-folder bookkeeping: config.json, an append-only metrics.csv, and summary.json keyed by run id.

config.json lists every distinct invocation that wrote to the folder (stages run one dataset group at a time
add to it rather than overwrite it). The summary is rewritten after every run, so an interrupted sweep restarts
where it stopped (``done``).
"""
import csv
import json
import math
import subprocess
from pathlib import Path


def git_state():
    """HEAD commit, and whether the experiment's *code* differs from it (result folders and results.md, which
    runs write while later stages start, are excluded)."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    code = [".", ":(exclude)exp_*", ":(exclude)results.md"]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", *code))}


class ExpDir:
    def __init__(self, path, config):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        config_path = self.path / "config.json"
        configs = json.loads(config_path.read_text())["invocations"] if config_path.exists() else []
        if config not in configs:
            config_path.write_text(json.dumps({"invocations": configs + [config]}, indent=2) + "\n")
        self.summary_path = self.path / "summary.json"
        self.summary = json.loads(self.summary_path.read_text()) if self.summary_path.exists() else {}
        self.summary.setdefault("runs", {})
        self.metrics_path = self.path / "metrics.csv"

    def done(self, run_id):
        return run_id in self.summary["runs"]

    def logger(self, run_id):
        def log(row):
            row = {"run": run_id, **row}
            new = not self.metrics_path.exists()
            with open(self.metrics_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        return log

    def record(self, run_id, result):
        self.summary["runs"][run_id] = result
        self._write()

    def select(self, cell, value):
        """Records a selection made from the runs (e.g. the lr chosen on val for a head config)."""
        self.summary.setdefault("selected", {})[cell] = value
        self._write()

    def _write(self):
        self.summary_path.write_text(json.dumps(self.summary, indent=2) + "\n")


def mean_se(xs):
    """Mean and standard error (sample std / sqrt(n)); SE is nan for n < 2."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in xs) / (n - 1)
    return m, math.sqrt(var / n)
