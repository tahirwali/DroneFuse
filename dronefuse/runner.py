"""
Run notebook steps so that one failure never stops "Save & Run All".

Each step is a plain function. `StepRunner.run` calls it, catches any error, prints the
traceback, records the status and time, and writes `steps.json` after every step, so a
partial run still leaves a readable record. Steps are skipped once the time budget is spent.
"""
import json
import time
import traceback
from pathlib import Path


class StepRunner:
    def __init__(self, out_dir, time_budget_minutes):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.budget = time_budget_minutes * 60
        self.start = time.time()
        self.steps = []

    def minutes_left(self):
        return (self.budget - (time.time() - self.start)) / 60

    def run(self, name, function, *args, **kwargs):
        """Return the function's result, or None if it failed or was skipped."""
        if self.minutes_left() <= 0:
            print(f"[skipped] {name}: time budget used up")
            self._record(name, "skipped", 0.0, "time budget used up")
            return None
        print(f"\n=== {name} ===")
        began = time.time()
        try:
            result = function(*args, **kwargs)
        except Exception as error:  # a failed step must not stop the notebook
            traceback.print_exc()
            self._record(name, "error", time.time() - began, f"{type(error).__name__}: {error}")
            print(f"[error] {name} failed after {time.time() - began:.0f} s; continuing with the next step.")
            return None
        self._record(name, "ok", time.time() - began, "")
        print(f"[ok] {name} ({time.time() - began:.0f} s, {self.minutes_left():.0f} min of budget left)")
        return result

    def _record(self, name, status, seconds, message):
        self.steps.append({"step": name, "status": status, "seconds": round(seconds, 1), "message": message})
        with open(self.out_dir / "steps.json", "w") as f:
            json.dump(self.steps, f, indent=1)

    def failed(self):
        return [s["step"] for s in self.steps if s["status"] != "ok"]
