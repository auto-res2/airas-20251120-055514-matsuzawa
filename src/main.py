import subprocess
import sys
from pathlib import Path

import hydra

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    if not cfg.get("run"):
        print("ERROR: run=<run_id> is required", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(cfg.get("results_dir", "outputs")).expanduser().resolve()
    mode = cfg.get("mode", "full")

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={results_dir}",
        f"mode={mode}",
    ]
    print("Launching training subprocess:\n" + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()