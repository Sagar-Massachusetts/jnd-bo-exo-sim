"""
demo_step8.py
-------------
Convenience wrapper: invokes the CLI runner with --quick so you can see the
summary table and figures in ~30s.
"""
import subprocess
import sys

def main():
    cmd = [
        sys.executable, "run_experiments.py",
        "--quick",
        "--out-dir", "results/run_quick",
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)

if __name__ == "__main__":
    main()
