from __future__ import annotations

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "-m", "dynamic_recon.cli.export_tracks", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
