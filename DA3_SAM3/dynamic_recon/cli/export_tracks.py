from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_recon.data.cache_io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--query-frame", type=int, default=0)
    parser.add_argument("--query-grid-step", type=int, default=32)
    parser.add_argument("--mode", choices=["static", "dynamic", "both"], default="both")
    args = parser.parse_args()
    save_json(Path(args.run_dir) / "exports" / "export_tracks_request.json", vars(args))


if __name__ == "__main__":
    main()
