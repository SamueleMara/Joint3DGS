#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


def _ensure_matplotlib_cache(out_dir: Path) -> None:
    cache = out_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _object_dirs(input_root: Path) -> list[Path]:
    return sorted([p for p in input_root.iterdir() if p.is_dir()])


def _manifest_gt_map(manifest_path: Path) -> dict[str, str]:
    manifest = _load_json(manifest_path)
    objects = manifest.get("objects", [])
    return {str(obj["name"]): str(obj.get("joint_type", "unknown")) for obj in objects}


def _mean_points(per_view_csv: Path) -> float | None:
    if not per_view_csv.exists():
        return None
    with per_view_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    vals = [float(row["num_points"]) for row in rows]
    return sum(vals) / len(vals)


def _state_range(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _build_env(script_dir: Path, cache_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(cache_dir / "mpl"))
    env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    env.setdefault("TORCH_HOME", str(cache_dir / "torchhub"))
    return env


def _run_object(
    script_dir: Path,
    dataset_dir: Path,
    object_out_dir: Path,
    tracker_config: str,
    seg_config: str,
    joint_config: str,
    max_frames: int | None,
    report_sample_views: int,
    force: bool,
    cache_dir: Path,
) -> dict[str, Any]:
    result_json = object_out_dir / "joint_fused_world.json"
    report_index = object_out_dir / "report" / "index.html"

    if not force and result_json.exists() and report_index.exists():
        return {"status": "skipped", "message": "existing outputs reused"}

    object_out_dir.mkdir(parents=True, exist_ok=True)
    env = _build_env(script_dir, cache_dir)

    run_cmd = [
        sys.executable,
        "scripts/run_pipeline_multiview.py",
        "--input-dir",
        str(dataset_dir),
        "--rgb-root",
        "rgb",
        "--depth-root",
        "depth_npy",
        "--mask-root",
        "mask",
        "--camera-pattern",
        "cam_*",
        "--cameras-json",
        "metadata/cameras.json",
        "--depth-npy-scale",
        "1.0",
        "--tracker-config",
        tracker_config,
        "--seg-config",
        seg_config,
        "--joint-config",
        joint_config,
        "--out-dir",
        str(object_out_dir),
        "--no-progress",
        "--no-refine-global",
    ]
    if max_frames is not None:
        run_cmd.extend(["--max-frames", str(max_frames)])

    _run_cmd(run_cmd, cwd=script_dir, env=env)

    report_cmd = [
        sys.executable,
        "scripts/build_multiview_report.py",
        "--run-dir",
        str(object_out_dir),
        "--report-dir",
        str(object_out_dir / "report"),
        "--sample-view-count",
        str(report_sample_views),
    ]
    _run_cmd(report_cmd, cwd=script_dir, env=env)
    return {"status": "success", "message": "run complete"}


def _collect_result(
    name: str,
    gt_map: dict[str, str],
    object_out_dir: Path,
    status: str,
    message: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "gt_joint_type": gt_map.get(name, "unknown"),
        "status": status,
        "message": message,
        "pred_best_model": "",
        "loss": None,
        "pitch": None,
        "state_range": None,
        "type_prior_revolute": None,
        "type_prior_prismatic": None,
        "type_prior_screw": None,
        "num_views": None,
        "mean_points": None,
        "report_relpath": f"objects/{name}/report/index.html",
        "run_relpath": f"objects/{name}",
        "match_gt": None,
    }

    result_json = object_out_dir / "joint_fused_world.json"
    report_summary = object_out_dir / "report" / "summary.json"
    per_view_csv = object_out_dir / "report" / "per_view_summary.csv"
    if not result_json.exists():
        return row

    fused = _load_json(result_json)
    summary = _load_json(report_summary) if report_summary.exists() else {}
    priors = fused.get("type_priors") or {}
    pred = str(fused.get("best_model", ""))
    gt = row["gt_joint_type"]
    row.update(
        {
            "pred_best_model": pred,
            "loss": float(fused.get("loss", 0.0)),
            "pitch": fused.get("pitch"),
            "state_range": _state_range(list(fused.get("state") or [])),
            "type_prior_revolute": float(priors.get("revolute", 0.0)),
            "type_prior_prismatic": float(priors.get("prismatic", 0.0)),
            "type_prior_screw": float(priors.get("screw", 0.0)),
            "num_views": int(summary.get("num_views", len((fused.get("per_view") or {}).keys()))),
            "mean_points": _mean_points(per_view_csv),
            "match_gt": (gt == pred) if gt in {"revolute", "prismatic", "screw"} and pred else None,
        }
    )
    return row


def _write_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "name",
        "gt_joint_type",
        "status",
        "message",
        "pred_best_model",
        "match_gt",
        "loss",
        "pitch",
        "state_range",
        "num_views",
        "mean_points",
        "type_prior_revolute",
        "type_prior_prismatic",
        "type_prior_screw",
        "report_relpath",
        "run_relpath",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_prediction_counts(rows: list[dict[str, Any]], out_path: Path) -> None:
    plt = _plt()
    counts = Counter(row["pred_best_model"] for row in rows if row["status"] == "success" and row["pred_best_model"])
    names = sorted(counts)
    values = [counts[name] for name in names]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(names, values, color="#4C78A8")
    ax.set_title("Predicted Joint Types")
    ax.set_ylabel("Objects")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_gt_vs_pred(rows: list[dict[str, Any]], out_path: Path) -> None:
    plt = _plt()
    gt_names = sorted({row["gt_joint_type"] for row in rows if row["gt_joint_type"]})
    pred_names = sorted({row["pred_best_model"] for row in rows if row["pred_best_model"]})
    if not gt_names:
        gt_names = ["unknown"]
    if not pred_names:
        pred_names = ["unknown"]

    mat = np.zeros((len(gt_names), len(pred_names)), dtype=np.int32)
    gt_idx = {name: i for i, name in enumerate(gt_names)}
    pred_idx = {name: i for i, name in enumerate(pred_names)}
    for row in rows:
        if row["status"] != "success" or not row["pred_best_model"]:
            continue
        mat[gt_idx[row["gt_joint_type"]], pred_idx[row["pred_best_model"]]] += 1

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(pred_names)))
    ax.set_xticklabels(pred_names)
    ax.set_yticks(np.arange(len(gt_names)))
    ax.set_yticklabels(gt_names)
    ax.set_title("GT vs Predicted")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_loss_by_object(rows: list[dict[str, Any]], out_path: Path) -> None:
    plt = _plt()
    success_rows = [row for row in rows if row["status"] == "success" and row["loss"] is not None]
    names = [row["name"] for row in success_rows]
    losses = [float(row["loss"]) for row in success_rows]
    colors = ["#0F766E" if row["match_gt"] else "#E45756" for row in success_rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(names, losses, color=colors)
    ax.set_title("Fused Loss by Object")
    ax.set_ylabel("Loss")
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _benchmark_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [row for row in rows if row["status"] == "success"]
    failed_rows = [row for row in rows if row["status"] != "success"]
    pred_counts = Counter(row["pred_best_model"] for row in success_rows if row["pred_best_model"])
    gt_counts = Counter(row["gt_joint_type"] for row in rows if row["gt_joint_type"])
    exact_matches = sum(1 for row in success_rows if row["match_gt"] is True)
    accuracy = (exact_matches / len(success_rows)) if success_rows else 0.0
    return {
        "total_objects": len(rows),
        "successful_runs": len(success_rows),
        "failed_runs": len(failed_rows),
        "exact_gt_matches": exact_matches,
        "accuracy": accuracy,
        "prediction_counts": dict(pred_counts),
        "gt_counts": dict(gt_counts),
    }


def _write_index_html(out_root: Path, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    summary_rows = [
        ("Total Objects", str(stats["total_objects"])),
        ("Successful Runs", str(stats["successful_runs"])),
        ("Failed Runs", str(stats["failed_runs"])),
        ("Exact GT Matches", str(stats["exact_gt_matches"])),
        ("Exact Accuracy", f"{stats['accuracy'] * 100:.1f}%"),
        ("GT Counts", ", ".join(f"{k}: {v}" for k, v in sorted(stats["gt_counts"].items()))),
        (
            "Prediction Counts",
            ", ".join(f"{k}: {v}" for k, v in sorted(stats["prediction_counts"].items())) or "none",
        ),
    ]

    rows_html = []
    for row in rows:
        match = ""
        if row["match_gt"] is True:
            match = "yes"
        elif row["match_gt"] is False:
            match = "no"
        report_link = (
            f'<a href="{html.escape(row["report_relpath"])}">report</a>'
            if row["status"] == "success"
            else ""
        )
        run_link = f'<a href="{html.escape(row["run_relpath"])}">files</a>'
        cols = [
            html.escape(str(row["name"])),
            html.escape(str(row["gt_joint_type"])),
            html.escape(str(row["pred_best_model"] or "")),
            html.escape(str(match)),
            html.escape(str(row["status"])),
            "" if row["loss"] is None else f"{float(row['loss']):.4f}",
            "" if row["state_range"] is None else f"{float(row['state_range']):.4f}",
            "" if row["mean_points"] is None else f"{float(row['mean_points']):.1f}",
            html.escape(str(row["message"])),
            report_link,
            run_link,
        ]
        row_html = "<tr>" + "".join(f"<td>{cell}</td>" for cell in cols) + "</tr>"
        rows_html.append(row_html)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Preparation All Benchmark</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #66788a;
      --line: #d8d2c6;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: linear-gradient(180deg, #f8f5ef 0%, #efe8dc 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 8px 24px rgba(35, 35, 35, 0.06);
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
    }}
    figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    a {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Preparation All Benchmark</h1>
      <table>
        <tbody>
          {"".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in summary_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Charts</h2>
      <div class="grid">
        <figure><img src="prediction_counts.png" alt="Prediction counts"><figcaption>How many objects fell into each predicted joint type.</figcaption></figure>
        <figure><img src="gt_vs_pred.png" alt="GT vs predicted"><figcaption>Confusion-style view of ground truth versus prediction.</figcaption></figure>
        <figure><img src="loss_by_object.png" alt="Loss by object"><figcaption>Fused loss per object. Green means exact GT match, red means mismatch.</figcaption></figure>
      </div>
    </section>

    <section>
      <h2>Objects</h2>
      <table>
        <thead>
          <tr>
            <th>Object</th>
            <th>GT</th>
            <th>Pred</th>
            <th>Match</th>
            <th>Status</th>
            <th>Loss</th>
            <th>State Range</th>
            <th>Mean Points</th>
            <th>Message</th>
            <th>Report</th>
            <th>Files</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    (out_root / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="../Dataset/preparation_all")
    parser.add_argument("--manifest-json", default="../Dataset/urdf/manifest.json")
    parser.add_argument("--out-root", default="outputs/preparation_all_benchmark_8f")
    parser.add_argument("--tracker-config", default="configs/tracker_usb.yaml")
    parser.add_argument("--seg-config", default="configs/segmentation_usb.yaml")
    parser.add_argument("--joint-config", default="configs/joint.yaml")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--sample-view-count", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    input_root = (script_dir / args.input_root).resolve()
    manifest_json = (script_dir / args.manifest_json).resolve()
    out_root = (script_dir / args.out_root).resolve()
    objects_root = out_root / "objects"
    cache_dir = out_root / ".cache"
    out_root.mkdir(parents=True, exist_ok=True)
    objects_root.mkdir(parents=True, exist_ok=True)
    _ensure_matplotlib_cache(out_root)

    gt_map = _manifest_gt_map(manifest_json)
    rows: list[dict[str, Any]] = []

    for dataset_dir in _object_dirs(input_root):
        name = dataset_dir.name
        object_out_dir = objects_root / name
        try:
            status_info = _run_object(
                script_dir=script_dir,
                dataset_dir=dataset_dir,
                object_out_dir=object_out_dir,
                tracker_config=args.tracker_config,
                seg_config=args.seg_config,
                joint_config=args.joint_config,
                max_frames=args.max_frames,
                report_sample_views=args.sample_view_count,
                force=args.force,
                cache_dir=cache_dir,
            )
            row = _collect_result(
                name=name,
                gt_map=gt_map,
                object_out_dir=object_out_dir,
                status=status_info["status"] if status_info["status"] != "skipped" else "success",
                message=status_info["message"],
            )
        except Exception as exc:
            row = _collect_result(
                name=name,
                gt_map=gt_map,
                object_out_dir=object_out_dir,
                status="failed",
                message=str(exc),
            )
        rows.append(row)

    stats = _benchmark_stats(rows)
    _write_summary_csv(rows, out_root / "benchmark_summary.csv")
    (out_root / "benchmark_summary.json").write_text(
        json.dumps({"stats": stats, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    _plot_prediction_counts(rows, out_root / "prediction_counts.png")
    _plot_gt_vs_pred(rows, out_root / "gt_vs_pred.png")
    _plot_loss_by_object(rows, out_root / "loss_by_object.png")
    _write_index_html(out_root, rows=rows, stats=stats)


if __name__ == "__main__":
    main()
