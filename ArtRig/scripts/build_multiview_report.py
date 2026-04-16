#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

from articulation.data import SegmentationResult, load_tracks_npz
from articulation.utils.viz import (
    save_cog_trajectories,
    save_point_label_overlay,
    save_segmentation_mask_preview,
    save_state_vs_time,
)


def _ensure_matplotlib_cache(report_dir: Path) -> None:
    cache = report_dir / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_segmentation(path: Path) -> SegmentationResult:
    payload = torch.load(path, map_location="cpu")
    return SegmentationResult(
        point_logits=payload["point_logits"],
        point_probs=payload["point_probs"],
        point_labels=payload["point_labels"],
        masks_per_frame=payload["masks_per_frame"],
        transforms_part0=payload["transforms_part0"],
        transforms_part1=payload["transforms_part1"],
        diagnostics=payload.get("diagnostics", {}),
    )


def _plot_tracks_and_weights(rows: list[dict[str, Any]], out_path: Path) -> None:
    plt = _plt()
    views = [row["camera"] for row in rows]
    track_counts = [row["num_points"] for row in rows]
    weights = [row["view_weight"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    bars = ax1.bar(views, track_counts, color="#4C78A8", alpha=0.9)
    ax1.set_ylabel("Tracked Points")
    ax1.set_xlabel("Camera")
    ax1.set_title("Per-view Tracks and Fusion Weights")
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(views, weights, color="#E45756", marker="o", linewidth=2)
    ax2.set_ylabel("View Weight")

    for bar, value in zip(bars, track_counts):
        ax1.text(
            bar.get_x() + bar.get_width() * 0.5,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_candidate_heatmap(
    rows: list[dict[str, Any]],
    model_names: list[str],
    out_path: Path,
) -> None:
    plt = _plt()
    data = np.array(
        [
            [float(row["candidate_losses"].get(model, np.nan)) for model in model_names]
            for row in rows
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.55 * len(rows))))
    im = ax.imshow(data, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["camera"] for row in rows])
    ax.set_title("Per-view Candidate Losses")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_model_votes(
    rows: list[dict[str, Any]],
    fused_best_model: str,
    out_path: Path,
) -> None:
    plt = _plt()
    names = sorted({row["best_model"] for row in rows} | {fused_best_model})
    counts = [sum(1 for row in rows if row["best_model"] == name) for name in names]
    colors = ["#F58518" if name == fused_best_model else "#72B7B2" for name in names]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(names, counts, color=colors)
    ax.set_title("Per-view Model Votes")
    ax.set_ylabel("Views")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_per_view_csv(rows: list[dict[str, Any]], out_path: Path, model_names: list[str]) -> None:
    fieldnames = [
        "camera",
        "num_points",
        "valid_ratio_mean",
        "view_weight",
        "best_model",
        "nonfinite_steps",
    ] + [f"loss_{name}" for name in model_names]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "camera": row["camera"],
                "num_points": row["num_points"],
                "valid_ratio_mean": f"{row['valid_ratio_mean']:.6f}",
                "view_weight": f"{row['view_weight']:.6f}",
                "best_model": row["best_model"],
                "nonfinite_steps": row["nonfinite_steps"],
            }
            for name in model_names:
                value = row["candidate_losses"].get(name)
                record[f"loss_{name}"] = "" if value is None else f"{float(value):.6f}"
            writer.writerow(record)


def _write_html(
    report_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    sample_views: list[str],
    model_names: list[str],
) -> None:
    summary_rows = [
        ("Input", summary["input_dir"]),
        ("Views", str(summary["num_views"])),
        ("Frames per View", str(summary["frames_per_view"])),
        ("Fused Best Model", summary["best_model"]),
        ("Fused Loss", f"{summary['loss']:.6f}"),
        ("Axis Dir World", ", ".join(f"{v:.4f}" for v in summary["axis_dir_world"])),
        ("Pitch", "None" if summary["pitch"] is None else f"{float(summary['pitch']):.6f}"),
    ]

    per_view_head = "".join(
        f"<th>{html.escape(name)}</th>" for name in ["camera", "points", "valid", "weight", "best"] + model_names
    )
    per_view_body = []
    for row in rows:
        cols = [
            html.escape(row["camera"]),
            str(row["num_points"]),
            f"{row['valid_ratio_mean']:.3f}",
            f"{row['view_weight']:.3f}",
            html.escape(row["best_model"]),
        ]
        cols.extend(
            "" if row["candidate_losses"].get(name) is None else f"{float(row['candidate_losses'][name]):.3f}"
            for name in model_names
        )
        per_view_body.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cols) + "</tr>")

    sample_blocks = []
    for camera in sample_views:
        sample_blocks.append(
            f"""
            <section class="sample">
              <h3>{html.escape(camera)}</h3>
              <div class="grid">
                <figure><img src="sample_views/{html.escape(camera)}_points_overlay.png" alt="{html.escape(camera)} overlay"><figcaption>Point overlay</figcaption></figure>
                <figure><img src="sample_views/{html.escape(camera)}_masks_preview.png" alt="{html.escape(camera)} masks"><figcaption>Segmentation masks</figcaption></figure>
                <figure><img src="sample_views/{html.escape(camera)}_cog_trajectories.png" alt="{html.escape(camera)} cogs"><figcaption>CoG trajectories</figcaption></figure>
              </div>
            </section>
            """
        )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ArtRig Multiview Report</title>
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
      max-width: 1180px;
      margin: 0 auto;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px 0;
      line-height: 1.15;
    }}
    p {{
      color: var(--muted);
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 8px 24px rgba(35, 35, 35, 0.06);
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
    }}
    th {{
      color: var(--accent);
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .charts figure, .sample figure {{
      margin: 0;
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
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .mono {{
      font-family: "SFMono-Regular", ui-monospace, Menlo, monospace;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>ArtRig Multiview Report</h1>
      <p class="mono">{html.escape(str(summary["run_dir"]))}</p>
      <div class="meta">
        <table>
          <tbody>
            {"".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in summary_rows)}
          </tbody>
        </table>
        <div>
          <p>This report focuses on the fused motion estimate, per-view quality, and two representative views.</p>
          <p>Raw files are available one level up in the run folder.</p>
        </div>
      </div>
    </section>

    <section>
      <h2>Charts</h2>
      <div class="charts">
        <figure><img src="fused_state.png" alt="Fused state"><figcaption>Fused joint state over time</figcaption></figure>
        <figure><img src="per_view_tracks_weights.png" alt="Tracks and weights"><figcaption>Track counts and fusion weights per camera</figcaption></figure>
        <figure><img src="per_view_candidate_losses.png" alt="Candidate losses"><figcaption>Relative fit loss of revolute, prismatic, and screw per camera</figcaption></figure>
        <figure><img src="per_view_model_votes.png" alt="Model votes"><figcaption>How many views prefer each model</figcaption></figure>
      </div>
    </section>

    <section>
      <h2>Per-view Table</h2>
      <table>
        <thead><tr>{per_view_head}</tr></thead>
        <tbody>
          {"".join(per_view_body)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Sample Views</h2>
      {"".join(sample_blocks)}
    </section>
  </main>
</body>
</html>
"""
    (report_dir / "index.html").write_text(html_text, encoding="utf-8")


def _report_rows(run_dir: Path, fused: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    model_names = sorted(
        {
            model
            for payload in (fused.get("per_view") or {}).values()
            for model in (payload.get("candidate_losses") or {}).keys()
        }
    )

    for camera, payload in sorted((fused.get("per_view") or {}).items()):
        tracks = load_tracks_npz(run_dir / "views" / camera / "tracks.npz")
        rows.append(
            {
                "camera": camera,
                "num_points": int(payload.get("num_points", int(tracks.xy.shape[0]))),
                "valid_ratio_mean": float(tracks.valid.float().mean()),
                "view_weight": float(payload.get("view_weight", 0.0)),
                "best_model": str(payload.get("best_model", "")),
                "candidate_losses": dict(payload.get("candidate_losses") or {}),
                "nonfinite_steps": len(payload.get("nonfinite_steps") or []),
            }
        )
    return rows, model_names


def _sample_views(rows: list[dict[str, Any]], count: int) -> list[str]:
    chosen = sorted(rows, key=lambda row: (-row["view_weight"], row["camera"]))
    return [row["camera"] for row in chosen[: max(1, count)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="ArtRig multiview output folder")
    parser.add_argument("--report-dir", default=None, help="Default: <run-dir>/report")
    parser.add_argument("--sample-view-count", type=int, default=2)
    parser.add_argument("--frame-idx", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    _ensure_matplotlib_cache(report_dir)

    fused = _load_json(run_dir / "joint_fused_world.json")
    meta = _load_json(run_dir / "multiview_run_meta.json")
    rows, model_names = _report_rows(run_dir, fused)

    summary = {
        "run_dir": str(run_dir),
        "input_dir": meta.get("input_dir", ""),
        "num_views": int(meta.get("num_views", len(rows))),
        "frames_per_view": meta.get("frames_per_view", {}),
        "best_model": str(fused.get("best_model", "")),
        "loss": float(fused.get("loss", 0.0)),
        "axis_dir_world": list(fused.get("axis_dir_world") or []),
        "axis_point_world": fused.get("axis_point_world"),
        "pitch": fused.get("pitch"),
        "type_priors": fused.get("type_priors", {}),
    }

    sample_views = _sample_views(rows, count=args.sample_view_count)
    summary["sample_views"] = sample_views

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_per_view_csv(rows, report_dir / "per_view_summary.csv", model_names=model_names)

    save_state_vs_time(torch.tensor(fused.get("state") or []), report_dir / "fused_state.png")
    _plot_tracks_and_weights(rows, report_dir / "per_view_tracks_weights.png")
    _plot_candidate_heatmap(rows, model_names=model_names, out_path=report_dir / "per_view_candidate_losses.png")
    _plot_model_votes(rows, fused_best_model=summary["best_model"], out_path=report_dir / "per_view_model_votes.png")

    sample_dir = report_dir / "sample_views"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for camera in sample_views:
        tracks = load_tracks_npz(run_dir / "views" / camera / "tracks.npz")
        seg = _load_segmentation(run_dir / "views" / camera / "segmentation.pt")
        save_point_label_overlay(tracks, seg, sample_dir / f"{camera}_points_overlay.png", frame_idx=args.frame_idx)
        save_segmentation_mask_preview(seg.masks_per_frame, sample_dir / f"{camera}_masks_preview.png", frame_idx=args.frame_idx)
        save_cog_trajectories(tracks, seg, sample_dir / f"{camera}_cog_trajectories.png")

    _write_html(report_dir, summary=summary, rows=rows, sample_views=sample_views, model_names=model_names)


if __name__ == "__main__":
    main()
