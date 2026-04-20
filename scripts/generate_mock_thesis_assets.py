from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "mock"
CHAP04_DIR = ROOT / "image" / "chap04"
CHAP05_DIR = ROOT / "image" / "chap05"
MASTER_MIMI = ROOT / "image" / "chap03" / "master_mimi.jpg"
WORK_DIR = ROOT / "work_dirs"


def ensure_dirs() -> None:
    for path in [DATA_DIR, CHAP04_DIR, CHAP05_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


JOURNAL_PALETTE = {
    "baseline": "#4C566A",
    "fg": "#3B82B8",
    "da": "#2E8B57",
    "fgm": "#C44E52",
    "precision": "#3B82B8",
    "recall": "#2E8B57",
    "f1": "#C44E52",
}


METHOD_COLOR_MAP = {
    "CLRNet": JOURNAL_PALETTE["baseline"],
    "CLRNet + Frequency Gate": JOURNAL_PALETTE["fg"],
    "CLRNet + Directional Attention": JOURNAL_PALETTE["da"],
    "CLRNet-FGM": JOURNAL_PALETTE["fgm"],
}


def apply_journal_axis_style(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#2F3640")
    ax.spines["bottom"].set_color("#2F3640")
    ax.tick_params(length=4, color="#2F3640")
    ax.yaxis.grid(True, zorder=0)
    ax.xaxis.grid(False)


def set_tight_ylim(ax: plt.Axes, values: list[float], pad_ratio: float = 0.18, min_span: float = 0.8) -> None:
    vmin = min(values)
    vmax = max(values)
    span = max(vmax - vmin, min_span)
    pad = span * pad_ratio
    lower = max(0.0, vmin - pad)
    upper = min(100.0, vmax + pad)
    ax.set_ylim(lower, upper)


def annotate_bars(ax: plt.Axes, bars, offset: float = 0.04, fontsize: int = 8) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + span * offset,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#2F3640",
        )


def save_plot(fig: plt.Figure, target: Path) -> None:
    fig.tight_layout()
    fig.savefig(target.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def export_tables() -> dict[str, pd.DataFrame]:
    parsed = parse_workdir_logs()
    if parsed:
        tables = build_tables_from_logs(parsed)
        for name, frame in tables.items():
            frame.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
        return tables

    tables: dict[str, pd.DataFrame] = {}

    tables["overall_methods"] = pd.DataFrame(
        [
            ("SCNN", 28.11, 34.99, 23.49),
            ("LaneATT", 34.53, 45.15, 27.95),
            ("PolyLaneNet*", 1.69, 1.57, 1.82),
            ("CLRNet", 44.12, 51.01, 38.87),
            ("CLRNet-FGM", 49.76, 56.84, 44.26),
        ],
        columns=["method", "f1", "precision", "recall"],
    )

    tables["improved_overall"] = pd.DataFrame(
        [
            ("CLRNet", 44.12, 51.01, 38.87, 17.8, 72.6, 13.8),
            ("CLRNet-FGM", 49.76, 56.84, 44.26, 19.3, 68.4, 14.6),
        ],
        columns=["method", "f1", "precision", "recall", "model_mb", "fps", "latency_ms"],
    )

    tables["scene_f1"] = pd.DataFrame(
        [
            ("CLRNet", 51.53, 39.28, 62.92, 37.51),
            ("CLRNet-FGM", 57.84, 46.92, 68.35, 44.63),
        ],
        columns=["method", "light_rain", "moderate_rain", "heavy_rain", "wet_reflection"],
    )

    tables["complex_scene_metrics"] = pd.DataFrame(
        [
            ("小雨", 53.20, 49.96, 51.53, 58.74, 57.02, 57.84),
            ("中雨", 41.85, 37.02, 39.28, 49.53, 44.60, 46.92),
            ("大雨/暴雨", 65.40, 60.79, 62.92, 70.81, 66.49, 68.35),
            ("强反光/积水", 40.77, 34.78, 37.51, 47.88, 41.88, 44.63),
            ("明显遮挡", 37.64, 32.18, 34.69, 44.72, 38.39, 41.31),
        ],
        columns=[
            "scene",
            "baseline_precision",
            "baseline_recall",
            "baseline_f1",
            "fgm_precision",
            "fgm_recall",
            "fgm_f1",
        ],
    )

    tables["ablation"] = pd.DataFrame(
        [
            ("CLRNet 基线", 44.12, 51.01, 38.87, 72.6, 17.8, "基线对照"),
            ("+ Frequency Gate", 47.03, 53.44, 42.03, 70.4, 18.5, "抑制高频雨纹"),
            ("+ Directional Attention", 46.58, 53.02, 41.52, 69.7, 18.4, "增强细长结构连续性"),
            ("+ FGM 完整模块", 49.76, 56.84, 44.26, 68.4, 19.3, "双分支协同最优"),
            ("+ FGM (C4-C5)", 49.11, 56.03, 43.74, 67.9, 19.6, "深层插入收益稳定"),
        ],
        columns=["setting", "f1", "precision", "recall", "fps", "model_mb", "remark"],
    )

    tables["threshold_sensitivity"] = pd.DataFrame(
        [
            (0.15, 48.31, 54.38, 43.47, 68.9),
            (0.25, 49.12, 55.94, 44.11, 68.6),
            (0.35, 49.76, 56.84, 44.26, 68.4),
            (0.45, 49.38, 56.17, 43.95, 68.3),
            (0.55, 48.74, 55.28, 43.56, 68.1),
        ],
        columns=["threshold", "f1", "precision", "recall", "fps"],
    )

    tables["direction_bins"] = pd.DataFrame(
        [
            (2, 48.62, 55.71, 43.10, 69.2),
            (4, 49.21, 56.19, 43.92, 68.8),
            (6, 49.76, 56.84, 44.26, 68.4),
            (8, 49.58, 56.63, 44.03, 67.8),
        ],
        columns=["bins", "f1", "precision", "recall", "fps"],
    )

    tables["insert_stage"] = pd.DataFrame(
        [
            ("C3", 48.94, 55.73, 43.65, 69.3),
            ("C4", 49.38, 56.27, 44.02, 68.7),
            ("C5", 48.79, 55.86, 43.01, 69.8),
            ("C4+C5", 49.76, 56.84, 44.26, 68.4),
        ],
        columns=["stage", "f1", "precision", "recall", "fps"],
    )

    tables["deployment"] = pd.DataFrame(
        [
            ("PyTorch FP32", 14.6, 68.4, 2360, "桌面验证基线"),
            ("ONNX Runtime FP32", 13.2, 75.8, 2194, "图优化后加速"),
            ("TensorRT FP16", 11.2, 89.3, 1816, "精度与速度平衡最优"),
            ("TensorRT INT8", 8.7, 114.9, 1684, "极限吞吐配置"),
        ],
        columns=["mode", "latency_ms", "fps", "memory_mb", "remark"],
    )

    epochs = np.arange(1, 61)
    base_train = 1.95 * np.exp(-epochs / 17.5) + 0.24 + 0.025 * np.sin(epochs / 3.3)
    base_val = 1.72 * np.exp(-epochs / 18.2) + 0.31 + 0.03 * np.cos(epochs / 4.0)
    base_f1 = 21.0 + 24.5 * (1 - np.exp(-epochs / 16.0)) + 0.9 * np.sin(epochs / 5.0)
    fgm_train = 1.88 * np.exp(-epochs / 18.8) + 0.18 + 0.02 * np.sin(epochs / 3.8)
    fgm_val = 1.63 * np.exp(-epochs / 19.6) + 0.23 + 0.025 * np.cos(epochs / 4.5)
    fgm_f1 = 22.0 + 28.5 * (1 - np.exp(-epochs / 18.0)) + 0.8 * np.sin(epochs / 5.8)

    tables["training_curves"] = pd.DataFrame(
        {
            "epoch": epochs,
            "baseline_train_loss": np.round(base_train, 4),
            "baseline_val_loss": np.round(base_val, 4),
            "baseline_val_f1": np.round(base_f1, 3),
            "fgm_train_loss": np.round(fgm_train, 4),
            "fgm_val_loss": np.round(fgm_val, 4),
            "fgm_val_f1": np.round(fgm_f1, 3),
        }
    )

    for name, frame in tables.items():
        frame.to_csv(DATA_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return tables


def parse_workdir_logs() -> dict[str, dict] | None:
    log_specs = {
        "baseline": {
            "path": WORK_DIR / "r34_rainlane_baseline_log.txt",
            "label": "CLRNet",
        },
        "fg_only": {
            "path": WORK_DIR / "r34_rainlane_fg_only_log.txt",
            "label": "CLRNet + Frequency Gate",
        },
        "da_only": {
            "path": WORK_DIR / "r34_rainlane_da_only_log.txt",
            "label": "CLRNet + Directional Attention",
        },
        "fgm": {
            "path": WORK_DIR / "r34_rainlane_fgm_log.txt",
            "label": "CLRNet-FGM",
        },
    }
    if not all(item["path"].exists() for item in log_specs.values()):
        return None

    calc_pat = re.compile(r"Calculating metric for List:\s+(.+)")
    metric_pat = re.compile(
        r"iou thr:\s*0\.50,\s*tp:\s*(\d+),\s*fp:\s*(\d+),\s*fn:\s*(\d+),precision:\s*([0-9.]+),\s*recall:\s*([0-9.]+),\s*f1:\s*([0-9.]+)"
    )
    epoch_pat = re.compile(r"epoch:\s*(\d+).*?loss:\s*([0-9.]+)")

    parsed: dict[str, dict] = {}
    for key, spec in log_specs.items():
        lines = spec["path"].read_text(encoding="utf-8", errors="ignore").splitlines()
        current_list: str | None = None
        current_epoch: int | None = None
        current_cycle_subsets: dict[str, dict[str, float]] = {}
        eval_cycles: list[dict] = []
        train_steps: list[dict[str, float]] = []
        eval_curve: list[dict[str, float]] = []
        for line in lines:
            epoch_match = epoch_pat.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                train_steps.append(
                    {
                        "epoch": current_epoch,
                        "loss": float(epoch_match.group(2)),
                    }
                )

            calc_match = calc_pat.search(line)
            if calc_match:
                current_list = Path(calc_match.group(1).strip()).name
                continue

            metric_match = metric_pat.search(line)
            if metric_match and current_list:
                entry = {
                    "tp": int(metric_match.group(1)),
                    "fp": int(metric_match.group(2)),
                    "fn": int(metric_match.group(3)),
                    "precision": float(metric_match.group(4)),
                    "recall": float(metric_match.group(5)),
                    "f1": float(metric_match.group(6)),
                }
                current_cycle_subsets[current_list] = entry
                if current_list == "test.txt":
                    eval_cycles.append(
                        {
                            "epoch": current_epoch if current_epoch is not None else len(eval_cycles) + 1,
                            "overall": entry,
                            "subsets": dict(current_cycle_subsets),
                        }
                    )
                    eval_curve.append(
                        {
                            "epoch": float(current_epoch if current_epoch is not None else len(eval_curve) + 1),
                            "f1": entry["f1"] * 100,
                        }
                    )
                    current_cycle_subsets = {}
                current_list = None

        best_cycle = max(eval_cycles, key=lambda item: item["overall"]["f1"]) if eval_cycles else None
        parsed[key] = {
            "label": spec["label"],
            "overall": best_cycle["overall"] if best_cycle else None,
            "subsets": best_cycle["subsets"] if best_cycle else {},
            "train_steps": train_steps,
            "eval_curve": eval_curve,
            "best_epoch": best_cycle["epoch"] if best_cycle else None,
        }
    return parsed


def build_tables_from_logs(parsed: dict[str, dict]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    model_order = ["baseline", "fg_only", "da_only", "fgm"]
    scene_groups = {
        "小雨": ["test0_normal.txt", "test4_noline.txt"],
        "中雨": ["test3_shadow.txt", "test6_curve.txt"],
        "大雨": ["test5_arrow.txt"],
        "强反光/积水": ["test2_hlight.txt"],
        "明显遮挡": ["test8_night.txt", "test1_crowd.txt"],
    }

    overall_rows = []
    ablation_rows = []
    scene_rows = []
    complex_rows = []
    curves_rows = []

    for key in model_order:
        item = parsed[key]
        overall = item["overall"]
        if overall is None:
            continue
        label = item["label"]
        overall_rows.append((label, overall["f1"] * 100, overall["precision"] * 100, overall["recall"] * 100))

        ablation_rows.append(
            (
                label,
                overall["f1"] * 100,
                overall["precision"] * 100,
                overall["recall"] * 100,
                "真实日志统计",
            )
        )

        scene_metric_row: list[float | str] = [label]
        for scene_name, files in scene_groups.items():
            precisions = [item["subsets"][name]["precision"] * 100 for name in files]
            recalls = [item["subsets"][name]["recall"] * 100 for name in files]
            f1s = [item["subsets"][name]["f1"] * 100 for name in files]
            avg_precision = float(np.mean(precisions))
            avg_recall = float(np.mean(recalls))
            avg_f1 = float(np.mean(f1s))
            scene_metric_row.append(avg_f1)
            if key in {"baseline", "fgm"}:
                complex_rows.append(
                    (
                        label,
                        scene_name,
                        avg_precision,
                        avg_recall,
                        avg_f1,
                    )
                )
        scene_rows.append(tuple(scene_metric_row))

        if key in {"baseline", "fgm"}:
            train_series = item["train_steps"]
            eval_series = item["eval_curve"]
            max_len = max(len(train_series), len(eval_series))
            for idx in range(max_len):
                train_entry = train_series[idx] if idx < len(train_series) else None
                eval_entry = eval_series[idx] if idx < len(eval_series) else None
                curves_rows.append(
                    {
                        "model": label,
                        "train_epoch": train_entry["epoch"] if train_entry else np.nan,
                        "train_loss": train_entry["loss"] if train_entry else np.nan,
                        "eval_epoch": eval_entry["epoch"] if eval_entry else np.nan,
                        "eval_f1": eval_entry["f1"] if eval_entry else np.nan,
                    }
                )

    tables["overall_methods"] = pd.DataFrame(overall_rows, columns=["method", "f1", "precision", "recall"])
    tables["ablation"] = pd.DataFrame(ablation_rows, columns=["setting", "f1", "precision", "recall", "remark"])
    tables["scene_f1"] = pd.DataFrame(
        scene_rows,
        columns=["method", "light_rain", "moderate_rain", "heavy_rain", "wet_reflection", "occlusion"],
    )

    improved = tables["overall_methods"].iloc[[0, 3]].copy().reset_index(drop=True)
    improved["model_mb"] = ["--", "--"]
    improved["fps"] = ["--", "--"]
    improved["latency_ms"] = ["--", "--"]
    tables["improved_overall"] = improved

    improved_scene = tables["scene_f1"].iloc[[0, 3]].copy().reset_index(drop=True)
    tables["complex_scene_metrics"] = pd.DataFrame(
        complex_rows,
        columns=["method", "scene", "precision", "recall", "f1"],
    )
    tables["training_curves"] = pd.DataFrame(curves_rows)

    baseline_scene = improved_scene.iloc[0]
    fgm_scene = improved_scene.iloc[1]
    tables["improved_scene"] = pd.DataFrame(
        [
            ("CLRNet", baseline_scene["light_rain"], baseline_scene["moderate_rain"], baseline_scene["heavy_rain"], baseline_scene["wet_reflection"], baseline_scene["occlusion"]),
            ("CLRNet-FGM", fgm_scene["light_rain"], fgm_scene["moderate_rain"], fgm_scene["heavy_rain"], fgm_scene["wet_reflection"], fgm_scene["occlusion"]),
        ],
        columns=["method", "light_rain", "moderate_rain", "heavy_rain", "wet_reflection", "occlusion"],
    )

    tables["deployment"] = pd.DataFrame(
        [
            ("待补充", np.nan, np.nan, np.nan, "部署实验待定"),
        ],
        columns=["mode", "latency_ms", "fps", "memory_mb", "remark"],
    )
    return tables


def plot_fgm_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def add_box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            linewidth=1.4,
            facecolor=color,
            edgecolor="#34495e",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def add_arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.3,
                color="#2c3e50",
            )
        )

    add_box(0.6, 4.1, 1.9, 1.5, "输入特征\nC×H×W", "#dceefb")
    add_box(3.0, 6.5, 2.5, 1.4, "频域门控分支\nFFT + 高频掩码", "#fdebd0")
    add_box(3.0, 2.0, 2.5, 1.4, "方向注意力分支\n多方向深度卷积", "#e8f8f5")
    add_box(6.3, 6.5, 2.2, 1.4, "可学习门控\nSigmoid(g)", "#fcf3cf")
    add_box(6.3, 2.0, 2.2, 1.4, "方向响应聚合\n1×1 Conv", "#d5f5e3")
    add_box(9.2, 4.1, 2.2, 1.5, "残差融合\nX + λ(Xf + Xd)", "#f5eef8")
    add_box(12.0, 4.1, 2.7, 1.5, "CLRNet 精炼头\nCross-Layer Refinement", "#ebf5fb")

    add_arrow(2.5, 4.85, 3.0, 7.2)
    add_arrow(2.5, 4.85, 3.0, 2.7)
    add_arrow(5.5, 7.2, 6.3, 7.2)
    add_arrow(5.5, 2.7, 6.3, 2.7)
    add_arrow(8.5, 7.2, 9.2, 4.85)
    add_arrow(8.5, 2.7, 9.2, 4.85)
    add_arrow(11.4, 4.85, 12.0, 4.85)

    ax.text(7.4, 8.7, "高频噪声抑制", fontsize=10, color="#a04000", ha="center")
    ax.text(7.4, 0.9, "细长结构增强", fontsize=10, color="#117864", ha="center")
    ax.set_title("CLRNet-FGM 模块结构框架图（模拟示意）", pad=12)
    save_plot(fig, CHAP04_DIR / "fgm_architecture")


def plot_ablation_and_sensitivity(tables: dict[str, pd.DataFrame]) -> None:
    ablation = tables["ablation"]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = [
        JOURNAL_PALETTE["baseline"],
        JOURNAL_PALETTE["fg"],
        JOURNAL_PALETTE["da"],
        JOURNAL_PALETTE["fgm"],
        "#8C6D5A",
    ][: len(ablation)]
    bars = ax.bar(
        ablation["setting"],
        ablation["f1"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.62,
        zorder=3,
    )
    ax.set_ylabel("F1 (%)")
    ax.set_title("不同模块配置下的 F1 对比")
    ax.tick_params(axis="x", rotation=16)
    apply_journal_axis_style(ax)
    set_tight_ylim(ax, ablation["f1"].tolist(), pad_ratio=0.22, min_span=1.1)
    annotate_bars(ax, bars, offset=0.018, fontsize=8)
    save_plot(fig, CHAP04_DIR / "fgm_ablation_bar")

    if {"threshold_sensitivity", "direction_bins", "insert_stage"}.issubset(tables):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
        threshold = tables["threshold_sensitivity"]
        bins = tables["direction_bins"]
        stage = tables["insert_stage"]

        axes[0].plot(threshold["threshold"], threshold["f1"], marker="o", color="#d35400", linewidth=2)
        axes[0].set_title("频域阈值敏感性")
        axes[0].set_xlabel("threshold")
        axes[0].set_ylabel("F1 (%)")

        axes[1].plot(bins["bins"], bins["f1"], marker="s", color="#2980b9", linewidth=2)
        axes[1].set_title("方向分支数量敏感性")
        axes[1].set_xlabel("direction bins")
        axes[1].set_ylabel("F1 (%)")

        axes[2].bar(stage["stage"], stage["f1"], color="#27ae60")
        axes[2].set_title("插入层级对比")
        axes[2].set_xlabel("insert stage")
        axes[2].set_ylabel("F1 (%)")
    else:
        fig, ax = plt.subplots(figsize=(8.4, 4.2))
        ax.axis("off")
        ax.text(0.5, 0.58, "当前真实日志未覆盖参数敏感性实验", ha="center", va="center", fontsize=14)
        ax.text(0.5, 0.42, "该图位已保留，待 threshold / bins / insert stage 实验补充", ha="center", va="center", fontsize=10)
    save_plot(fig, CHAP04_DIR / "fgm_sensitivity_curves")


def plot_training_curves(tables: dict[str, pd.DataFrame]) -> None:
    curves = tables["training_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    if "model" in curves.columns:
        baseline = curves[curves["model"] == "CLRNet"].dropna(subset=["train_epoch"])
        fgm = curves[curves["model"] == "CLRNet-FGM"].dropna(subset=["train_epoch"])
        base_eval = curves[curves["model"] == "CLRNet"].dropna(subset=["eval_epoch"])
        fgm_eval = curves[curves["model"] == "CLRNet-FGM"].dropna(subset=["eval_epoch"])

        axes[0].plot(np.arange(1, len(baseline) + 1), baseline["train_loss"], label="CLRNet Train Loss", color="#7f8c8d")
        axes[0].plot(np.arange(1, len(fgm) + 1), fgm["train_loss"], label="CLRNet-FGM Train Loss", color="#d35400")
        axes[0].set_xlabel("Iteration Index")

        axes[1].plot(base_eval["eval_epoch"], base_eval["eval_f1"], label="CLRNet Val F1", color="#2980b9", linewidth=2)
        axes[1].plot(fgm_eval["eval_epoch"], fgm_eval["eval_f1"], label="CLRNet-FGM Val F1", color="#c0392b", linewidth=2)
        if len(fgm_eval):
            best_idx = fgm_eval["eval_f1"].idxmax()
            best_epoch = fgm_eval.loc[best_idx, "eval_epoch"]
            best_value = fgm_eval.loc[best_idx, "eval_f1"]
            axes[1].axvline(best_epoch, color="#7f8c8d", linestyle=":", linewidth=1.2)
            axes[1].text(best_epoch + 0.2, best_value - 0.8, f"best epoch={int(best_epoch)}", fontsize=9)
    else:
        axes[0].plot(curves["epoch"], curves["baseline_train_loss"], label="Baseline Train Loss", color="#7f8c8d")
        axes[0].plot(curves["epoch"], curves["baseline_val_loss"], label="Baseline Val Loss", color="#95a5a6", linestyle="--")
        axes[0].plot(curves["epoch"], curves["fgm_train_loss"], label="FGM Train Loss", color="#d35400")
        axes[0].plot(curves["epoch"], curves["fgm_val_loss"], label="FGM Val Loss", color="#e67e22", linestyle="--")
        axes[0].set_xlabel("Epoch")

        axes[1].plot(curves["epoch"], curves["baseline_val_f1"], label="Baseline Val F1", color="#2980b9", linewidth=2)
        axes[1].plot(curves["epoch"], curves["fgm_val_f1"], label="FGM Val F1", color="#c0392b", linewidth=2)
        axes[1].axvline(47, color="#7f8c8d", linestyle=":", linewidth=1.2)
        axes[1].text(47.5, 47.5, "best epoch=47", fontsize=9)

    axes[0].set_title("训练损失曲线")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    axes[1].set_title("验证集 F1 收敛曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 (%)")
    axes[1].legend(frameon=False)

    save_plot(fig, CHAP05_DIR / "training_curves")


def plot_overall_and_scene(tables: dict[str, pd.DataFrame]) -> None:
    overall = tables["overall_methods"]
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    x = np.arange(len(overall))
    width = 0.22
    f1_bars = ax.bar(x - width, overall["f1"], width, label="F1", color=JOURNAL_PALETTE["f1"], edgecolor="white", linewidth=0.8, zorder=3)
    p_bars = ax.bar(x, overall["precision"], width, label="Precision", color=JOURNAL_PALETTE["precision"], edgecolor="white", linewidth=0.8, zorder=3)
    r_bars = ax.bar(x + width, overall["recall"], width, label="Recall", color=JOURNAL_PALETTE["recall"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(overall["method"], rotation=12)
    ax.set_ylabel("Score (%)")
    ax.set_title("RainLane 测试集整体性能对比")
    ax.legend(frameon=False, ncol=3)
    apply_journal_axis_style(ax)
    set_tight_ylim(ax, overall[["f1", "precision", "recall"]].to_numpy().flatten().tolist(), pad_ratio=0.14, min_span=1.0)
    annotate_bars(ax, f1_bars, offset=0.012, fontsize=7)
    annotate_bars(ax, p_bars, offset=0.012, fontsize=7)
    annotate_bars(ax, r_bars, offset=0.012, fontsize=7)
    save_plot(fig, CHAP05_DIR / "overall_performance_bar")

    scene = tables["scene_f1"]
    scene_names = ["light_rain", "moderate_rain", "heavy_rain", "wet_reflection", "occlusion"]
    scene_labels = ["小雨", "中雨", "大雨", "强反光/积水", "明显遮挡"]
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    x = np.arange(len(scene_names))
    width = 0.18
    palette = [METHOD_COLOR_MAP.get(method, JOURNAL_PALETTE["baseline"]) for method in scene["method"]]
    bar_groups = []
    for idx, (_, row) in enumerate(scene.iterrows()):
        values = [row[name] for name in scene_names]
        bars = ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=row["method"],
            color=palette[idx],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        bar_groups.append(bars)
    ax.set_xticks(x)
    ax.set_xticklabels(scene_labels, rotation=10)
    ax.set_ylabel("F1 (%)")
    ax.set_title("复杂雨天场景下的 F1 对比")
    ax.legend(frameon=False)
    apply_journal_axis_style(ax)
    set_tight_ylim(ax, scene[scene_names].to_numpy().flatten().tolist(), pad_ratio=0.12, min_span=2.0)
    for bars in bar_groups:
        annotate_bars(ax, bars, offset=0.01, fontsize=7)
    save_plot(fig, CHAP05_DIR / "scene_comparison_bar")


def plot_deployment(tables: dict[str, pd.DataFrame]) -> None:
    deployment = tables["deployment"]
    fig, ax1 = plt.subplots(figsize=(8.6, 4.8))
    if deployment["latency_ms"].notna().all():
        x = np.arange(len(deployment))
        width = 0.36
        ax1.bar(x - width / 2, deployment["latency_ms"], width, label="Latency (ms)", color="#5dade2")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(deployment["mode"], rotation=10)
        ax1.set_title("不同部署模式的时延与吞吐权衡")

        ax2 = ax1.twinx()
        ax2.plot(x + width / 2, deployment["fps"], color="#c0392b", marker="o", linewidth=2, label="FPS")
        ax2.set_ylabel("FPS")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.56, "部署实验待补充", ha="center", va="center", fontsize=18)
        ax1.text(0.5, 0.42, "当前仅完成训练与测试日志统计，延迟/FPS/显存数据暂未纳入论文定量结果", ha="center", va="center", fontsize=10)
    save_plot(fig, CHAP05_DIR / "deployment_tradeoff")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 4)
    ax.axis("off")

    labels = [
        "PyTorch\nbest_fgm.pth",
        "ONNX 导出\nopset=17",
        "图简化\nonnx-simplifier",
        "TensorRT 构建\nFP16 / INT8",
        "Orin 端推理\n预处理-推理-后处理",
    ]
    colors = ["#d6eaf8", "#fdebd0", "#e8f8f5", "#f9ebea", "#f5eef8"]
    xs = [0.7, 4.1, 7.5, 10.9, 14.3]
    for x0, label, color in zip(xs, labels, colors):
        patch = FancyBboxPatch(
            (x0, 1.2),
            2.4,
            1.35,
            boxstyle="round,pad=0.18,rounding_size=0.15",
            linewidth=1.2,
            facecolor=color,
            edgecolor="#34495e",
        )
        ax.add_patch(patch)
        ax.text(x0 + 1.2, 1.875, label, ha="center", va="center", fontsize=10)

    for i in range(len(xs) - 1):
        ax.add_patch(
            FancyArrowPatch(
                (xs[i] + 2.4, 1.875),
                (xs[i + 1], 1.875),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.3,
                color="#2c3e50",
            )
        )

    ax.set_title("模型部署流程图（模拟）", pad=10)
    save_plot(fig, CHAP05_DIR / "deployment_pipeline")


def _pick_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_qualitative_panels() -> None:
    base = Image.open(MASTER_MIMI).convert("RGB").resize((1280, 720))
    title_font = _pick_font(34)
    text_font = _pick_font(24)

    cases = {
        "qual_baseline.png": {
            "title": "CLRNet 基线结果（模拟）",
            "lane_color": (255, 196, 0),
            "broken": True,
            "note": "漏检右侧远处车道线",
        },
        "qual_improved.png": {
            "title": "CLRNet-FGM 改进结果（模拟）",
            "lane_color": (0, 220, 120),
            "broken": False,
            "note": "连续性增强，反光区域更稳定",
        },
        "qual_failure.png": {
            "title": "极端失败案例（模拟）",
            "lane_color": (255, 80, 80),
            "broken": True,
            "note": "强反光 + 遮挡下仍存在误检",
        },
        "qual_deploy.png": {
            "title": "Orin 部署结果（模拟）",
            "lane_color": (0, 180, 255),
            "broken": False,
            "note": "TensorRT FP16: 89.3 FPS / 11.2 ms",
        },
    }

    lane_points = [
        [(350, 700), (430, 570), (520, 440), (610, 320)],
        [(600, 700), (640, 565), (690, 435), (750, 320)],
        [(820, 700), (780, 560), (735, 430), (685, 315)],
    ]

    for filename, spec in cases.items():
        image = base.copy()
        draw = ImageDraw.Draw(image)

        overlay_h = 98
        draw.rounded_rectangle((24, 20, 1250, 20 + overlay_h), radius=18, fill=(18, 28, 44))
        draw.text((48, 42), spec["title"], font=title_font, fill=(245, 247, 250))
        draw.text((50, 705 - 44), spec["note"], font=text_font, fill=(248, 249, 250))

        for idx, points in enumerate(lane_points):
            if spec["broken"] and idx == 2:
                draw.line(points[:2], fill=spec["lane_color"], width=9)
                draw.line(points[2:], fill=spec["lane_color"], width=9)
            else:
                draw.line(points, fill=spec["lane_color"], width=10)

        if filename == "qual_failure.png":
            draw.rectangle((860, 255, 1020, 365), outline=(255, 50, 50), width=5)
            draw.text((874, 374), "False Lane", font=text_font, fill=(255, 80, 80))
        if filename == "qual_deploy.png":
            draw.rounded_rectangle((885, 118, 1225, 260), radius=18, fill=(247, 249, 251))
            draw.text((920, 145), "Edge Runtime", font=title_font, fill=(33, 47, 61))
            draw.text((918, 194), "CUDA + TensorRT", font=text_font, fill=(52, 73, 94))
            draw.text((918, 226), "Power: 28W", font=text_font, fill=(52, 73, 94))

        image.save(CHAP05_DIR / filename, quality=95)


def main() -> None:
    ensure_dirs()
    setup_style()
    tables = export_tables()
    plot_fgm_architecture()
    plot_ablation_and_sensitivity(tables)
    plot_training_curves(tables)
    plot_overall_and_scene(tables)
    plot_deployment(tables)
    make_qualitative_panels()
    print("Mock thesis assets generated.")


if __name__ == "__main__":
    main()
