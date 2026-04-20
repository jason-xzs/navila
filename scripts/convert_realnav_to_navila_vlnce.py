#!/usr/bin/env python3
import argparse
import json
import math
import shutil
from pathlib import Path


def wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_trajectory_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            frame_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            yaw = float(parts[3])
            rows.append((frame_id, x, y, yaw))
    rows.sort(key=lambda x: x[0])
    return rows


def find_trajectory_file(session_dir: Path) -> Path:
    files = sorted(session_dir.glob("trajectory_*.txt"))
    if not files:
        raise FileNotFoundError(f"No trajectory file found in {session_dir}")
    return files[0]


def find_image_path(images_dir: Path, frame_id: int) -> Path:
    return images_dir / f"{frame_id:06d}.jpg"


def to_local_delta(x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float):
    dx_w = x1 - x0
    dy_w = y1 - y0
    d_forward = math.cos(yaw0) * dx_w + math.sin(yaw0) * dy_w
    d_lateral = -math.sin(yaw0) * dx_w + math.cos(yaw0) * dy_w
    d_yaw = wrap_to_pi(yaw1 - yaw0)
    return d_forward, d_lateral, d_yaw


def action_text_from_delta(d_forward: float, d_lateral: float, d_yaw_rad: float, stop_dist: float, turn_deg: float):
    d_yaw_deg = math.degrees(d_yaw_rad)

    if abs(d_forward) < stop_dist and abs(d_yaw_deg) < turn_deg:
        action = "Stop."
    elif abs(d_yaw_deg) >= turn_deg and abs(d_forward) < 0.18:
        if d_yaw_deg > 0:
            action = "Turn left 15 degrees."
        else:
            action = "Turn right 15 degrees."
    else:
        # Clamp to positive forward action style used by NaVILA instructions.
        meters = max(0.1, min(0.5, round(d_forward / 0.05) * 0.05))
        action = f"Move forward {meters:.2f} meters."

    return action, d_yaw_deg


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unknown image mode: {mode}")


def build_samples_for_session(
    session_dir: Path,
    out_train_root: Path,
    image_mode: str,
    history_len: int,
    sample_stride: int,
    horizon_steps: int,
    stop_dist: float,
    turn_deg: float,
    instruction_text: str,
):
    images_dir = session_dir / "images"
    if not images_dir.exists():
        return []

    traj_file = find_trajectory_file(session_dir)
    traj_rows = parse_trajectory_file(traj_file)
    if len(traj_rows) <= history_len + horizon_steps:
        return []

    session_name = session_dir.name
    out_session_dir = out_train_root / session_name
    ensure_dir(out_session_dir)

    # Keep only rows with matching image files.
    valid = []
    for row in traj_rows:
        frame_id = row[0]
        img_path = find_image_path(images_dir, frame_id)
        if img_path.exists():
            valid.append(row)

    samples = []
    if len(valid) <= history_len + horizon_steps:
        return samples

    for i in range(history_len - 1, len(valid) - horizon_steps, sample_stride):
        hist = valid[i - history_len + 1 : i + 1]
        cur = valid[i]
        fut = valid[i + horizon_steps]

        frame_rel_paths = []
        for frame_id, _, _, _ in hist:
            src_img = find_image_path(images_dir, frame_id)
            rel_path = f"{session_name}/frame_{frame_id}.jpg"
            dst_img = out_train_root / rel_path
            copy_or_link(src_img, dst_img, image_mode)
            frame_rel_paths.append(rel_path)

        _, x0, y0, yaw0 = cur
        _, x1, y1, yaw1 = fut
        d_forward, d_lateral, d_yaw = to_local_delta(x0, y0, yaw0, x1, y1, yaw1)
        action_text, d_yaw_deg = action_text_from_delta(d_forward, d_lateral, d_yaw, stop_dist, turn_deg)

        sample = {
            "video_id": session_name,
            "frames": frame_rel_paths,
            "q": instruction_text,
            "a": action_text,
            "meta": {
                "session": session_name,
                "anchor_frame_id": cur[0],
                "future_frame_id": fut[0],
                "delta_forward_m": round(d_forward, 4),
                "delta_lateral_m": round(d_lateral, 4),
                "delta_yaw_deg": round(d_yaw_deg, 3),
                "horizon_steps": horizon_steps,
            },
        }
        samples.append(sample)

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Convert real-robot 10Hz data to NaVILA vlnce training format")
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("/home/nvme04/public_data/xzs_data/navila_eval_data"),
        help="Root directory containing session folders like 20260319_162857",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/nvme04/public_data/xzs_data/NaVILA-RealNav"),
        help="Output dataset root",
    )
    parser.add_argument("--history_len", type=int, default=8, help="Number of historical frames per sample")
    parser.add_argument("--sample_stride", type=int, default=3, help="Frame stride between adjacent samples")
    parser.add_argument("--horizon_steps", type=int, default=5, help="Steps in future to define supervision delta")
    parser.add_argument("--stop_dist", type=float, default=0.05, help="Stop threshold in meters")
    parser.add_argument("--turn_deg", type=float, default=12.0, help="Turn threshold in degrees")
    parser.add_argument(
        "--image_mode",
        type=str,
        choices=["copy", "symlink"],
        default="symlink",
        help="How to place images in output train directory",
    )
    parser.add_argument(
        "--instruction_text",
        type=str,
        default="Imagine you are a robot programmed for navigation tasks. Analyze the observation history and output the next action.",
        help="Instruction text used as q field",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root

    ensure_dir(output_root)
    out_train_root = output_root / "train"
    ensure_dir(out_train_root)

    sessions = [d for d in sorted(input_root.iterdir()) if d.is_dir() and d.name[:2].isdigit()]
    all_samples = []

    for session_dir in sessions:
        session_samples = build_samples_for_session(
            session_dir=session_dir,
            out_train_root=out_train_root,
            image_mode=args.image_mode,
            history_len=args.history_len,
            sample_stride=args.sample_stride,
            horizon_steps=args.horizon_steps,
            stop_dist=args.stop_dist,
            turn_deg=args.turn_deg,
            instruction_text=args.instruction_text,
        )
        all_samples.extend(session_samples)
        print(f"[Session] {session_dir.name}: {len(session_samples)} samples")

    ann_path = output_root / "annotations.json"
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_sessions": len(sessions),
        "num_samples": len(all_samples),
        "history_len": args.history_len,
        "sample_stride": args.sample_stride,
        "horizon_steps": args.horizon_steps,
        "image_mode": args.image_mode,
    }
    summary_path = output_root / "conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. annotations: {ann_path}")
    print(f"Done. summary: {summary_path}")


if __name__ == "__main__":
    main()


# python3 convert_realnav_to_navila_vlnce.py \
# --input_root /home/nvme04/public_data/xzs_data/navila_eval_data \
# --output_root /home/nvme04/public_data/xzs_data/NaVILA-RealNav \
# --history_len 8 \
# --sample_stride 3 \
# --horizon_steps 5 \
# --image_mode symlink