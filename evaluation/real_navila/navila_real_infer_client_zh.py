"""
NaVILA 实机推理客户端（MMK2）。

流程：
1. 连接机器人并开启头部相机。
2. 用户把机器人推到起点，确认后缓存最近 8 帧。
3. 发送 8 帧 + 文本指令到服务端，接收单步动作。
4. 执行动作并更新历史帧，继续下一轮（闭环）直到 stop 或达最大步数。
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import websockets
from PIL import Image

from mmk2_sdk import MMK2Robot, RobotMode


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("navila_real_client")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def pil_from_frame_rgb(frame_rgb: np.ndarray) -> Image.Image:
    if frame_rgb.dtype != np.uint8:
        frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(frame_rgb, mode="RGB")


def b64_png_from_rgb(frame_rgb: np.ndarray) -> str:
    img = pil_from_frame_rgb(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class InputWaiter:
    def __init__(self):
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def clear(self):
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()

    def _run(self):
        while not self._stop.is_set():
            try:
                s = input().strip().lower()
            except EOFError:
                return
            if s == "y":
                self._event.set()


class RobotIO:
    def __init__(self, robot_ip: str, camera_sn: str, camera_resolution: str, frame_is_bgr: bool):
        self.robot = MMK2Robot(ip=robot_ip, mode=RobotMode.REAL)
        if not self.robot.is_connected():
            raise ConnectionError("机器人连接失败")
        self.camera_sn = camera_sn
        self.camera_resolution = camera_resolution
        self.frame_is_bgr = frame_is_bgr

    def init_camera(self):
        camera_config = {
            "head_camera": {
                "camera_type": "REALSENSE",
                "serial_no": f"'{self.camera_sn}'",
                "rgb_camera.color_profile": self.camera_resolution,
                "enable_depth": "false",
                "align_depth.enable": "false",
            }
        }
        self.robot.camera.set_camera_config(camera_config)
        self.robot.camera.start_stream(["head_camera"])
        time.sleep(2.0)

    def get_head_frame_rgb(self) -> Optional[np.ndarray]:
        frame = self.robot.camera.get_head_camera_frame()
        if frame is None or frame.get("rgb") is None:
            return None
        img = frame["rgb"]
        if img is None:
            return None
        if self.frame_is_bgr:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preview_until_confirm(self, window_name: str, overlay: str, frame_buffer: Optional[Deque[np.ndarray]], waiter: InputWaiter):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            img = self.get_head_frame_rgb()
            if img is None:
                time.sleep(0.05)
                continue
            if frame_buffer is not None:
                frame_buffer.append(img)

            disp = img.copy()
            cv2.putText(disp, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                disp,
                "press y in window or input y + Enter in terminal",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, disp_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key != 255 and chr(key).lower() == "y":
                cv2.destroyWindow(window_name)
                return
            if key != 255 and chr(key).lower() == "q":
                cv2.destroyWindow(window_name)
                raise KeyboardInterrupt("User quit in preview")
            if waiter.is_set():
                waiter.clear()
                cv2.destroyWindow(window_name)
                return

    def cleanup(self):
        try:
            self.robot.camera.stop_stream()
        except Exception:
            pass


async def ws_infer(server_url: str, obs_frames: List[np.ndarray], instruction: str, timeout_s: float) -> Dict:
    req = {
        "type": "inference_request",
        "timestamp": datetime.now().isoformat(),
        "instruction": instruction,
    }
    for i, frame in enumerate(obs_frames):
        req[f"obs_{i}"] = b64_png_from_rgb(frame)

    t0 = time.time()
    async with websockets.connect(server_url, ping_interval=None) as ws:
        await ws.send(json.dumps(req))
        resp_raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    latency_s = time.time() - t0

    resp = json.loads(resp_raw)
    resp["_client_latency_s"] = latency_s
    return resp


def parse_delta_actions(response_data: Dict) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for item in response_data.get("delta_actions", []):
        out.append((float(item["dx"]), float(item["dy"]), float(item["dtheta"])))
    return out


def execute_delta_actions(robot: MMK2Robot, delta_actions: List[Tuple[float, float, float]]) -> bool:
    all_ok = True
    for i, (dx, dy, dth) in enumerate(delta_actions):
        dist = float(np.hypot(dx, dy))
        alpha = float(np.arctan2(dy, dx)) if dist > 1e-9 else 0.0
        beta = normalize_angle(float(dth - alpha))

        logger.info(
            "Execute step=%d, dx=%.3f, dy=%.3f, dth=%.3f, alpha=%.3f, dist=%.3f, beta=%.3f",
            i,
            dx,
            dy,
            dth,
            alpha,
            dist,
            beta,
        )

        if abs(alpha) > 1e-6:
            ok = robot.turn_left(alpha, block=True) if alpha > 0 else robot.turn_right(-alpha, block=True)
            all_ok = all_ok and bool(ok)
        if dist > 1e-6:
            ok = robot.move_forward(dist, block=True)
            all_ok = all_ok and bool(ok)
        if abs(beta) > 1e-6:
            ok = robot.turn_left(beta, block=True) if beta > 0 else robot.turn_right(-beta, block=True)
            all_ok = all_ok and bool(ok)

    return all_ok


async def run(args):
    run_dir = ensure_dir(Path(args.run_dir) / f"navila_run_{now_ts()}")
    sent_dir = ensure_dir(run_dir / "sent")
    recv_dir = ensure_dir(run_dir / "received")
    exec_dir = ensure_dir(run_dir / "exec")

    io_robot = RobotIO(
        robot_ip=args.robot_ip,
        camera_sn=args.camera_sn,
        camera_resolution=args.camera_resolution,
        frame_is_bgr=args.frame_is_bgr,
    )
    io_robot.init_camera()

    waiter = InputWaiter()
    waiter.start()

    server_url = f"ws://{args.server_host}:{args.server_port}"
    frame_buf: Deque[np.ndarray] = deque(maxlen=args.num_video_frames)

    try:
        logger.info("请把机器人推到起点，确认后开始闭环推理")
        io_robot.preview_until_confirm(
            window_name="NaVILA Start Setup",
            overlay="Push robot to start and press y",
            frame_buffer=frame_buf,
            waiter=waiter,
        )

        if len(frame_buf) == 0:
            raise RuntimeError("未采集到起始帧")
        while len(frame_buf) < args.num_video_frames:
            frame_buf.appendleft(frame_buf[0])

        instruction = args.instruction
        if not instruction:
            instruction = input("请输入导航指令: ").strip()
        if not instruction:
            raise ValueError("导航指令不能为空")

        logger.info("Start closed-loop inference: max_steps=%d", args.max_steps)

        history = []
        for step_idx in range(args.max_steps):
            obs_frames = list(frame_buf)[-args.num_video_frames :]

            # save request images
            req_dir = ensure_dir(sent_dir / f"step_{step_idx:04d}")
            for i, f in enumerate(obs_frames):
                cv2.imwrite(str(req_dir / f"obs_{i}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

            resp = await ws_infer(server_url, obs_frames, instruction, args.timeout_s)
            (recv_dir / f"step_{step_idx:04d}.json").write_text(json.dumps(resp, indent=2, ensure_ascii=False), encoding="utf-8")

            if resp.get("type") != "inference_response" or resp.get("status") != "success":
                raise RuntimeError(f"服务端返回失败: {resp}")

            pred_text = resp.get("predicted_text", "")
            parsed_action = resp.get("parsed_action", {})
            is_stop = bool(resp.get("is_stop", False))
            latency = float(resp.get("_client_latency_s", 0.0))
            delta_actions = parse_delta_actions(resp)

            logger.info(
                "Step=%d | latency=%.3fs | text=%s | action=%s",
                step_idx,
                latency,
                pred_text,
                parsed_action,
            )

            if is_stop:
                logger.info("Model predicted STOP, finish loop")
                history.append(
                    {
                        "step": step_idx,
                        "latency_s": latency,
                        "predicted_text": pred_text,
                        "parsed_action": parsed_action,
                        "delta_actions": delta_actions,
                        "executed": False,
                    }
                )
                break

            if args.confirm_each_step:
                s = input(f"执行 step {step_idx} 的动作吗? (y/n): ").strip().lower()
                if s != "y":
                    logger.info("用户中止执行")
                    break

            ok = execute_delta_actions(io_robot.robot, delta_actions)

            # capture new observation after action and push into history buffer
            new_obs = io_robot.get_head_frame_rgb()
            if new_obs is not None:
                frame_buf.append(new_obs)
                cv2.imwrite(
                    str(exec_dir / f"obs_after_step_{step_idx:04d}.png"),
                    cv2.cvtColor(new_obs, cv2.COLOR_RGB2BGR),
                )

            history.append(
                {
                    "step": step_idx,
                    "latency_s": latency,
                    "predicted_text": pred_text,
                    "parsed_action": parsed_action,
                    "delta_actions": delta_actions,
                    "executed": True,
                    "execute_ok": bool(ok),
                }
            )

            if not ok:
                logger.warning("机器人执行返回失败，提前结束")
                break

        (run_dir / "run_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("运行完成，结果目录: %s", run_dir)

    finally:
        waiter.stop()
        io_robot.cleanup()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser("NaVILA 实机推理客户端")

    p.add_argument("--robot-ip", type=str, default="192.168.11.200")
    p.add_argument("--camera-sn", type=str, default="233522073186")
    p.add_argument("--camera-resolution", type=str, default="1280,720,30")
    p.add_argument("--frame-is-bgr", action="store_true")

    p.add_argument("--server-host", type=str, required=True)
    p.add_argument("--server-port", type=int, default=8001)
    p.add_argument("--timeout-s", type=float, default=120.0)

    p.add_argument("--instruction", type=str, default="")
    p.add_argument("--num-video-frames", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--confirm-each-step", action="store_true")

    p.add_argument("--run-dir", type=str, default="./runs_navila")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
