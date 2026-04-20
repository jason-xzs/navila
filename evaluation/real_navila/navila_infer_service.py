"""
NaVILA real-robot inference service (WebSocket).

Protocol:
- request:
  {
    "type": "inference_request",
    "obs_0": "<base64 PNG>",
    ...,
    "obs_7": "<base64 PNG>",
    "instruction": "... optional ...",
    "meta": { ... optional ... }
  }

- response:
  {
    "type": "inference_response",
    "status": "success",
    "inference_id": 0,
    "predicted_text": "Move forward 0.15 meters.",
    "parsed_action": {"action": "move_forward", "meters": 0.15},
    "delta_actions": [{"dx": 0.15, "dy": 0.0, "dtheta": 0.0}],
    "num_waypoints": 1,
    "is_stop": false,
    "latency_ms": 153
  }
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import websockets
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("navila_infer_service")


DEFAULT_INSTRUCTION = "Imagine you are a robot programmed for navigation tasks. Analyze the observation history and output the next action."


class NaVILAInferenceService:
    def __init__(
        self,
        model_path: str,
        host: str,
        port: int,
        num_video_frames: int = 8,
        conv_mode: str = "llama_3",
        max_new_tokens: int = 64,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.num_video_frames = num_video_frames
        self.conv_mode = conv_mode
        self.max_new_tokens = max_new_tokens
        self.inference_count = 0

        logger.info("Loading NaVILA model from: %s", model_path)
        model_name = Path(model_path).name
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, model_name)
        self.model = self.model.cuda().eval()
        logger.info("Model loaded")

    @staticmethod
    def decode_image(base64_str: str) -> Image.Image:
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data)).convert("RGB")

    def collect_obs_images(self, data: Dict) -> List[Image.Image]:
        # Preferred format: obs_0..obs_7
        obs_images = []
        for i in range(self.num_video_frames):
            key = f"obs_{i}"
            if key in data:
                obs_images.append(self.decode_image(data[key]))

        # Compatible format: data["obs_images"] = [base64, ...]
        if not obs_images and "obs_images" in data:
            for b64 in data["obs_images"]:
                obs_images.append(self.decode_image(b64))

        if not obs_images:
            raise ValueError("No observation images found in request")

        # Pad or truncate to model's expected frame count.
        if len(obs_images) < self.num_video_frames:
            pad_img = obs_images[0]
            while len(obs_images) < self.num_video_frames:
                obs_images.insert(0, pad_img)
        elif len(obs_images) > self.num_video_frames:
            obs_images = obs_images[-self.num_video_frames :]

        return obs_images

    def build_prompt(self, instruction: str, n_images: int) -> str:
        image_tokens = "<image>\n" * max(0, n_images - 1)
        question = (
            "Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {image_tokens}, and current observation <image>\n. Your assigned task is: "{instruction}" '
            "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            "degree, moving forward a certain distance, or stop if the task is completed."
        )
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate_text(self, obs_images: List[Image.Image], instruction: str) -> str:
        prompt = self.build_prompt(instruction, len(obs_images))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        image_tensor = process_images(obs_images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)

        conv = conv_templates[self.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.half().cuda(),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return outputs.strip()

    @staticmethod
    def parse_action_text(text: str) -> Dict:
        s = text.strip().lower()

        if "stop" in s:
            return {"action": "stop"}

        left_match = re.search(r"turn\s+left\s+([0-9]*\.?[0-9]+)", s)
        if left_match:
            deg = float(left_match.group(1))
            return {"action": "turn_left", "degrees": deg}

        right_match = re.search(r"turn\s+right\s+([0-9]*\.?[0-9]+)", s)
        if right_match:
            deg = float(right_match.group(1))
            return {"action": "turn_right", "degrees": deg}

        forward_match = re.search(r"move\s+forward\s+([0-9]*\.?[0-9]+)\s*(meters?|meter|m|cm)", s)
        if forward_match:
            value = float(forward_match.group(1))
            unit = forward_match.group(2)
            meters = value / 100.0 if unit == "cm" else value
            return {"action": "move_forward", "meters": meters}

        # Keyword fallback.
        if "turn left" in s:
            return {"action": "turn_left", "degrees": 15.0}
        if "turn right" in s:
            return {"action": "turn_right", "degrees": 15.0}
        if "move forward" in s:
            return {"action": "move_forward", "meters": 0.25}

        return {"action": "unknown", "raw": text}

    @staticmethod
    def parsed_action_to_delta_actions(parsed: Dict) -> List[Dict]:
        action = parsed.get("action")
        if action == "stop":
            return []
        if action == "move_forward":
            meters = float(parsed.get("meters", 0.0))
            meters = max(0.0, min(meters, 2.0))
            return [{"dx": meters, "dy": 0.0, "dtheta": 0.0}]
        if action == "turn_left":
            deg = float(parsed.get("degrees", 15.0))
            return [{"dx": 0.0, "dy": 0.0, "dtheta": deg * 3.141592653589793 / 180.0}]
        if action == "turn_right":
            deg = float(parsed.get("degrees", 15.0))
            return [{"dx": 0.0, "dy": 0.0, "dtheta": -deg * 3.141592653589793 / 180.0}]
        return []

    async def handle_client(self, websocket):
        client_addr = websocket.remote_address
        logger.info("Client connected: %s", client_addr)

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong", "time": datetime.now().isoformat()}))
                        continue

                    if data.get("type") != "inference_request":
                        raise ValueError("Unknown message type")

                    t0 = time.time()
                    obs_images = self.collect_obs_images(data)
                    instruction = data.get("instruction") or data.get("meta", {}).get("instruction") or DEFAULT_INSTRUCTION

                    pred_text = self.generate_text(obs_images, instruction)
                    parsed = self.parse_action_text(pred_text)
                    delta_actions = self.parsed_action_to_delta_actions(parsed)
                    latency_ms = int((time.time() - t0) * 1000)

                    response = {
                        "type": "inference_response",
                        "status": "success",
                        "inference_id": self.inference_count,
                        "predicted_text": pred_text,
                        "parsed_action": parsed,
                        "delta_actions": delta_actions,
                        "num_waypoints": len(delta_actions),
                        "is_stop": parsed.get("action") == "stop",
                        "latency_ms": latency_ms,
                    }
                    self.inference_count += 1
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                    logger.info("Done inference_id=%d, text=%s", response["inference_id"], pred_text)

                except Exception as e:
                    logger.exception("Request handling failed")
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "status": "failed",
                                "message": str(e),
                            },
                            ensure_ascii=False,
                        )
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected: %s", client_addr)

    async def start(self):
        logger.info("=" * 68)
        logger.info("NaVILA inference service started")
        logger.info("Host: %s", self.host)
        logger.info("Port: %s", self.port)
        logger.info("Checkpoint: %s", self.model_path)
        logger.info("=" * 68)

        async with websockets.serve(self.handle_client, self.host, self.port, max_size=None):
            await asyncio.Future()


def parse_args():
    parser = argparse.ArgumentParser(description="NaVILA real-robot inference service")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/nvme01/public_data/xzs_data/checkpoints/navila-8b-realnav-sft",
        help="Path to NaVILA checkpoint",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    return parser.parse_args()


def main():
    args = parse_args()
    service = NaVILAInferenceService(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        num_video_frames=args.num_video_frames,
        conv_mode=args.conv_mode,
        max_new_tokens=args.max_new_tokens,
    )
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")


if __name__ == "__main__":
    main()
