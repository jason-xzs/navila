#!/usr/bin/env python3
"""
NaVILA数据流程测试 - 使用VLN-CE环境渲染
按照navila_trainer.py的方式
"""

import gzip
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# 路径配置
R2R_JSON = "/root/autodl-tmp/jason/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
OUTPUT_DIR = Path("./test_out")

# 确保在evaluation目录下运行
EVAL_DIR = Path("/root/autodl-tmp/jason/NaVILA/evaluation")
if EVAL_DIR.exists():
    os.chdir(EVAL_DIR)
    sys.path.insert(0, str(EVAL_DIR))

def setup_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "json_data").mkdir(exist_ok=True)
    (OUTPUT_DIR / "rgb_images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "videos").mkdir(exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR.absolute()}")

def read_and_save_json(json_path, num_episodes=5):
    """读取JSON.gz"""
    print(f"\n{'='*80}")
    print("Step 1: 读取JSON数据")
    print(f"{'='*80}")
    
    with gzip.open(json_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    episodes = data.get('episodes', [])
    print(f"总共 {len(episodes)} 个episodes")
    
    sample_data = {
        'episodes': episodes[:num_episodes],
        'total_episodes': len(episodes)
    }
    
    output_json = OUTPUT_DIR / "json_data" / "episodes_sample.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"已保存: {output_json}")
    
    if episodes:
        ep = episodes[0]
        print(f"\n示例Episode:")
        print(f"  ID: {ep.get('episode_id')}")
        print(f"  Scene: {ep.get('scene_id')}")
        print(f"  指令: {ep['instruction']['instruction_text'][:60]}...")
    
    return episodes

def render_with_vlnce_env(num_episodes=3):
    """使用VLN-CE环境渲染（navila_trainer.py的方式）"""
    print(f"\n{'='*80}")
    print(f"Step 2: 使用VLN-CE环境渲染 ({num_episodes} episodes)")
    print(f"{'='*80}")
    
    try:
        # 导入VLN-CE模块
        from habitat import Config
        from habitat_baselines.common.environments import get_env_class
        from habitat_baselines.common.baseline_registry import baseline_registry
        from habitat_baselines.utils.common import batch_obs
        from vlnce_baselines.config.default import get_config
        from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
        from habitat_extensions.utils import observations_to_image, generate_video
        
        print("\n加载配置...")
        # 使用现有的配置文件
        config_path = "vlnce_baselines/config/r2r_baselines/navila.yaml"
        config = get_config(config_path)
        
        # 修改配置
        config.defrost()
        config.NUM_ENVIRONMENTS = 1
        config.EVAL.SPLIT = "val_unseen"
        config.EVAL.EPISODE_COUNT = num_episodes
        config.VIDEO_OPTION = ["disk"]
        config.VIDEO_DIR = str(OUTPUT_DIR / "videos")
        
        # 添加缺失的参数
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
        config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
        
        config.freeze()
        
        print("构建环境...")
        # 构建环境（navila_trainer.py 第121行）
        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        
        print(f"开始渲染 {num_episodes} 个episodes...\n")
        
        rendered_info = []
        ep_count = 0
        
        while ep_count < num_episodes:
            # Reset环境
            observations = envs.reset()
            print(observations[0].keys() if observations else "无观察结果")
            # batch = batch_obs(observations, device='cpu')
            
            # 【修改点】过滤掉字典类型的传感器
            filtered_obs = []
            for obs in observations:
                filtered = {}
                for key, value in obs.items():
                    # 只保留可以转换为tensor的数据（排除字典）
                    if not isinstance(value, dict):
                        filtered[key] = value
                filtered_obs.append(filtered)
            
            # 使用过滤后的观察数据
            batch = batch_obs(filtered_obs, device='cpu')

            current_episodes = envs.current_episodes()
            episode = current_episodes[0]
            
            print(f"Episode {ep_count+1}/{num_episodes}")
            print(f"  ID: {episode.episode_id}")
            print(f"  Scene: {episode.scene_id}")
            print(f"  指令: {episode.instruction.instruction_text[:60]}...")
            
            # 收集帧序列
            rgb_frames = []
            step = 0
            done = False
            
            while not done and step < 15:  # 最多5步
                # 获取RGB（navila_trainer.py 第161行的方式）
                rgb = batch[0]["rgb"].cpu().numpy()
                
                # 保存第一帧
                if step == 0:
                    rgb_image = Image.fromarray(rgb[:, :, :3], 'RGB')
                    img_path = OUTPUT_DIR / "rgb_images" / f"episode_{episode.episode_id}.jpg"
                    rgb_image.save(img_path)
                    print(f"  保存起始帧: {img_path.name}")
                
                # 使用habitat_extensions的可视化函数（如果有info）
                try:
                    # 执行动作：前进
                    outputs = envs.step([1])  # 1=MOVE_FORWARD
                    observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
                    
                    # 生成可视化帧（navila_trainer.py 第292-294行）
                    frame = observations_to_image(observations[0], infos[0])
                    rgb_frames.append(frame)
                    
                    done = dones[0]
                    # batch = batch_obs(observations, device='cpu')
                    # 【关键修改】step后也要过滤字典类型传感器
                    filtered_obs = []
                    for obs in observations:
                        filtered = {}
                        for key, value in obs.items():
                            if not isinstance(value, dict):
                                filtered[key] = value
                        filtered_obs.append(filtered)
                    
                    batch = batch_obs(filtered_obs, device='cpu')  # 使用过滤后的数据

                    step += 1
                    
                except Exception as e:
                    print(f"  步骤{step}失败: {e}")
                    break
            
            print(f"  渲染了 {len(rgb_frames)} 帧")
            
            # 生成视频（navila_trainer.py 第316-324行）
            if len(rgb_frames) > 0:
                try:
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(OUTPUT_DIR / "videos"),
                        images=rgb_frames,
                        episode_id=episode.episode_id,
                        checkpoint_idx=0,
                        metrics={"steps": len(rgb_frames)},
                        tb_writer=None,
                        fps=10
                    )
                    print(f"  生成视频: episode={episode.episode_id}.mp4")
                except Exception as e:
                    print(f"  视频生成失败: {e}")
            
            rendered_info.append({
                'episode_id': episode.episode_id,
                'scene_id': episode.scene_id,
                'instruction': episode.instruction.instruction_text[:80],
                'num_frames': len(rgb_frames)
            })
            
            ep_count += 1
            print()
        
        envs.close()
        
        # 保存渲染信息
        info_path = OUTPUT_DIR / "rgb_images" / "render_info.json"
        with open(info_path, 'w') as f:
            json.dump(rendered_info, f, indent=2)
        
        print(f"渲染完成: {len(rendered_info)} 个episodes")
        print(f"视频保存在: {OUTPUT_DIR / 'videos'}")
        
        return rendered_info
        
    except Exception as e:
        print(f"\n渲染失败: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\n如果遇到导入错误，请确保:")
        print(f"  1. 在evaluation目录下运行")
        print(f"  2. 已安装VLN-CE环境")
        print(f"  3. Habitat-Sim v0.1.7已正确安装")
        
        return []

def show_summary():
    """显示总结"""
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    
    print(f"\n输出目录: {OUTPUT_DIR}/")
    print(f"├── json_data/episodes_sample.json")
    print(f"├── rgb_images/episode_*.jpg")
    print(f"├── rgb_images/render_info.json")
    print(f"└── videos/episode_*.mp4")
    
    print(f"\n帧采样逻辑 (navila_trainer.py):")
    print(f"  1. 维护历史帧: past_rgbs")
    print(f"  2. 添加当前帧: curr_rgb")
    print(f"  3. sample_and_pad_images() 采样到8帧")
    print(f"  4. 送入模型推理")

def main():
    print("\n" + "="*80)
    print(" "*25 + "NaVILA数据流程测试")
    print("="*80)
    
    # 检查路径
    if not Path(R2R_JSON).exists():
        print(f"\n数据文件不存在: {R2R_JSON}")
        return
    
    if not EVAL_DIR.exists():
        print(f"\n评估目录不存在: {EVAL_DIR}")
        print(f"请修改EVAL_DIR为你的实际路径")
        return
    
    setup_output_dir()
    
    # Step 1: 读取JSON
    episodes = read_and_save_json(R2R_JSON, num_episodes=5)
    if not episodes:
        return
    
    # Step 2: 使用VLN-CE环境渲染
    rendered = render_with_vlnce_env(num_episodes=3)
    
    # Step 3: 总结
    show_summary()
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")
    print(f"\n查看视频:")
    print(f"  ls {OUTPUT_DIR.absolute()}/videos/")

if __name__ == "__main__":
    main()