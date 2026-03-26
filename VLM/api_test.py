import json  
from openai import OpenAI

client = OpenAI(
    api_key="sk-xmanyoxtyouposqtuzyctaohruuopsvehvfhfyycalyplzio", 
    base_url="https://api.siliconflow.cn/v1"
)

response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": "? 2020 年世界奥运会乒乓球男子和女子单打冠军分别是谁? "
             "Please respond in the format {\"男子冠军\": ..., \"女子冠军\": ...}"}
        ],
        response_format={"type": "json_object"}
    )

print(response.choices[0].message.content)


import requests
import base64

def encode_image_to_base64(image_path):
    """将图片文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image(image_path, api_token):
    """
    调用API描述图片内容
    
    参数:
        image_path: 图片文件路径
        api_token: API令牌
    """
    # 将图片转换为base64
    base64_image = encode_image_to_base64(image_path)
    
    # API端点
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    # 构造请求payload
    payload = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",  # 注意：需要确认该模型是否支持视觉功能
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请详细描述这张图片的内容。"
                    }
                ]
            }
        ],
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1
    }
    
    # 设置请求头
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    # 发送请求
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 检查HTTP错误
        
        # 解析响应
        result = response.json()
        
        # 提取描述文本
        if 'choices' in result and len(result['choices']) > 0:
            description = result['choices'][0]['message']['content']
            return description
        else:
            return f"未能获取描述: {result}"
            
    except requests.exceptions.RequestException as e:
        return f"请求错误: {e}"
    except Exception as e:
        return f"处理错误: {e}"

# 使用示例
if __name__ == "__main__":
    # 配置参数
    IMAGE_PATH = "/home/jason/vla_agent/test_dataset/episodes/ep_000001/uav/rgb/000000.jpg"  # 替换为你的图片路径
    API_TOKEN = "sk-xmanyoxtyouposqtuzyctaohruuopsvehvfhfyycalyplzio"  # 替换为你的API令牌
    
    print(f"正在处理图片: {IMAGE_PATH}")
    print("正在调用API...")
    
    # 调用函数获取图片描述
    description = describe_image(IMAGE_PATH, API_TOKEN)
    
    print("\n图片描述:")
    print("=" * 50)
    print(description)
    print("=" * 50)