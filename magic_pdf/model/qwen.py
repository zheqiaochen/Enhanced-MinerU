import io
import base64
# import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# cache_dir = "./model_cache/Qwen2_5_VL-7B-Instruct"  # 指定本地缓存目录
# checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto",cache_dir=cache_dir)
# processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)

def pil_image_to_base64(img, format="PNG"):
    # 将图片保存到内存中的 BytesIO 对象
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    # 对内存中的数据进行 base64 编码，并解码为 UTF-8 字符串
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str
    
def extract_html_content(text):
  """
  Extracts the content between <html> and </html> tags, including the tags themselves.
  """
  match = re.search(r"<table>.*?</table>", text, re.DOTALL)
  if match:
    return match.group(0)
  else:
    return None

# def inference(image_path, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False):
#     image = Image.open(image_path)
#     image_local_path = "file://" + image_path
#     messages = [
#         {"role": "system", "content": sys_prompt},
#         {"role": "user", "content": [
#                 {"type": "text", "text": prompt},
#                 {"image": image_local_path},
#             ]
#         },
#     ]
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     print("text:", text)
#     # image_inputs, video_inputs = process_vision_info([messages])
#     inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
#     inputs = inputs.to('cuda')

#     output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
#     output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     return extract_html_content(output_text[0])

def inference(image_path):
    image = Image.open(image_path)
    base64_image = pil_image_to_base64(image)
    completion = client.chat.completions.create(
        model="qwen2.5-vl-7b-instruct",
        messages=[
    	{
    	    "role": "system",
            "content": [{"type":"text","text": "你是一个识别表格的助理"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                    # PNG图像：  f"data:image/png;base64,{base64_image}"
                    # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                    # WEBP图像： f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}, 
                },
                {"type": "text", "text": "解析并输出为html格式"},
            ],
        }
    ],
    )
    return extract_html_content(completion.choices[0].message.content) 

if __name__ == "__main__":
    print(inference("try.png"))