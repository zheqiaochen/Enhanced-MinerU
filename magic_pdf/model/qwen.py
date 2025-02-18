import io
import base64
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
    match = re.search(r"<html>.*?</html>", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def inference(image):
    """
    接收一个 PIL Image 对象或图片文件路径，返回解析后的 HTML 字符串。
    """
    # 如果传入的是文件路径，则打开图片
    if isinstance(image, str):
        image = Image.open(image)
    base64_image = pil_image_to_base64(image)
    completion = client.chat.completions.create(
        model="qwen2.5-vl-7b-instruct",
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "你是一个识别表格的助理"}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 传入 Base64 编码的图片，注意图片格式需与 Content Type 保持一致
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    },
                    {"type": "text", "text": "解析并输出为html格式，以<html>和</html>标签包裹"},
                ],
            }
        ],
    )
    return extract_html_content(completion.choices[0].message.content)

if __name__ == "__main__":
    # 测试时可以传入图片路径或 PIL Image 对象
    img = Image.open("try.png")
    html_result = inference(img)
    print(html_result)
