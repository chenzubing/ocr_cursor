import os
import sys
from paddleocr import PaddleOCR
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
from openai import OpenAI
import json
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

class ColorTextAnalyzer:
    def __init__(self):
        # 预定义的颜色映射
        self.color_map = {
            'red': ((255, 0, 0), "红色可能表示警告、错误或重要信息"),
            'green': ((0, 255, 0), "绿色可能表示成功、通过或正确信息"),
            'blue': ((0, 0, 255), "蓝色可能表示链接或附加信息"),
            'black': ((0, 0, 0), "黑色为普通文本"),
            'yellow': ((255, 255, 0), "黄色可能表示警告或需要注意的信息"),
        }
    
    def get_closest_color(self, rgb):
        min_diff = float('inf')
        closest_color = 'unknown'
        
        rgb_color = sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        lab_color = convert_color(rgb_color, LabColor)
        
        for color_name, (color_rgb, _) in self.color_map.items():
            target_rgb = sRGBColor(color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255)
            target_lab = convert_color(target_rgb, LabColor)
            diff = delta_e_cie2000(lab_color, target_lab)
            
            if diff < min_diff:
                min_diff = diff
                closest_color = color_name
        
        return closest_color, self.color_map.get(closest_color, (None, "未知颜色类型"))[1]

def analyze_with_llm(text_color_data, client):
    """使用 LLM 分析文本和颜色的语义"""
    try:
        prompt = """
        请分析以下文本内容，文本包含了不同的颜色信息。请：
        1. 总结文本的主要内容
        2. 解释不同颜色文字的含义和作用
        3. 指出重要的信息点
        
        文本和颜色信息如下：
        """
        
        for item in text_color_data:
            prompt += f"\n颜色: {item['color']}\n文本: {item['text']}\n颜色含义: {item['color_meaning']}\n"
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的文档分析助手，擅长分析带有颜色标记的文本内容。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 分析错误: {str(e)}")
        return None

def get_text_color(img, box):
    """获取文本区域的主要颜色"""
    x1, y1 = map(int, box[0])
    x2, y2 = map(int, box[2])
    
    # 提取文本区域
    text_region = img[y1:y2, x1:x2]
    if text_region.size == 0:
        return (0, 0, 0)  # 返回黑色作为默认值
    
    # 计算主要颜色
    pixels = text_region.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, 3, None, criteria, 10, flags)
    
    # 获取出现次数最多的颜色
    unique_labels, counts = np.unique(labels, return_counts=True)
    dominant_color = palette[unique_labels[np.argmax(counts)]]
    
    return tuple(map(int, dominant_color))

def process_image(image_path, client):
    """处理图片并提取文本和颜色信息"""
    try:
        # 读取图片
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 初始化 OCR 和颜色分析器
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        color_analyzer = ColorTextAnalyzer()
        
        # 存储识别结果
        text_color_data = []
        
        # OCR识别
        result = ocr.ocr(image_path, cls=True)
        
        if result and result[0]:
            for line in result[0]:
                box = line[0]  # 文本框坐标
                text = line[1][0]  # 识别的文本
                
                # 获取文本颜色
                rgb_color = get_text_color(img_rgb, box)
                color_name, color_meaning = color_analyzer.get_closest_color(rgb_color)
                
                text_color_data.append({
                    'text': text,
                    'color': color_name,
                    'color_meaning': color_meaning,
                    'rgb': rgb_color
                })
        
        # 使用 LLM 分析结果
        analysis = analyze_with_llm(text_color_data, client)
        
        return text_color_data, analysis
    
    except Exception as e:
        print(f"处理错误：{str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

def save_results(text_color_data, analysis, output_dir):
    """保存识别和分析结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存原始识别结果
    raw_output = os.path.join(output_dir, f'ocr_result_{timestamp}.txt')
    with open(raw_output, 'w', encoding='utf-8') as f:
        for item in text_color_data:
            f.write(f"文本: {item['text']}\n")
            f.write(f"颜色: {item['color']}\n")
            f.write(f"颜色含义: {item['color_meaning']}\n")
            f.write(f"RGB值: {item['rgb']}\n")
            f.write("-" * 50 + "\n")
    
    # 保存分析结果
    if analysis:
        analysis_output = os.path.join(output_dir, f'analysis_{timestamp}.txt')
        with open(analysis_output, 'w', encoding='utf-8') as f:
            f.write(analysis)
    
    return raw_output, analysis_output if analysis else None

def main():
    if len(sys.argv) != 2:
        print("使用方法：python app.py <图片路径>")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("错误：找不到指定的图片文件")
        return
    
    # 检查环境变量中是否设置了 API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("警告：未设置 OPENAI_API_KEY 环境变量，将只进行 OCR 识别而不进行语义分析")
    
    # 创建 OpenAI 客户端
    client = OpenAI(api_key=api_key) if api_key else None
    
    # 创建输出目录
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在处理图片，请稍候...")
    text_color_data, analysis = process_image(image_path, client)
    
    if text_color_data:
        raw_output, analysis_output = save_results(text_color_data, analysis, output_dir)
        print(f"识别结果已保存到：{raw_output}")
        if analysis_output:
            print(f"分析结果已保存到：{analysis_output}")
    else:
        print("处理失败！")

if __name__ == "__main__":
    main() 