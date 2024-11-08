import os
import sys
import easyocr
from datetime import datetime
from PIL import Image
import cv2
import numpy as np
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
            'purple': ((128, 0, 128), "紫色可能表示特殊标记或重点内容"),
            'gray': ((128, 128, 128), "灰色可能表示次要信息或注释"),
        }
    
    def get_closest_color(self, rgb):
        """获取最接近的预定义颜色"""
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

def get_text_color(img, box):
    """获取文本区域的主要颜色"""
    # 转换坐标格式
    x1, y1 = map(int, box[0][0])
    x2, y2 = map(int, box[2][0])
    
    # 提取文本区域
    text_region = img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    if text_region.size == 0:
        return (0, 0, 0)
    
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

def create_output_dir():
    """创建输出目录"""
    if not os.path.exists('output'):
        os.makedirs('output')

def image_to_text(image_path):
    """将图片转换为文本"""
    try:
        # 读取图片
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 初始化 EasyOCR 和颜色分析器
        reader = easyocr.Reader(['ch_sim', 'en'])
        color_analyzer = ColorTextAnalyzer()
        
        print("正在识别文字和颜色，请稍候...")
        # 执行文字识别
        result = reader.readtext(image_path)
        
        if not result:
            print("警告：未能识别出任何文字")
            return None
        
        # 按照垂直位置排序结果
        sorted_result = sorted(result, key=lambda x: x[0][0][1])
        
        # 提取文本和颜色信息
        text_color_data = []
        for detection in sorted_result:
            box = detection[0]  # 文本框坐标
            text = detection[1]  # 识别的文本
            confidence = detection[2]  # 置信度
            
            # 获取文本颜色
            rgb_color = get_text_color(img_rgb, box)
            color_name, color_meaning = color_analyzer.get_closest_color(rgb_color)
            
            text_color_data.append({
                'text': text,
                'color': color_name,
                'color_meaning': color_meaning,
                'rgb': rgb_color,
                'confidence': confidence
            })
        
        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join('output', f'ocr_result_{timestamp}.txt')
        
        # 保存文本和颜色信息
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in text_color_data:
                f.write(f"文本: {item['text']}\n")
                f.write(f"颜色: {item['color']} (RGB: {item['rgb']})\n")
                f.write(f"颜色含义: {item['color_meaning']}\n")
                f.write(f"置信度: {item['confidence']:.2f}\n")
                f.write("-" * 50 + "\n")
            
        return output_file
    
    except Exception as e:
        print(f"错误：{str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    if len(sys.argv) != 2:
        print("使用方法：python app.py <图片路径>")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("错误：找不到指定的图片文件")
        return
    
    create_output_dir()
    print("正在处理图片，请稍候...")
    output_file = image_to_text(image_path)
    
    if output_file:
        print(f"转换成功！结果保存在：{output_file}")
    else:
        print("转换失败！")

if __name__ == "__main__":
    main() 