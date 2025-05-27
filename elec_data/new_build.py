import os
import csv
import xml.etree.ElementTree as ET
from PIL import Image

def process_number(raw_number):
    """将原始数值转换为6位数字字符串（包含小数处理）"""
    if '.' in raw_number:
        integer_part, decimal_part = raw_number.split('.', 1)
        decimal_part = decimal_part.ljust(1, '0')  # 小数位补零
    else:
        integer_part = raw_number
        decimal_part = '0'
    
    combined = integer_part.zfill(5) + decimal_part[0]
    return combined[:6]  

def generate_voc_xml(jpg_folder, csv_path, output_folder):
    #标注字典 {filename: [(digits_str, xmin, ymin, xmax, ymax)]}
    annotations = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  
        for row in reader:
            filename = row['filename']
            try:
                # 解析坐标并转换为整数
                xmin = int(round(float(row['xmin'])))
                ymin = int(round(float(row['ymin'])))
                xmax = int(round(float(row['xmax'])))
                ymax = int(round(float(row['ymax'])))

                digits_str = process_number(row['number'])
            except (KeyError, ValueError) as e:
                print(f"invalid row: {row}, error: {e}")
                continue
            
            annotations.setdefault(filename, []).append((digits_str, xmin, ymin, xmax, ymax))

    os.makedirs(output_folder, exist_ok=True)
    for jpg_file in os.listdir(jpg_folder):
        if not jpg_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        if jpg_file not in annotations:
            print(f"don't found: {jpg_file}")
            continue

        try:
            with Image.open(os.path.join(jpg_folder, jpg_file)) as img:
                width, height = img.size
        except Exception as e:
            print(f"can't read {jpg_file}: {e}")
            continue

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "Dataset"
        ET.SubElement(root, "filename").text = jpg_file
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        
        ET.SubElement(root, "segmented").text = "0"

        # 处理每个数字区域
        for digits_str, xmin, ymin, xmax, ymax in annotations[jpg_file]:
            total_width = xmax - xmin
            segment_width = total_width // 6
            remainder = total_width % 6
            
            current_x = xmin
            for i in range(6):
                # 分配余数到前几个区域
                w = segment_width + (1 if i < remainder else 0)
                
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = digits_str[i]
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(current_x)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "xmax").text = str(current_x + w)
                ET.SubElement(bndbox, "ymax").text = str(ymax)
                
                current_x += w

        xml_path = os.path.join(output_folder, f"{os.path.splitext(jpg_file)[0]}.xml")
        ET.ElementTree(root).write(xml_path, encoding='utf-8', xml_declaration=True)
        print(f"saved: {xml_path}")

if __name__ == "__main__":
    generate_voc_xml(
        jpg_folder="/home/yx/yx_search/search/new-rcnn/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/elec_data/JPEGImages",
        csv_path="/home/yx/yx_search/search/new-rcnn/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/elec_data/JPEGImages/labels.csv",
        output_folder="/home/yx/yx_search/search/new-rcnn/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/elec_data/Detailed_Annotations"
    )