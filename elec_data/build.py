import os
import csv
import xml.etree.ElementTree as ET
from PIL import Image

def generate_voc_xml(jpg_folder, csv_path, output_folder):
    # 读取CSV数据并组织为字典 {filename: [objects]}
    annotations = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            filename = row[0]
            xmin, ymin, xmax, ymax = map(int, map(float, row[2:6]))  # 处理浮点数转整数
            if filename not in annotations:
                annotations[filename] = []
            annotations[filename].append((xmin, ymin, xmax, ymax))

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历JPG文件夹
    for jpg_file in os.listdir(jpg_folder):
        if not jpg_file.lower().endswith('.jpg'):
            continue

        # 检查是否有对应标注
        if jpg_file not in annotations:
            print(f"跳过未标注文件: {jpg_file}")
            continue

        # 获取图片尺寸
        jpg_path = os.path.join(jpg_folder, jpg_file)
        try:
            with Image.open(jpg_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"无法读取图片尺寸: {jpg_file} - {str(e)}")
            continue

        # 创建XML结构
        root = ET.Element("annotation")
        
        # 基础信息
        ET.SubElement(root, "folder").text = "Dataset"
        ET.SubElement(root, "filename").text = jpg_file
        
        # 图片尺寸
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        
        ET.SubElement(root, "segmented").text = "0"

        # 添加所有object
        for obj in annotations[jpg_file]:
            obj_elem = ET.SubElement(root, "object")
            ET.SubElement(obj_elem, "name").text = "number"
            ET.SubElement(obj_elem, "truncated").text = "1"
            ET.SubElement(obj_elem, "difficult").text = "0"
            
            bbox = ET.SubElement(obj_elem, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(obj[0])
            ET.SubElement(bbox, "ymin").text = str(obj[1])
            ET.SubElement(bbox, "xmax").text = str(obj[2])
            ET.SubElement(bbox, "ymax").text = str(obj[3])

        # 美化XML格式
        xml_str = ET.tostring(root, encoding="utf-8")
        dom = ET.ElementTree(root)
        
        # 保存文件
        xml_file = os.path.splitext(jpg_file)[0] + ".xml"
        xml_path = os.path.join(output_folder, xml_file)
        dom.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"已生成: {xml_file}")

if __name__ == "__main__":
    generate_voc_xml(
        jpg_folder="/home/yx/yx_search/search/new-rcnn/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/elec_data/Dataset",
        csv_path="/home/yx/yx_search/search/new-rcnn/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/elec_data/Dataset/labels.csv",
        output_folder="/home/yx/yx_search/search/new-rcnn/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/elec_data/Annotations"
    )