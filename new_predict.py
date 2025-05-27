import time
import csv
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    mode = "predict"

    crop = False
    count = False

    VOCdevkit_path = "./elec_data"
    
    if mode == "predict":
        test_txt_path = os.path.join(VOCdevkit_path, "Detailed_ImageSets/Main/test_detail.txt")
        jpeg_images_dir = os.path.join(VOCdevkit_path, "JPEGImages")
        jpeg_detail_path = os.path.join(VOCdevkit_path, "JPEGImages/labels.csv")
        output_dir = os.path.abspath("./output/elec_data_detail/")
        predicted_output = os.path.abspath("./output/elec_data_detail/predicted_results.csv")
        
        os.makedirs(output_dir, exist_ok=True)

        actual_values = {}
        try:
            with open(jpeg_detail_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    actual_values[row['filename']] = float(row['number'])
        except FileNotFoundError:
            print(f"error: file not found: {jpeg_detail_path}")
            exit(1)

        with open(predicted_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'actual_number', 'predicted_number'])

            with open(test_txt_path, "r", encoding='utf-8') as f:
                base_names = [line.strip() for line in f.readlines()]

            for base_name in base_names:
                found = False
                image_ext = ""
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_path = os.path.join(jpeg_images_dir, f"{base_name}{ext}")
                    if os.path.exists(image_path):
                        found = True
                        image_ext = ext
                        break
                
                if not found:
                    print(f"skip: file not found: {base_name}")
                    continue

                try:
                    image = Image.open(image_path)
                    print(f"detecting: {base_name}{image_ext}")
                    
                    r_image, combined_number = frcnn.detect_image_detail(image, crop=crop, count=count)
                    
                    try:
                        predicted_float = float(combined_number)
                    except ValueError:
                        print(f"invalid predicted value '{combined_number}', reset to 0.0")
                        predicted_float = 0.0        
                    filename = f"{base_name}{image_ext}"
                    actual_number = actual_values.get(filename, 0.0)

                    writer.writerow([filename, actual_number, predicted_float])

                    output_path = os.path.join(output_dir, f"{base_name}.result.jpg")
                    r_image.save(output_path, quality=95, subsampling=0)
                    print(f"result saved to: {output_path}\n")

                except Exception as e:
                    print(f"error processing {base_name}: {str(e)}")
                    continue

        print(f"batch prediction complete! {len(base_names)} files processed")
        print(f"prediction results saved to: {predicted_output}")

    else:
        raise AssertionError("please specify correct mode: 'predict'")