import os
import cv2
import json
from PIL import Image
from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
from PaddleOCR import PaddleOCR

def load_unique_ids(unique_id_file):
    """Load unique IDs from a text file."""
    with open(unique_id_file, 'r') as f:
        unique_ids = {line.strip() for line in f}
    return unique_ids

def predict(recognitor, detector, img_path, padding=5):
    """Perform OCR prediction on a single image."""
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        return [], []

    # Text detection
    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    result = result[:][:][0]

    # Filter Boxes
    boxes = []
    for line in result:
        x1, y1 = int(line[0][0]), int(line[0][1])
        x2, y2 = int(line[2][0]), int(line[2][1])
        boxes.append([max(0, x1 - padding), max(0, y1 - padding), x2 + padding, y2 + padding])

    boxes = boxes[::-1]

    texts = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cropped_image = img[y1:y2, x1:x2]
        try:
            cropped_image = Image.fromarray(cropped_image)
            text = recognitor.predict(cropped_image)
            texts.append(text)
        except Exception as e:
            texts.append("")  # Append empty string if an error occurs

    return texts, boxes

def process_images(input_folder, output_json, recognitor, detector, unique_ids):
    """Process images in a folder, filtering by unique IDs."""
    all_results = []

    for image_file in os.listdir(input_folder):
        # Check if the filename (without extension) is in the unique IDs set
        file_id, ext = os.path.splitext(image_file)
        if file_id not in unique_ids or ext.lower() != '.png':
            continue

        image_path = os.path.join(input_folder, image_file)

        if os.path.isfile(image_path):
            print(f"Processing image: {image_file}")

            # Predict words and boxes
            texts, boxes = predict(recognitor, detector, image_path)

            # Prepare result
            result = {
                "image": image_file,
                "words": texts,
                "boxes": boxes
            }
            all_results.append(result)

    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(all_results, json_file, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_json}")

# Configuration
input_folder = '../content/drive/MyDrive/NCKH/Images/Images/'  # Folder containing images
output_json = '../content/results.json'  # Output JSON file
unique_id_file = '../content/unique_ids.txt'  # Path to the file containing unique IDs

detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=True)

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained'] = True
config['predictor']['beamsearch'] = True
config['device'] = 'cuda'  # Use 'cpu' if GPU is unavailable
recognitor = Predictor(config)

# Load unique IDs
unique_ids = load_unique_ids(unique_id_file)

# Process images
process_images(input_folder, output_json, recognitor, detector, unique_ids)

