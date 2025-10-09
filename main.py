import easyocr
import cv2
import numpy as np
from regex_matching import *

path = r'./generated_labels/'
path = r'./temp/'
img_name = "label_0_base.png"
img_name = "tuna_crop.jpg"

def pretty_print_easyocr(results):
    """
    Pretty-print EasyOCR output.
    Expected format: [
        ([ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ], 'text', confidence),
        ...
    ]
    """
    print(f"{'Index':<6} {'Bounding Box (x,y)':<45} {'Text':<20} {'Confidence':<10}")
    print("-" * 85)
    
    for i, (bbox, text, conf) in enumerate(results):
        # Flatten bounding box for compact display
        bbox_str = ", ".join([f"({int(x)},{int(y)})" for x, y in bbox])
        print(f"{i:<6} {bbox_str:<45} {text:<20} {conf:.6f}")

reader = easyocr.Reader(['en'], gpu=False)

# Run OCR
results = reader.readtext(path + img_name)

pretty_print_easyocr(results)

# Load the original image
image = cv2.imread(path + img_name)

# Loop through OCR results
for (bbox, text, prob) in results:
    # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    pts = [tuple(map(int, point)) for point in bbox]
    
    # Draw the bounding box (polygon)
    cv2.polylines(image, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 2)
    
    # Put the text above the top-left corner of the bbox
    cv2.putText(image, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 
                2, (75, 250, 55), 3, cv2.LINE_AA)

print(easyocr_to_lines(results))

final_res = extract_nutrients(easyocr_to_lines(results), nutrient_aliases)

# Save the result
output_path = path + "detected_text_tuna2.png"
cv2.imwrite(output_path, image)

print(final_res)
print(f"Saved result with bounding boxes to {output_path}")

