import easyocr
import cv2
import numpy as np
from regex_matching import *

path = r'./generated_labels/'
img_name = "nutrition_label_0.png"

reader = easyocr.Reader(['en'], gpu=False)

# Run OCR
results = reader.readtext(path + img_name)

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
                0.8, (255, 0, 0), 2, cv2.LINE_AA)

print(easyocr_to_lines(results))

final_res = extract_nutrients(easyocr_to_lines(results), nutrient_aliases)

# Save the result
output_path = path + "detected_text.png"
cv2.imwrite(output_path, image)

print(final_res)
print(f"Saved result with bounding boxes to {output_path}")
