
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create output directory
os.makedirs("generated_labels", exist_ok=True)

def generate_nutrition_data():
    """Generate a dictionary with random nutrition values."""
    return {
        "Serving Size": f"{random.randint(20, 200)}g",
        "Calories": random.randint(50, 600),
        "Total Fat": f"{random.uniform(0, 30):.1f}g",
        "Saturated Fat": f"{random.uniform(0, 15):.1f}g",
        "Trans Fat": f"{random.uniform(0, 2):.1f}g",
        "Cholesterol": f"{random.randint(0, 100)}mg",
        "Sodium": f"{random.randint(0, 1500)}mg",
        "Total Carbohydrate": f"{random.uniform(0, 80):.1f}g",
        "Dietary Fiber": f"{random.uniform(0, 15):.1f}g",
        "Sugars": f"{random.uniform(0, 40):.1f}g",
        "Protein": f"{random.uniform(0, 50):.1f}g"
    }

def create_nutrition_label_image(data, filename):
    """Render the nutrition label as a table and save as an image."""
    df = pd.DataFrame(list(data.items()), columns=["Nutrient", "Value"])
    
    fig, ax = plt.subplots(figsize=(3, 5))
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Make header bold
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

# Generate sample images
for i in range(5):
    data = generate_nutrition_data()
    create_nutrition_label_image(data, f"generated_labels/nutrition_label_{i}.png")

print("5 sample nutrition labels saved in the 'labels' folder.")

import cv2
import numpy as np

# Load image
img = cv2.imread("labels/nutrition_label_0.png")

h, w = img.shape[:2]

# Original corner points
src_pts = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
])

# Target points (simulate 45° view)
dst_pts = np.float32([
    [0, 0],        # left top moved inward
    [w*0.7, 0],        # right top moved inward
    [w, h*1.1],            # right bottom stays
    [0, h]             # left bottom stays
])

# Perspective transformation matrix
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Warp the image
warped = cv2.warpPerspective(img, matrix, (w, h))

# Save result
cv2.imwrite("table_transformed.png", warped)
