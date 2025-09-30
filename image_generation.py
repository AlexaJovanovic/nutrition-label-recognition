import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class NutritionLabelGenerator:
    def __init__(self, output_dir="generated_labels"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def generate_nutrition_data():
        """Generate a dictionary with random nutrition values."""
        return {
            "Serving Size": f"{random.randint(20, 200)}g",
            "Calories": random.randint(50, 600),
            "Total Fat": f"{random.uniform(0, 30):.1f}g",
            "Saturated Fat": f"{random.uniform(0, 15):.1f}g",
            "Carbohydrate": f"{random.uniform(0, 80):.1f}g",
            "Dietary Fiber": f"{random.uniform(0, 15):.1f}g",
            "Sugars": f"{random.uniform(0, 40):.1f}g",
            "Protein": f"{random.uniform(0, 50):.1f}g"
        }

    def generate_image(self, data, filename="nutrition_label.png", 
                    bg_color="#ffffff", header_color="#cccccc", stripe=True):
        """Render the nutrition label as an image from dict data."""
        df = pd.DataFrame(list(data.items()), columns=["Nutrient", "Value"])

        fig, ax = plt.subplots(figsize=(3, 5))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="left",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Style header + rows
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # header row
                cell.set_text_props(weight="bold", color="black")
                cell.set_facecolor(header_color)
            elif stripe and row % 2 == 0:  # alternate rows
                cell.set_facecolor("#f9f9f9")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

        return filepath


# --------------------------
# Augmentation Functions
# --------------------------

def add_perspective(img, strength=0.3):
    """Apply perspective transform (simulates angled photo)."""
    h, w = img.shape[:2]
    src_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst_pts = np.float32([
        [0, 0],
        [w*(1-strength), 0],
        [w, h*(1+strength)],
        [0, h]
    ])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h))

def scale_image(img, scale=1.2):
    """Resize image by a scaling factor."""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale)))

def add_noise(img, amount=0.02):
    """Add random Gaussian noise."""
    noise = np.random.normal(0, 255*amount, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def adjust_brightness(img, factor=1.2):
    """Adjust brightness by multiplying pixel values."""
    bright = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return bright

def apply_blur(img, ksize=3):
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def rotate_image(img, angle):
    """Rotate image by angle (in degrees), expanding canvas to avoid cropping."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix to account for translation
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Perform rotation
    return cv2.warpAffine(img, rot_mat, (new_w, new_h), borderValue=(255, 255, 255))

# --------------------------
# Example Usage
# --------------------------

def test_augmentations():
    generator = NutritionLabelGenerator()

    for i in range(2):  # generate 2 base labels for testing
        # Step 1: Generate base label
        data = generator.generate_nutrition_data()
        base_path = generator.generate_image(data, f"label_{i}_base.png")
        img = cv2.imread(base_path)

        # --- Augmentation Path A: Rotation + Noise ---
        angle = random.uniform(-15, 15)
        rotated = rotate_image(img, angle)
        cv2.imwrite(f"generated_labels/label_{i}_rotated.png", rotated)

        noisy = add_noise(rotated, amount=0.03)
        cv2.imwrite(f"generated_labels/label_{i}_rotated_noisy.png", noisy)

        # --- Augmentation Path B: Perspective + Blur ---
        warped = add_perspective(img, strength=0.2)
        cv2.imwrite(f"generated_labels/label_{i}_warped.png", warped)

        blurred = apply_blur(warped, ksize=5)
        cv2.imwrite(f"generated_labels/label_{i}_warped_blurred.png", blurred)

        # --- Augmentation Path C: Brightness + Rotation ---
        bright = adjust_brightness(img, factor=random.uniform(0.7, 1.3))
        cv2.imwrite(f"generated_labels/label_{i}_bright.png", bright)

        rotated_bright = rotate_image(bright, random.uniform(0.4, 0.6))
        cv2.imwrite(f"generated_labels/label_{i}_bright_rotated.png", rotated_bright)

        print(f"Saved augmented versions for label_{i}")

if __name__ == "__main__":
    generator = NutritionLabelGenerator()

    # Step 1: Generate base label
    data = generator.generate_nutrition_data()
    base_path = generator.generate_image(data, "label_base.png")
    colored_img_path = generator.generate_image(data, "label_colored.png", bg_color="#919102", header_color="#913602")


    # Step 2: Load image for augmentations
    img = cv2.imread(base_path)

    # Apply augmentations
    warped = add_perspective(img)
    noisy = add_noise(img)
    brighter = adjust_brightness(img, 0.5)
    blurred = apply_blur(img, 3)
    blurred2 = apply_blur(img, 7)
    rotated15 = rotate_image(img, 15)
    rotated45 = rotate_image(img, 45)

    # Save results
    cv2.imwrite("generated_labels/label_perspective.png", warped)
    cv2.imwrite("generated_labels/label_noisy.png", noisy)
    cv2.imwrite("generated_labels/label_bright.png", brighter)
    cv2.imwrite("generated_labels/label_blur.png", blurred)
    cv2.imwrite("generated_labels/label_blur2.png", blurred2)
    cv2.imwrite("generated_labels/label_rot15.png", rotated15)
    cv2.imwrite("generated_labels/label_rot45.png", rotated45)
    
    test_augmentations()

    print("Generated base + augmented images in 'generated_labels'.")
