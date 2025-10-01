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
        """Generate a dictionary with realistic random nutrition values."""

        # Macronutrients in grams
        protein = round(random.uniform(0, 50), 1)
        fat = round(random.uniform(0, 30), 1)
        carbs = round(random.uniform(0, 80), 1)

        # Calories calculated exactly
        calories = int(round(4 * (protein + carbs) + 9 * fat))

        # Sugars + fiber <= carbs
        fiber = round(random.uniform(0, min(15, carbs)), 1)
        sugars = round(random.uniform(0, max(0, carbs - fiber)), 1)

        # Saturated fat ≤ total fat
        sat_fat = round(random.uniform(0, fat), 1)

        return {
            "Serving Size": f"{random.randint(20, 200)}g",
            "Calories": calories,
            "Total Fat": f"{fat}g",
            "Saturated Fat": f"{sat_fat}g",
            "Carbohydrate": f"{carbs}g",
            "Dietary Fiber": f"{fiber}g",
            "Sugars": f"{sugars}g",
            "Protein": f"{protein}g"
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


from augmentation import *

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
    noisy = add_noise(img, 0.5)
    brighter = adjust_brightness(img, 0.5)
    blurred = apply_gaussian_blur(img, 3)
    blurred2 = apply_gaussian_blur(img, 7)
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
