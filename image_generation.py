import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from label_extraction_core.nutrition_label import NutritionLabelData
from augmentation import *

class NutritionLabelGenerator:
    def __init__(self, output_dir="generated_labels"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def generate_nutrition_data():
        """Generate a dictionary with realistic random nutrition values."""

        # Step 1: total macros between 10–90 g
        total_macros = random.uniform(10, 90)

        # Step 2: random split into protein, fat, carbs
        parts = [random.random() for _ in range(3)]
        total_parts = sum(parts)
        protein = round(total_macros * parts[0] / total_parts, 1)
        fat     = round(total_macros * parts[1] / total_parts, 1)
        carbs   = round(total_macros * parts[2] / total_parts, 1)

        # Calories calculated exactly
        calories = int(round(4 * (protein + carbs) + 9 * fat))

        # Sugars + fiber <= carbs
        fiber = round(random.uniform(0, min(15, carbs)), 1)
        sugars = round(random.uniform(0, max(0, carbs - fiber)), 1)

        # Saturated fat ≤ total fat
        sat_fat = round(random.uniform(0, fat), 1)

        return NutritionLabelData(
            calories=calories,
            total_fat=fat,
            saturated_fat=sat_fat,
            carbohydrates=carbs,
            dietary_fiber=fiber,
            sugars=sugars,
            protein=protein,
        )

    def generate_image(
        self,
        label,  # <-- NutritionLabel instance
        filename="nutrition_label.png",
        bg_color="#ffffff",
        header_color="#cccccc",
        stripe=True
    ):
        """Render the nutrition label as an image from dict data."""
         # Build table rows (formatted, slightly indented)
        rows = [
            ("Calories", f"{label.calories:.0f} kcal"),
            ("Total fat", f"{label.total_fat:.1f} g"),
            ("   Saturated fat", f"{label.saturated_fat:.1f} g"),
            ("Carbohydrate", f"{label.carbohydrates:.1f} g"),
            ("   Dietary fiber", f"{label.dietary_fiber:.1f} g"),
            ("   Sugars", f"{label.sugars:.1f} g"),
            ("Protein", f"{label.protein:.1f} g"),
        ]

        df = pd.DataFrame(rows, columns=["Nutrient", "Value"])
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
    
    def generate_dataset(self, n: int):
        """Generate N random samples: (image_path, NutritionLabel, AugmentationParams)."""
        dataset = []
        for i in range(n):
            label = self.generate_nutrition_data()
            augment = AugmentationParams.random()
            img_name = f"label_{i:04d}.png"
            img_path = self.generate_image(label, img_name)

            img = cv2.imread(img_path)

            img = apply_augmentations(img, augment)

            cv2.imwrite(img_path, img)

            dataset.append((img_path, label, augment))
        return dataset



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
    brighter = adjust_brightness(img, 0.4)
    blurred = apply_gaussian_blur(img, 3)
    blurred2 = apply_gaussian_blur(img, 7)
    rotated15 = rotate_image(img, 15)
    rotated45 = rotate_image(img, 45)
    scaled = scale_image(img, 0.35)

    rotated_noise_blur = apply_gaussian_blur(add_noise(rotated15, 0.4), 7)
    rotated_blur_noise = add_noise(apply_gaussian_blur(rotated15, 7), 0.4)

    # Save results
    cv2.imwrite("generated_labels/label_perspective.png", warped)
    cv2.imwrite("generated_labels/label_noisy.png", noisy)
    cv2.imwrite("generated_labels/label_bright.png", brighter)
    cv2.imwrite("generated_labels/label_blur.png", blurred)
    cv2.imwrite("generated_labels/label_blur2.png", blurred2)
    cv2.imwrite("generated_labels/label_rot15.png", rotated15)
    cv2.imwrite("generated_labels/label_rot45.png", rotated45)
    cv2.imwrite("generated_labels/label_scaled.png", scaled)
    
    cv2.imwrite("generated_labels/label_RNB.png", rotated_noise_blur)
    cv2.imwrite("generated_labels/label_RBN.png", rotated_blur_noise)
    #test_augmentations()

    print("Generated base + augmented images in 'generated_labels'.")

    print(data)

    gen = NutritionLabelGenerator()
    dataset = gen.generate_dataset(5)

    for path, label, aug in dataset:
        print(path)
        print(label)
        print(aug)
        print("-" * 40)

