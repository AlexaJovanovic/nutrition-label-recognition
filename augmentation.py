from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import random

@dataclass
class AugmentationParams:
    rotate_angle: Optional[float] = None          # degrees
    scale_factor: Optional[float] = None          # e.g. 1.2 = 20% bigger
    perspective_strength: Optional[float] = None  # 0.0–1.0
    brightness_factor: Optional[float] = None     # >1 brighter, <1 darker
    noise_amount: Optional[float] = None          # 0.0–1.0
    gaussian_blur_kernel: Optional[int] = None    # odd int (3,5,…)

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

def apply_gaussian_blur(img, ksize=3):
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

import image_generation

def test_augmentations():
    generator = image_generation.NutritionLabelGenerator()

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

        blurred = apply_gaussian_blur(warped, ksize=5)
        cv2.imwrite(f"generated_labels/label_{i}_warped_blurred.png", blurred)

        # --- Augmentation Path C: Brightness + Rotation ---
        bright = adjust_brightness(img, factor=random.uniform(0.7, 1.3))
        cv2.imwrite(f"generated_labels/label_{i}_bright.png", bright)

        rotated_bright = rotate_image(bright, random.uniform(0.4, 0.6))
        cv2.imwrite(f"generated_labels/label_{i}_bright_rotated.png", rotated_bright)

        print(f"Saved augmented versions for label_{i}")


# --- main pipeline ---

def apply_augmentations(img, params: AugmentationParams):
    """Apply augmentations in a fixed order to img."""
    if params.rotate_angle is not None:
        img = rotate_image(img, params.rotate_angle)

    if params.scale_factor is not None:
        img = scale_image(img, params.scale_factor)

    if params.perspective_strength is not None:
        img = add_perspective(img, params.perspective_strength)

    if params.brightness_factor is not None:
        img = adjust_brightness(img, params.brightness_factor)

    if params.noise_amount is not None:
        img = add_noise(img, params.noise_amount)

    if params.gaussian_blur_kernel is not None:
        img = apply_gaussian_blur(img, params.gaussian_blur_kernel)

    return img


if __name__ == "__main__":
    img = cv2.imread("generated_labels/nutrition_label_0.png")

    params = AugmentationParams(
        rotate_angle=10,
        scale_factor=1.1,
        perspective_strength=0.2,
        brightness_factor=1.2,
        noise_amount=0.02,
        gaussian_blur_kernel=5
    )

    aug_img = apply_augmentations(img, params)
    cv2.imwrite("generated_labels/augmented.png", aug_img)
