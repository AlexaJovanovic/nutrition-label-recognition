"""
Advanced Deskewing Module for Nutrition Label Images

This module provides multiple methods for detecting and correcting image rotation:
1. Morphological contour-based detection (good for text blocks)
2. Hough line transform (robust for documents with lines)
3. Projection profile analysis (fast for dense text)
4. Ensemble method that combines multiple approaches

Each method returns both the detected angle and a confidence score.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class RotationResult:
    """Result of rotation detection."""
    angle: float  # Detected rotation angle in degrees
    confidence: float  # Confidence score (0-1)
    method: str  # Method used for detection


class DeskewingPipeline:
    """
    Advanced deskewing pipeline with multiple detection methods.
    """

    def __init__(self, debug: bool = False, debug_dir: str = "temp"):
        """
        Initialize the deskewing pipeline.

        Args:
            debug: Whether to save intermediate images for debugging
            debug_dir: Directory to save debug images
        """
        self.debug = debug
        self.debug_dir = debug_dir
        self.img_idx = 0

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for rotation detection.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (grayscale, binary threshold) images
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive thresholding works better for varying lighting
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/preprocess_gray.jpg", gray)
            cv2.imwrite(f"{self.debug_dir}/preprocess_thresh.jpg", thresh)

        return gray, thresh

    def method_contour_based(self, image: np.ndarray) -> RotationResult:
        """
        Detect rotation using morphological operations and contour analysis.

        This is your original method, improved with confidence scoring.

        Args:
            image: Input BGR image

        Returns:
            RotationResult with angle, confidence, and method name
        """
        gray, thresh = self.preprocess_image(image)

        # Morphological dilation to connect text into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=2)

        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/contour_dilate.jpg", dilate)

        # Find contours
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return RotationResult(0.0, 0.0, "contour_based")

        # Sort by area and take largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]

        # Get minimum area rectangle
        min_area_rect = cv2.minAreaRect(largest_contour)
        center, (width, height), angle = min_area_rect

        # OpenCV's minAreaRect returns:
        # - angle in range [-90, 0)
        # - The rectangle's width is always >= height
        # - angle is measured from the horizontal to the first side (counterclockwise)

        # The angle represents the rotation of the bounding box
        # We need to convert this to the image rotation angle

        # If width > height, the box is horizontal-ish, use angle as-is
        # If height > width, the box is vertical-ish, add 90
        if width < height:
            # Swap width/height interpretation
            angle = angle + 90

        # Now angle should be in range [-90, 90]
        # Normalize to [-45, 45] for small rotations
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90

        # Calculate confidence based on contour quality
        total_area = image.shape[0] * image.shape[1]
        contour_area = cv2.contourArea(largest_contour)
        area_ratio = contour_area / total_area

        # More area covered = higher confidence (to a point)
        confidence = min(area_ratio * 3, 1.0) if area_ratio > 0.1 else 0.3

        if self.debug:
            debug_img = image.copy()
            box = cv2.boxPoints(min_area_rect)
            box = np.int32(box)
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
            # Draw angle text
            cv2.putText(debug_img, f"Angle: {angle:.1f}deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(f"{self.debug_dir}/contour_result.jpg", debug_img)

        # Return the detected rotation angle (positive = image rotated counterclockwise)
        return RotationResult(angle, confidence, "contour_based")

    def method_hough_lines(self, image: np.ndarray) -> RotationResult:
        """
        Detect rotation using Hough line transform.

        Works well for documents with clear horizontal lines.

        Args:
            image: Input BGR image

        Returns:
            RotationResult with angle, confidence, and method name
        """
        gray, thresh = self.preprocess_image(image)

        # maybe add dilatation to accentuate lines 
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/hough_edges_{self.img_idx}.jpg", edges)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            print("No Hugh lines detected")
            return RotationResult(0.0, 0.0, "hough_lines")

        print(lines)

        # Calculate angles of all detected lines
        angles = []
        for line in lines:
            rho, theta = line[0]
            # theta is in range [0, pi]
            # Convert to degrees: theta=0 is vertical, theta=pi/2 is horizontal
            angle_deg = np.degrees(theta)

            # Convert to rotation angle: 0° = no rotation, positive = counterclockwise
            # When text lines are horizontal, theta ≈ 90°
            # When rotated clockwise, theta < 90°
            # When rotated counterclockwise, theta > 90°
            rotation_angle = angle_deg - 90

            # Focus on near-horizontal lines (within 45 degrees of horizontal)
            if abs(rotation_angle) < 45:
                angles.append(rotation_angle)

        if len(angles) == 0:
            return RotationResult(0.0, 0.0, "hough_lines")

        # Use median angle (more robust than mean)
        # Negate to get the angle needed to correct the rotation
        median_angle = -float(np.median(angles))

        # Confidence based on consistency of angles
        angle_std = float(np.std(angles))
        confidence = max(0.0, 1.0 - (angle_std / 10.0))  # Lower std = higher confidence

        if self.debug:
            debug_img = image.copy()
            for line in lines[:20]:  # Draw first 20 lines
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(f"{self.debug_dir}/hough_lines__{self.img_idx}.jpg", debug_img)

            print(f"Lines_{self.img_idx}", angles)

        self.img_idx = self.img_idx + 1

        return RotationResult(median_angle, confidence, "hough_lines")

    def method_projection_profile(self, image: np.ndarray) -> RotationResult:
        """
        Detect rotation using projection profile analysis.

        Fast method that works well for text-heavy images.

        Args:
            image: Input BGR image

        Returns:
            RotationResult with angle, confidence, and method name
        """
        gray, thresh = self.preprocess_image(image)

        # Try angles in range [-45, 45] with 0.5 degree steps
        angles_to_test = np.arange(-45, 45, 0.5)
        variances = []

        h, w = thresh.shape
        center = (w // 2, h // 2)

        for angle in angles_to_test:
            # Rotate image
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)

            # Calculate horizontal projection (sum of pixels per row)
            projection = np.sum(rotated, axis=1)

            # Variance of projection - higher variance = better alignment
            variance = np.var(projection)
            variances.append(variance)

        # Find angle with maximum variance
        max_idx = np.argmax(variances)
        best_angle = float(angles_to_test[max_idx])
        max_variance = float(variances[max_idx])

        # Confidence based on how much better the best angle is
        mean_variance = float(np.mean(variances))
        if mean_variance > 0:
            confidence = min((max_variance / mean_variance - 1.0) / 2.0, 1.0)
        else:
            confidence = 0.0

        return RotationResult(best_angle, confidence, "projection_profile")

    def method_ensemble(self, image: np.ndarray,
                       methods: Optional[list] = None) -> RotationResult:
        """
        Combine multiple methods using weighted voting based on confidence.

        Args:
            image: Input BGR image
            methods: List of method names to use. If None, uses all methods.

        Returns:
            RotationResult with ensemble angle, combined confidence, and method name
        """
        if methods is None:
            methods = ["contour_based", "hough_lines"]  # Exclude projection as it's slow

        results = []

        if "contour_based" in methods:
            results.append(self.method_contour_based(image))

        if "hough_lines" in methods:
            results.append(self.method_hough_lines(image))

        if "projection_profile" in methods:
            results.append(self.method_projection_profile(image))

        # Filter out low-confidence results
        valid_results = [r for r in results if r.confidence > 0.2]

        if len(valid_results) == 0:
            return RotationResult(0.0, 0.0, "ensemble_failed")

        # Weighted average based on confidence
        total_weight = sum(r.confidence for r in valid_results)
        weighted_angle = sum(r.angle * r.confidence for r in valid_results) / total_weight
        avg_confidence = total_weight / len(valid_results)

        return RotationResult(weighted_angle, avg_confidence, "ensemble")

    def detect_rotation(self, image: np.ndarray,
                       method: str = "ensemble") -> RotationResult:
        """
        Detect rotation angle using specified method.

        Args:
            image: Input BGR image
            method: Detection method - "contour_based", "hough_lines",
                   "projection_profile", or "ensemble"

        Returns:
            RotationResult with detected angle and confidence
        """
        if method == "contour_based":
            return self.method_contour_based(image)
        elif method == "hough_lines":
            return self.method_hough_lines(image)
        elif method == "projection_profile":
            return self.method_projection_profile(image)
        elif method == "ensemble":
            return self.method_ensemble(image)
        else:
            raise ValueError(f"Unknown method: {method}")

    def rotate_image(self, image: np.ndarray, angle: float,
                    expand: bool = True) -> np.ndarray:
        """
        Rotate image by specified angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counterclockwise)
            expand: Whether to expand canvas to fit entire rotated image

        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        if expand:
            # Calculate new bounding dimensions
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust rotation matrix for new center
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            # Rotate with expanded canvas
            rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        else:
            # Rotate without expanding
            rotated = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)

        return rotated

    def deskew(self, image: np.ndarray,
              method: str = "ensemble",
              confidence_threshold: float = 0.3,
              max_angle: float = 45.0) -> Tuple[np.ndarray, RotationResult]:
        """
        Detect and correct image rotation.

        Args:
            image: Input BGR image
            method: Detection method to use
            confidence_threshold: Minimum confidence to apply rotation
            max_angle: Maximum rotation angle to correct (degrees)

        Returns:
            Tuple of (deskewed_image, rotation_result)
        """
        # Detect rotation
        result = self.detect_rotation(image, method=method)

        # Only apply rotation if confidence is high enough and angle is reasonable
        if result.confidence >= confidence_threshold and abs(result.angle) <= max_angle:
            deskewed = self.rotate_image(image, -result.angle, expand=False)
            return deskewed, result
        else:
            # Return original image if confidence too low
            return image, result


def compare_methods(image_path: str, save_results: bool = True):
    """
    Compare all deskewing methods on a single image.

    Args:
        image_path: Path to input image
        save_results: Whether to save comparison results
    """
    import os

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    pipeline = DeskewingPipeline(debug=True)

    methods = ["contour_based", "hough_lines", "ensemble"]
    results = {}

    print(f"\nComparing deskewing methods on: {os.path.basename(image_path)}")
    print("=" * 80)

    for method in methods:
        result = pipeline.detect_rotation(image, method=method)
        results[method] = result

        print(f"\n{method.upper()}:")
        print(f"  Angle: {result.angle:.2f}°")
        print(f"  Confidence: {result.confidence:.2f}")

        if save_results:
            deskewed = pipeline.rotate_image(image, -result.angle)
            output_path = f"temp/deskew_{method}.jpg"
            cv2.imwrite(output_path, deskewed)
            print(f"  Saved: {output_path}")

    print("\n" + "=" * 80)

    # Recommend best method
    best_method = max(results.items(), key=lambda x: x[1].confidence)
    print(f"\nRecommended: {best_method[0]} (confidence: {best_method[1].confidence:.2f})")

    return results


def validate_deskewing(n_samples: int = 50, output_dir: str = "deskew_validation",
                       save_debug_images: bool = False, verbose: bool = False):
    """
    Validate deskewing accuracy by generating images with known rotation angles.

    Args:
        n_samples: Number of test images to generate
        output_dir: Directory to save validation results
        save_debug_images: Whether to save images showing detected angles
        verbose: Whether to print per-image results

    Returns:
        Dictionary with validation statistics
    """
    import os
    from image_generation import NutritionLabelGenerator
    from augmentation import AugmentationParams, apply_augmentations

    print("=" * 80)
    print("DESKEWING VALIDATION TEST")
    print("=" * 80)
    print(f"\nGenerating {n_samples} test images with known rotation angles...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if save_debug_images:
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    # Initialize
    generator = NutritionLabelGenerator(output_dir=output_dir)
    pipeline = DeskewingPipeline(debug=save_debug_images, debug_dir=os.path.join(output_dir, "debug"))

    # Storage for results
    results_data = {
        "true_angles": [],
        "predictions": {
            "contour_based": [],
            "hough_lines": [],
            "ensemble": []
        }
    }

    # Generate and test images
    for i in range(n_samples):
        # Generate base label
        label_data = generator.generate_nutrition_data()
        base_path = generator.generate_image(label_data, f"test_{i:04d}_base.png")
        base_img = cv2.imread(base_path)

        # Apply random rotation with known angle
        true_angle = np.random.uniform(-30, 30)

        # Create augmentation with only rotation
        augment = AugmentationParams(
            rotate_angle=true_angle,
            scale_factor=None,
            perspective_strength=None,
            brightness_factor=None,
            noise_amount=None,
            gaussian_blur_kernel=None
        )

        # Random augmentation     
        augment = AugmentationParams.random()
        augment.rotate_angle = true_angle

        rotated_img = apply_augmentations(base_img, augment)
        rotated_path = os.path.join(output_dir, f"test_{i:04d}_rotated.png")
        cv2.imwrite(rotated_path, rotated_img)

        # Test all deskewing methods
        results_data["true_angles"].append(true_angle)

        for method in ["contour_based", "hough_lines", "ensemble"]:
            result = pipeline.detect_rotation(rotated_img, method=method)
            results_data["predictions"][method].append(result)

        # Verbose output for debugging
        if verbose and i < 5:  # Show first 5 samples
            print(f"\nSample {i}:")
            print(f"  True angle: {true_angle:.2f}°")
            for method in ["contour_based", "hough_lines", "ensemble"]:
                pred = results_data["predictions"][method][-1]
                error = abs(true_angle - pred.angle)
                print(f"  {method}: {pred.angle:.2f}° (error: {error:.2f}°, conf: {pred.confidence:.2f})")

        # Save debug images
        if save_debug_images and i < 10:  # Save first 10 samples
            debug_img = rotated_img.copy()
            # Add text showing angles
            cv2.putText(debug_img, f"True: {true_angle:.1f}deg", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ensemble_pred = results_data["predictions"]["ensemble"][-1]
            cv2.putText(debug_img, f"Pred: {ensemble_pred.angle:.1f}deg", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            debug_path = os.path.join(output_dir, "debug", f"sample_{i:04d}.png")
            cv2.imwrite(debug_path, debug_img)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_samples} images...")

    print(f"\nAll {n_samples} images processed.")
    print("\nAnalyzing results...")

    # Calculate statistics for each method
    stats = {}

    for method in ["contour_based", "hough_lines", "ensemble"]:
        predictions = results_data["predictions"][method]
        true_angles = np.array(results_data["true_angles"])
        pred_angles = np.array([p.angle for p in predictions])
        confidences = np.array([p.confidence for p in predictions])

        # Calculate errors
        angle_errors = np.abs(true_angles - pred_angles)

        # Metrics
        mae = float(np.mean(angle_errors))
        rmse = float(np.sqrt(np.mean(angle_errors ** 2)))
        max_error = float(np.max(angle_errors))
        median_error = float(np.median(angle_errors))

        # Accuracy at different thresholds
        acc_1deg = float(np.mean(angle_errors < 1.0))
        acc_2deg = float(np.mean(angle_errors < 2.0))
        acc_5deg = float(np.mean(angle_errors < 5.0))

        avg_confidence = float(np.mean(confidences))

        stats[method] = {
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
            "median_error": median_error,
            "accuracy_1deg": acc_1deg,
            "accuracy_2deg": acc_2deg,
            "accuracy_5deg": acc_5deg,
            "avg_confidence": avg_confidence,
            "predictions": predictions,
            "true_angles": true_angles,
            "pred_angles": pred_angles,
            "errors": angle_errors
        }

    # Generate report
    report = generate_validation_report(stats, n_samples)

    # Save report
    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print("\n" + report)
    print(f"\nValidation results saved to: {output_dir}")
    print(f"Report saved to: {report_path}")

    return stats


def generate_validation_report(stats: dict, n_samples: int) -> str:
    """Generate a text report for deskewing validation."""
    report = []
    report.append("=" * 80)
    report.append("DESKEWING VALIDATION REPORT")
    report.append("=" * 80)
    report.append(f"\nTest Dataset: {n_samples} images with random rotations (-45° to 45°)")
    report.append("")

    # Compare methods
    methods = ["contour_based", "hough_lines", "ensemble"]

    for method in methods:
        s = stats[method]
        report.append("-" * 80)
        report.append(f"METHOD: {method.upper().replace('_', ' ')}")
        report.append("-" * 80)
        report.append(f"Mean Absolute Error (MAE):     {s['mae']:.2f}°")
        report.append(f"Root Mean Square Error (RMSE): {s['rmse']:.2f}°")
        report.append(f"Median Error:                  {s['median_error']:.2f}°")
        report.append(f"Maximum Error:                 {s['max_error']:.2f}°")
        report.append("")
        report.append("Accuracy (% within threshold):")
        report.append(f"  ±1°:  {s['accuracy_1deg']*100:.1f}%")
        report.append(f"  ±2°:  {s['accuracy_2deg']*100:.1f}%")
        report.append(f"  ±5°:  {s['accuracy_5deg']*100:.1f}%")
        report.append("")
        report.append(f"Average Confidence: {s['avg_confidence']:.2f}")
        report.append("")

    # Comparison summary
    report.append("=" * 80)
    report.append("COMPARISON SUMMARY")
    report.append("=" * 80)
    report.append(f"{'Method':<20} {'MAE':<10} {'RMSE':<10} {'Acc@±2°':<12} {'Confidence':<12}")
    report.append("-" * 80)

    for method in methods:
        s = stats[method]
        report.append(
            f"{method:<20} "
            f"{s['mae']:<10.2f} "
            f"{s['rmse']:<10.2f} "
            f"{s['accuracy_2deg']*100:<12.1f} "
            f"{s['avg_confidence']:<12.2f}"
        )

    # Best method
    report.append("")
    best_method = min(methods, key=lambda m: stats[m]['mae'])
    report.append(f"Best Method (by MAE): {best_method.upper().replace('_', ' ')}")
    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--validate":
            # Run validation test
            n_samples = 50
            verbose = False
            debug = False

            # Parse additional arguments
            for arg in sys.argv[2:]:
                if arg.isdigit():
                    n_samples = int(arg)
                elif arg == "--verbose":
                    verbose = True
                elif arg == "--debug":
                    debug = True

            validate_deskewing(n_samples=n_samples,
                             save_debug_images=debug,
                             verbose=verbose)
        else:
            # Compare methods on single image
            image_path = sys.argv[1]
            print("Advanced Deskewing Pipeline")
            print("=" * 80)
            results = compare_methods(image_path)
    else:
        # Default: run validation with verbose mode
        print("Running deskewing validation test...")
        print("(Use: python deskewing_advanced.py <image_path> to test single image)")
        print("(Use: python deskewing_advanced.py --validate [n_samples] [--verbose] [--debug] for validation)")
        print("")
        validate_deskewing(n_samples=50, verbose=True, save_debug_images=True)
