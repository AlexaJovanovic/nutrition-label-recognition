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
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

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
        angle = min_area_rect[-1]

        # Normalize angle to [-45, 45] range
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Calculate confidence based on contour quality
        total_area = image.shape[0] * image.shape[1]
        contour_area = cv2.contourArea(largest_contour)
        area_ratio = contour_area / total_area

        # More area covered = higher confidence (to a point)
        confidence = min(area_ratio * 3, 1.0) if area_ratio > 0.1 else 0.3

        if self.debug:
            debug_img = image.copy()
            box = cv2.boxPoints(min_area_rect)
            box = np.int0(box)
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
            cv2.imwrite(f"{self.debug_dir}/contour_result.jpg", debug_img)

        return RotationResult(-angle, confidence, "contour_based")

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

        # Edge detection
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/hough_edges.jpg", edges)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return RotationResult(0.0, 0.0, "hough_lines")

        # Calculate angles of all detected lines
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convert to degrees and normalize to [-90, 90]
            angle = np.degrees(theta) - 90

            # Focus on near-horizontal lines (within 45 degrees of horizontal)
            if abs(angle) < 45:
                angles.append(angle)

        if len(angles) == 0:
            return RotationResult(0.0, 0.0, "hough_lines")

        # Use median angle (more robust than mean)
        median_angle = float(np.median(angles))

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
            cv2.imwrite(f"{self.debug_dir}/hough_lines.jpg", debug_img)

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


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = "generated_labels/label_rot15.png"

    print("Advanced Deskewing Pipeline")
    print("=" * 80)

    # Compare methods
    results = compare_methods(image_path)
