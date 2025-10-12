# label_extractors/base_extractor.py

import easyocr
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
import numpy as np
import cv2
from label_extraction_core.nutrition_label import NutritionLabelData


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

nutrient_aliases = {
    "calories": ["calories", "energy", "kalorije", "energetska vrednost", "energija", "енергетска вредност"],
    "fat": ["Total fat", "fett",  "masti", "масти"],
    "saturated_fat": ["saturated fat", "saturates", "zasićene", "засићене"],
    "carbohydrates": ["carbohydrate", "carbs", "ugljeni hidrati", "угљени хидрати"],
    "sugar": ["sugar", "sucre", "sećeri", "шећери"],
    "fiber": ["fiber", "vlakna", "влакна"],
    "protein": ["protein", "proteini", "протеини"]
}

class AbstractLabelExtractor(ABC):
    """
    Base class for nutrition label detection models using OCR.
    Handles preprocessing, OCR extraction, and delegates nutrition data parsing
    to subclasses.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.reader: easyocr.Reader = easyocr.Reader(['en', 'rs_latin'], gpu=False)

    def predict_from_image(self, image: np.ndarray) -> Tuple[NutritionLabelData, List[Tuple[List[Tuple[int, int]], str, float]]]:
        """
        Runs preprocessing (if provided), OCR extraction, line reconstruction,
        and delegates nutrition data parsing to the subclass.

        Returns:
            A tuple of (nutrition_data, ocr_output)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")

        # Step 2: OCR text extraction
        ocr_output: List[Tuple[List[Tuple[int, int]], str, float]] = self.reader.readtext(image)

        # Step 3: Convert OCR output to structured line text
        lines: List[str] = self._easyocr_to_lines(ocr_output)

        if self.debug:
            pretty_print_easyocr(ocr_output)
            print("Lines of text:", lines)

        # Step 4: Delegate to subclass for actual data extraction
        nutrition_data = self._extract_nutrition_data(lines)

        # Return both the nutrition data and the OCR output
        return nutrition_data, ocr_output


    @abstractmethod
    def _extract_nutrition_data(self, lines: List[str]) -> NutritionLabelData:
        """
        Abstract method that parses OCR lines into structured NutritionLabelData.
        Must be implemented by subclasses.
        """
        pass

    def predict_from_path(self, image_path: str) -> NutritionLabelData:
        """
        Loads an image from disk, preprocesses, runs OCR, and extracts data.
        """
        image: Optional[np.ndarray] = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        return self.predict_from_image(image)

    def predict_batch(self, image_paths: List[str]) -> List[Optional[NutritionLabelData]]:
        """
        Runs prediction on multiple images given their file paths.
        Returns list of results or None for failed cases.
        """
        results: List[Optional[NutritionLabelData]] = []
        for path in image_paths:
            try:
                results.append(self.predict_from_path(path))
            except Exception as e:
                print(f"[WARN] Failed to process {path}: {e}")
                results.append(None)
        return results

    # ----------------------------------------------------------
    # Common OCR postprocessing utilities
    # ----------------------------------------------------------

    def _easyocr_to_lines(
        self,
        ocr_output: List[Tuple[List[Tuple[int, int]], str, float]],
        line_threshold: int = 15,
    ) -> List[str]:
        """
        Convert raw EasyOCR output into line-based text with proper X-axis ordering.

        Args:
            ocr_output (list): List of tuples from EasyOCR:
                               [([box_points], text, confidence), ...]
            line_threshold (int): Pixel distance to decide if words belong to the same line.

        Returns:
            list[str]: List of reconstructed text lines.
        """
        # Sort by Y (row), then X (column)
        results_sorted = sorted(ocr_output, key=lambda x: (x[0][0][1], x[0][0][0]))

        lines: List[str] = []
        current_line: List[Tuple[List[Tuple[int, int]], str]] = []
        prev_y: Optional[int] = None

        for (bbox, text, prob) in results_sorted:
            y = bbox[0][1]  # top-left corner's Y coordinate
            if prev_y is None or abs(y - prev_y) < line_threshold:
                current_line.append((bbox, text))
            else:
                # Sort current line by X and join
                current_line.sort(key=lambda item: item[0][0][0])
                line_text = " ".join(text for _, text in current_line)
                lines.append(line_text.lower())
                current_line = [(bbox, text)]
            prev_y = y

        # Add last line
        if current_line:
            current_line.sort(key=lambda item: item[0][0][0])
            line_text = " ".join(text for _, text in current_line)
            lines.append(line_text.lower())

        return lines