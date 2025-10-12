# label_extractors/levenshtein_matching_extractor.py

import re
from rapidfuzz import fuzz, process
from .base_extractor import AbstractLabelExtractor
from label_extraction_core.nutrition_label import NutritionLabelData
from .base_extractor import nutrient_aliases


class LevenshteinMatchingExtractor(AbstractLabelExtractor):
    """
    Extractor tolerant to OCR typos using fuzzy Levenshtein distance.
    Uses partial_ratio to handle cases where the nutrient name
    appears within a longer or noisy OCR line.
    """

    def __init__(self, threshold=60, debug=False):
        # Call superclass constructor
        super().__init__(debug)
        self.threshold = threshold

    def _extract_nutrition_data(self, lines):
        nutrients = {}

        for line in lines:
            lower_line = line.lower()

            best_score = 0
            best_nutrient = None

            # Compare this OCR line to all nutrient aliases
            for nutrient, aliases in nutrient_aliases.items():
                alias_match, score, _ = process.extractOne(
                    lower_line, aliases, scorer=fuzz.partial_ratio
                )
                if score > best_score:
                    best_score = score
                    best_nutrient = nutrient

            # If the fuzzy similarity is above threshold, extract numeric value
            if best_nutrient and best_score >= self.threshold:
                match = re.search(r"(\d+[.,]?\d*)\s*(kcal|kj|g|mg|mcg|µg)?", lower_line)
                if match:
                    value = float(match.group(1).replace(",", "."))
                    unit = match.group(2) or ""

                    # Convert kJ → kcal
                    if "kj" in unit:
                        value = int(value / 4.2)

                    nutrients[best_nutrient] = value

        # Return structured NutritionLabelData
        return NutritionLabelData(
            calories=nutrients.get("calories"),
            total_fat=nutrients.get("fat"),
            saturated_fat=nutrients.get("saturated_fat"),
            carbohydrates=nutrients.get("carbohydrates"),
            sugars=nutrients.get("sugar"),
            dietary_fiber=nutrients.get("fiber"),
            protein=nutrients.get("protein"),
        )
