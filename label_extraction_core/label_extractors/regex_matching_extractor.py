# label_extractors/regex_matching_extractor.py

import re
from .base_extractor import AbstractLabelExtractor
from label_extraction_core.nutrition_label import NutritionLabelData
from .base_extractor import nutrient_aliases

class RegexMatchingExtractor(AbstractLabelExtractor):
    def __init__(self,  debug=False):
        # Call superclass constructor
        super().__init__(debug)

    
    def _extract_nutrition_data(self, lines):
        nutrients = {}

        for line in lines:
            lower_line = line.lower()
            for nutrient, aliases in nutrient_aliases.items():
                if any(alias in lower_line for alias in aliases):
                    match = re.search(r"(\d+[.,]?\d*)\s*(kcal|kj|g|mg|mcg|µg)?", lower_line)
                    if match:
                        value = float(match.group(1).replace(",", "."))
                        if "kj" in match.group():
                            value = int(value / 4.2)
                        nutrients[nutrient] = value
                        break

        return NutritionLabelData(
            calories=nutrients.get("calories"),
            total_fat=nutrients.get("fat"),
            saturated_fat=nutrients.get("saturated_fat"),
            carbohydrates=nutrients.get("carbohydrates"),
            sugars=nutrients.get("sugar"),
            dietary_fiber=nutrients.get("fiber"),
            protein=nutrients.get("protein"),
        )
