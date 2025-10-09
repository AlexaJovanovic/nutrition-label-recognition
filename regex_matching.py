from nutrition_label import NutritionLabelData

nutrient_aliases = {
    "calories": ["calories", "energy", "energie", "energía", "kalorien", "kalorije", "energetska vrednost", "energija", "енергетска вредност"],
    "fat": ["fat", "fett", "grasas", "lipides", "masti"],
    "saturated_fat": ["saturated fat", "saturates", "gesättigte", "saturadas", "zasićene", "засићене"],
    "carbohydrates": ["carbohydrate", "carbs", "kohlenhydrate", "glucides", "ugljeni hidrati", "угљени хидрати"],
    "sugar": ["sugar", "zucker", "azúcares", "sucre", "sećeri", "шећери"],
    "fiber": ["fiber", "fibre", "ballaststoffe", "vlakna", "влакна"],
    "protein": ["protein", "eiweiß", "proteínas", "protéine", "proteini", "протеини"]
}

import unicodedata

def normalize(text):
    text = text.lower()
    #text = unicodedata.normalize("NFD", text)
    #text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text

import re

def extract_nutrients(lines, nutrient_aliases):
    """Extracts numeric nutrient values from OCR text lines and returns a NutritionLabelData instance."""
    nutrients = {}

    for line in lines:
        lower_line = line.lower()
        for nutrient, aliases in nutrient_aliases.items():
            if any(alias in lower_line for alias in aliases):
                # Match number (int or float) possibly followed by a unit like g/kcal/mg
                match = re.search(r"(\d+[.,]?\d*)\s*(kcal|kj|g|mg|mcg|µg)?", lower_line)
                if match:
                    value = float(match.group(1).replace(",", "."))
                    # convertion to kcal from kj if that was detected
                    print(match)
                    if ("kj" in match.group()):
                        value = int(value/4.2)
                    nutrients[nutrient] = value
                    break  # stop at first match for this line

    return NutritionLabelData(
        calories=nutrients.get("calories"),
        total_fat=nutrients.get("fat"),
        saturated_fat=nutrients.get("saturated_fat"),
        carbohydrates=nutrients.get("carbohydrates"),
        sugars=nutrients.get("sugar"),
        dietary_fiber=nutrients.get("fiber"),
        protein=nutrients.get("protein"),
    )

def easyocr_to_lines(ocr_output, line_threshold=15):
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

    lines = []
    current_line = []
    prev_y = None

    for (bbox, text, prob) in results_sorted:
        y = bbox[0][1]  # top-left corner's Y
        if prev_y is None or abs(y - prev_y) < line_threshold:
            current_line.append((bbox, text))
        else:
            # Before saving the previous line, sort its words by X
            current_line.sort(key=lambda item: item[0][0][0])  # sort by leftmost X
            line_text = " ".join(text for _, text in current_line)
            lines.append(line_text.lower())

            # Start a new line
            current_line = [(bbox, text)]
        prev_y = y

    # Add the last line
    if current_line:
        current_line.sort(key=lambda item: item[0][0][0])
        line_text = " ".join(text for _, text in current_line)
        lines.append(line_text.lower())

    return lines
