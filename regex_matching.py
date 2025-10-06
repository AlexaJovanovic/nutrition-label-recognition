nutrient_aliases = {
    "calories": ["calories", "energie", "energía", "kalorien", "kalorije", "energetska vrednost", "energija", "енергетска вредност"],
    "protein": ["protein", "eiweiß", "proteínas", "protéine", "proteini", "протеини"],
    "carbohydrates": ["carbohydrate", "carbs", "kohlenhydrate", "glucides", "ugljeni hidrati", "угљени хидрати"],
    "fat": ["fat", "fett", "grasas", "lipides", "masti"],
    "sugar": ["sugar", "zucker", "azúcares", "sucre", "sećeri", "шећери"],
}

import unicodedata

def normalize(text):
    text = text.lower()
    #text = unicodedata.normalize("NFD", text)
    #text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


import re

def extract_nutrients(lines, nutrient_aliases):
    results = {}
    
    for line in lines:
        for nutrient, aliases in nutrient_aliases.items():
            for alias in aliases:
                if alias in line:
                    match = re.search(r"(\d+[.,]?\d*)", line)
                    if match:
                        results[nutrient] = float(match.group(1).replace(",", "."))
    return results

def easyocr_to_lines(ocr_output, line_threshold=15):
    """
    Convert raw EasyOCR output into line-based text.
    
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
        y = bbox[0][1]  # take top-left corner's Y
        if prev_y is None or abs(y - prev_y) < line_threshold:
            current_line.append(text)
        else:
            # new line
            lines.append(" ".join(current_line).lower())
            current_line = [text]
        prev_y = y

    # add the last line if not empty
    if current_line:
        lines.append(" ".join(current_line))

    return lines
