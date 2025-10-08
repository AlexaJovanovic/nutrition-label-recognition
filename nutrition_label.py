from dataclasses import asdict, dataclass
from typing import List, Optional

@dataclass
class NutritionLabelData:
    calories: Optional[float]
    total_fat: Optional[float]
    saturated_fat: Optional[float]
    carbohydrates: Optional[float]
    dietary_fiber: Optional[float]
    sugars: Optional[float]
    protein: Optional[float]

    def __str__(self) -> str:
        """Human-readable nutrition label that gracefully handles missing values."""

        # Helper function to format each nutrient value
        def format_value(value, unit="", digits=1):
            if value is None:
                return f"No value"   # You could also return "N/A" or "—"
            if digits == 0:
                return f"{value:.0f}{unit}"
            return f"{value:.{digits}f}{unit}"

        return (
            f"Calories: {format_value(self.calories, ' kcal', 0)}\n"
            f"Total fat: {format_value(self.total_fat, ' g')}\n"
            f"  Saturated fat: {format_value(self.saturated_fat, ' g')}\n"
            f"Carbohydrate: {format_value(self.carbohydrates, ' g')}\n"
            f"  Dietary fiber: {format_value(self.dietary_fiber, ' g')}\n"
            f"  Sugars: {format_value(self.sugars, ' g')}\n"
            f"Protein: {format_value(self.protein, ' g')}"
        )

    def to_dict(self) -> dict:
        """Return a dict representation (useful for DataFrames or JSON)."""
        return asdict(self)


def nutrition_diff(label_true: NutritionLabelData, label_pred: NutritionLabelData) -> float:
    diffs = []
    for field in label_true.__dataclass_fields__:
        true_val = getattr(label_true, field)
        pred_val = getattr(label_pred, field)

        if true_val is None and pred_val is None:
            diff = 0.0  # agree on missing
        elif true_val is None or pred_val is None:
            diff = 1.0  # full penalty for missing mismatch
        else:
            # relative absolute difference (capped at 1)
            diff = min(abs(pred_val - true_val) / max(abs(true_val), 1e-6), 1.0)

        diffs.append(diff)

    mean_diff = sum(diffs) / len(diffs)
    return mean_diff

def nutrition_similarity(label_true: NutritionLabelData, label_pred: NutritionLabelData) -> float:
    similarity = 1 - nutrition_diff(label_true, label_pred)  # 1.0 = perfect match, 0.0 = completely wrong
    return similarity

def presence_metrics(label_true: NutritionLabelData, label_pred: NutritionLabelData):
    tp = fp = fn = tn = 0
    fields = label_true.__dataclass_fields__

    for field in fields:
        true_present = getattr(label_true, field) is not None
        pred_present = getattr(label_pred, field) is not None

        if true_present and pred_present:
            tp += 1
        elif not true_present and pred_present:
            fp += 1
        elif true_present and not pred_present:
            fn += 1
        else:
            tn += 1

    total = len(fields)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

def evaluate_dataset(
    true_labels: List[NutritionLabelData],
    pred_labels: List[NutritionLabelData]
):
    assert len(true_labels) == len(pred_labels), "Mismatch in dataset lengths"

    similarities = []
    total_tp = total_fp = total_fn = total_tn = 0

    for true, pred in zip(true_labels, pred_labels):
        similarities.append(nutrition_similarity(true, pred))
        m = presence_metrics(true, pred)
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        total_tn += m["tn"]

    # Mean numeric similarity
    avg_similarity = sum(similarities) / len(similarities)

    # Aggregate presence metrics globally
    total = total_tp + total_fp + total_fn + total_tn
    accuracy = (total_tp + total_tn) / total
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "avg_similarity": avg_similarity,
        "presence": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn
        }
    }