from dataclasses import dataclass, asdict

@dataclass
class NutritionLabel:
    calories: float
    total_fat: float
    saturated_fat: float
    carbohydrate: float
    dietary_fiber: float
    sugars: float
    protein: float

    def __str__(self) -> str:
        """Human-readable nutrition label."""
        return (
            f"Calories: {self.calories:.0f} kcal\n"
            f"Total fat: {self.total_fat:.1f} g\n"
            f"  Saturated fat: {self.saturated_fat:.1f} g\n"
            f"Carbohydrate: {self.carbohydrate:.1f} g\n"
            f"  Dietary fiber: {self.dietary_fiber:.1f} g\n"
            f"  Sugars: {self.sugars:.1f} g\n"
            f"Protein: {self.protein:.1f} g"
        )

    def to_dict(self) -> dict:
        """Return a dict representation (useful for DataFrames or JSON)."""
        return asdict(self)
