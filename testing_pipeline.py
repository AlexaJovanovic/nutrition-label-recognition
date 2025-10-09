"""
Complete Testing Pipeline for Nutrition Label OCR System

This module provides an end-to-end testing pipeline that:
1. Generates a synthetic dataset of nutrition labels with known ground truth
2. Runs OCR and regex-based extraction on each image
3. Evaluates predictions against ground truth using multiple metrics
4. Generates detailed reports and visualizations
"""

import os
import json
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import asdict
import cv2
import numpy as np
import easyocr

# Optional progress bar - fallback to simple iteration if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        """Fallback when tqdm is not installed"""
        print(f"{desc}...")
        return iterable

from nutrition_label import NutritionLabelData, evaluate_dataset, nutrition_similarity, presence_metrics
from image_generation import NutritionLabelGenerator
from regex_matching import extract_nutrients, easyocr_to_lines, nutrient_aliases


class TestingPipeline:
    """
    End-to-end testing pipeline for nutrition label recognition.
    """

    def __init__(
        self,
        output_dir: str = "test_results",
        dataset_dir: str = "test_dataset",
        use_gpu: bool = False
    ):
        """
        Initialize the testing pipeline.

        Args:
            output_dir: Directory to save test results and reports
            dataset_dir: Directory to save generated test dataset
            use_gpu: Whether to use GPU for OCR (if available)
        """
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.use_gpu = use_gpu

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        # Initialize OCR reader
        print("Initializing OCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        print("OCR reader initialized.")

        # Storage for results
        self.ground_truth: List[NutritionLabelData] = []
        self.predictions: List[NutritionLabelData] = []
        self.image_paths: List[str] = []
        self.ocr_times: List[float] = []
        self.extraction_times: List[float] = []

    def generate_dataset(self, n_samples: int = 50) -> List[Tuple[str, NutritionLabelData]]:
        """
        Generate a synthetic dataset of nutrition labels.

        Args:
            n_samples: Number of samples to generate

        Returns:
            List of tuples (image_path, ground_truth_label)
        """
        print(f"\nGenerating dataset with {n_samples} samples...")
        generator = NutritionLabelGenerator(output_dir=self.dataset_dir)
        dataset = generator.generate_dataset(n_samples)

        # Extract just image paths and labels (ignore augmentation params)
        processed_dataset = [(img_path, label) for img_path, label, _ in dataset]

        print(f"Dataset generated: {len(processed_dataset)} images in '{self.dataset_dir}'")
        return processed_dataset

    def run_ocr(self, image_path: str) -> Tuple[List, float]:
        """
        Run OCR on a single image.

        Args:
            image_path: Path to the image

        Returns:
            Tuple of (ocr_results, processing_time_seconds)
        """
        start_time = time.time()
        results = self.reader.readtext(image_path)
        elapsed = time.time() - start_time
        return results, elapsed

    def extract_prediction(self, ocr_results: List) -> Tuple[NutritionLabelData, float]:
        """
        Extract nutrition data from OCR results using regex matching.

        Args:
            ocr_results: Raw OCR output from EasyOCR

        Returns:
            Tuple of (predicted_label, processing_time_seconds)
        """
        start_time = time.time()
        lines = easyocr_to_lines(ocr_results)
        prediction = extract_nutrients(lines, nutrient_aliases)
        elapsed = time.time() - start_time
        return prediction, elapsed

    def run_predictions(self, dataset: List[Tuple[str, NutritionLabelData]]):
        """
        Run predictions on the entire dataset.

        Args:
            dataset: List of (image_path, ground_truth_label) tuples
        """
        print(f"\nRunning predictions on {len(dataset)} images...")

        self.ground_truth = []
        self.predictions = []
        self.image_paths = []
        self.ocr_times = []
        self.extraction_times = []

        for img_path, true_label in tqdm(dataset, desc="Processing images"):
            # Run OCR
            ocr_results, ocr_time = self.run_ocr(img_path)

            # Extract nutrients
            pred_label, extraction_time = self.extract_prediction(ocr_results)

            # Store results
            self.ground_truth.append(true_label)
            self.predictions.append(pred_label)
            self.image_paths.append(img_path)
            self.ocr_times.append(ocr_time)
            self.extraction_times.append(extraction_time)

        print(f"Predictions complete. Avg OCR time: {np.mean(self.ocr_times):.3f}s")

    def evaluate(self) -> Dict:
        """
        Evaluate predictions against ground truth.

        Returns:
            Dictionary containing all evaluation metrics
        """
        print("\nEvaluating results...")

        # Overall dataset metrics
        dataset_metrics = evaluate_dataset(self.ground_truth, self.predictions)

        # Per-sample metrics
        per_sample_similarities = []
        per_sample_presence = []

        for true_label, pred_label in zip(self.ground_truth, self.predictions):
            similarity = nutrition_similarity(true_label, pred_label)
            presence = presence_metrics(true_label, pred_label)

            per_sample_similarities.append(similarity)
            per_sample_presence.append(presence)

        # Performance metrics
        performance_metrics = {
            "avg_ocr_time": float(np.mean(self.ocr_times)),
            "avg_extraction_time": float(np.mean(self.extraction_times)),
            "total_time": float(sum(self.ocr_times) + sum(self.extraction_times)),
            "avg_total_time_per_image": float(np.mean([o + e for o, e in zip(self.ocr_times, self.extraction_times)]))
        }

        results = {
            "dataset_metrics": dataset_metrics,
            "per_sample_similarities": per_sample_similarities,
            "per_sample_presence": per_sample_presence,
            "performance": performance_metrics,
            "n_samples": len(self.ground_truth)
        }

        return results

    def generate_report(self, results: Dict) -> str:
        """
        Generate a detailed text report of the evaluation results.

        Args:
            results: Dictionary from evaluate() method

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("NUTRITION LABEL OCR TESTING PIPELINE - RESULTS REPORT")
        report.append("=" * 80)
        report.append("")

        # Dataset info
        report.append(f"Dataset Size: {results['n_samples']} images")
        report.append("")

        # Overall performance
        report.append("-" * 80)
        report.append("OVERALL METRICS")
        report.append("-" * 80)

        dm = results['dataset_metrics']
        report.append(f"Average Similarity Score: {dm['avg_similarity']:.4f} (1.0 = perfect match)")
        report.append("")

        report.append("Presence Detection Metrics (field-level):")
        pm = dm['presence']
        report.append(f"  Accuracy:  {pm['accuracy']:.4f}")
        report.append(f"  Precision: {pm['precision']:.4f}")
        report.append(f"  Recall:    {pm['recall']:.4f}")
        report.append(f"  F1 Score:  {pm['f1']:.4f}")
        report.append(f"  TP: {pm['tp']}, FP: {pm['fp']}, FN: {pm['fn']}, TN: {pm['tn']}")
        report.append("")

        # Performance metrics
        report.append("-" * 80)
        report.append("PERFORMANCE METRICS")
        report.append("-" * 80)
        perf = results['performance']
        report.append(f"Average OCR Time:        {perf['avg_ocr_time']:.3f}s")
        report.append(f"Average Extraction Time: {perf['avg_extraction_time']:.3f}s")
        report.append(f"Average Total Time:      {perf['avg_total_time_per_image']:.3f}s per image")
        report.append(f"Total Processing Time:   {perf['total_time']:.2f}s")
        report.append("")

        # Distribution statistics
        report.append("-" * 80)
        report.append("DISTRIBUTION STATISTICS")
        report.append("-" * 80)
        similarities = results['per_sample_similarities']
        report.append(f"Similarity Score Distribution:")
        report.append(f"  Min:    {np.min(similarities):.4f}")
        report.append(f"  Q1:     {np.percentile(similarities, 25):.4f}")
        report.append(f"  Median: {np.median(similarities):.4f}")
        report.append(f"  Q3:     {np.percentile(similarities, 75):.4f}")
        report.append(f"  Max:    {np.max(similarities):.4f}")
        report.append(f"  Std:    {np.std(similarities):.4f}")
        report.append("")

        # Worst performing samples
        report.append("-" * 80)
        report.append("WORST PERFORMING SAMPLES (Top 10)")
        report.append("-" * 80)

        # Get indices of worst samples
        worst_indices = np.argsort(similarities)[:min(10, len(similarities))]

        for rank, idx in enumerate(worst_indices, 1):
            report.append(f"{rank}. Image: {os.path.basename(self.image_paths[idx])}")
            report.append(f"   Similarity: {similarities[idx]:.4f}")
            report.append(f"   Ground Truth: {self._label_to_compact_str(self.ground_truth[idx])}")
            report.append(f"   Prediction:   {self._label_to_compact_str(self.predictions[idx])}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def _label_to_compact_str(self, label: NutritionLabelData) -> str:
        """Convert a label to a compact string representation."""
        def fmt(val):
            return f"{val:.1f}" if val is not None else "None"

        return (f"Cal:{fmt(label.calories)} Fat:{fmt(label.total_fat)} "
                f"SatFat:{fmt(label.saturated_fat)} Carb:{fmt(label.carbohydrates)} "
                f"Fiber:{fmt(label.dietary_fiber)} Sugar:{fmt(label.sugars)} "
                f"Prot:{fmt(label.protein)}")

    def save_results(self, results: Dict, report: str):
        """
        Save results to disk.

        Args:
            results: Results dictionary
            report: Text report
        """
        # Save JSON results
        json_path = os.path.join(self.output_dir, "results.json")

        # Convert to JSON-serializable format
        json_results = {
            "dataset_metrics": results["dataset_metrics"],
            "performance": results["performance"],
            "n_samples": results["n_samples"],
            "per_sample_similarities": results["per_sample_similarities"],
            "image_paths": self.image_paths,
            "ground_truth": [asdict(label) for label in self.ground_truth],
            "predictions": [asdict(label) for label in self.predictions]
        }

        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {json_path}")

        # Save text report
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Report saved to: {report_path}")

    def run_full_pipeline(self, n_samples: int = 50, save_output: bool = True):
        """
        Run the complete testing pipeline from start to finish.

        Args:
            n_samples: Number of test samples to generate
            save_output: Whether to save results to disk
        """
        print("=" * 80)
        print("STARTING COMPLETE TESTING PIPELINE")
        print("=" * 80)

        # Step 1: Generate dataset
        dataset = self.generate_dataset(n_samples)

        # Step 2: Run predictions
        self.run_predictions(dataset)

        # Step 3: Evaluate
        results = self.evaluate()

        # Step 4: Generate report
        report = self.generate_report(results)

        # Step 5: Save results
        if save_output:
            self.save_results(results, report)

        # Print report to console
        print("\n" + report)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)

        return results, report


def main():
    """
    Main function to run the testing pipeline with default parameters.
    """
    # Configuration
    N_SAMPLES = 50  # Number of test images to generate
    USE_GPU = False  # Set to True if you have CUDA-capable GPU

    # Initialize and run pipeline
    pipeline = TestingPipeline(
        output_dir="test_results",
        dataset_dir="test_dataset",
        use_gpu=USE_GPU
    )

    # Run complete pipeline
    results, report = pipeline.run_full_pipeline(n_samples=N_SAMPLES)

    return results, report


if __name__ == "__main__":
    main()
