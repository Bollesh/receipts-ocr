"""
Main entry point for the OCR receipt processing pipeline.

This module orchestrates the complete pipeline:
1. Image preprocessing (deskewing, scaling, noise reduction)
2. OCR text extraction using EasyOCR
3. Structured field parsing with confidence scoring
4. LLM fallback for low-confidence results with retry logic
5. JSON output generation

The pipeline processes all images in the `input/` directory and generates
structured JSON files in the `output/` directory.
"""

import os
import easyocr
import json
from preprocessor import preprocess_receipt
from result_formatter import format_assignment_output, pretty_print
from llm_fallback import llm_parse_image


def collect_confidences(obj):
    """
    Recursively collects all confidence values from a structured JSON object.

    This function traverses dictionaries and lists to extract all values
    associated with the key "confidence". Used to calculate average confidence
    score for determining whether to trigger LLM fallback.

    Args:
        obj: Dictionary or list containing confidence scores

    Returns:
        List of confidence values (floats)
    """
    confidences = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "confidence":
                confidences.append(value)
            else:
                confidences.extend(collect_confidences(value))

    elif isinstance(obj, list):
        for item in obj:
            confidences.extend(collect_confidences(item))

    return confidences


def main():
    """
    Main pipeline execution function.

    Processes all images in the `input/` directory through:
    1. Preprocessing (image enhancement)
    2. OCR extraction with EasyOCR
    3. Structured parsing with confidence scoring
    4. LLM fallback with retry logic if confidence is low or flags present
    5. JSON output generation

    Outputs:
        - Console: OCR results and final structured JSON
        - Files: JSON files in the `output/` directory named <original_filename>.json
    """
    # Initialize OCR reader for English text
    reader = easyocr.Reader(['en'])

    # Walk through all files in the input directory
    for root, dirs, files in os.walk("input/"):
        for name in files:
            llm_retries = 0  # Track retry attempts for LLM fallback
            path = os.path.join(root, name)

            # Step 1: Preprocess image (deskew, scale, denoise)
            preprocess_receipt(path)

            # Ensure preprocessed directory exists (in case preprocessor doesn't create it)
            if not os.path.exists('preprocessed/'):
                os.mkdir('preprocessed/')

            # Step 2: Perform OCR on preprocessed image
            preprocessed_path = os.path.join("preprocessed/", name)
            results = reader.readtext(preprocessed_path)

            # Log individual OCR detections for debugging
            for (bbox, text, prob) in results:
                print(f"Text: {text} | Confidence: {prob:.4f}\n")

            # Step 3: Format OCR results into structured JSON with confidence scores
            final_json = format_assignment_output(results)

            # Step 4: Calculate average confidence across all fields
            conf_values = collect_confidences(final_json)
            average_conf = sum(conf_values) / len(conf_values) if conf_values else 0

            # Step 5: Trigger LLM fallback if confidence is low or flags present
            # LLM fallback conditions:
            # 1. Any flags present (missing fields or low confidence warnings)
            # 2. Average confidence across all fields below 0.7
            if len(final_json['flags']) > 0 or average_conf < 0.7:
                print(f"Low confidence detected (avg: {average_conf:.3f}), triggering LLM fallback...")

                # Retry LLM parsing up to 5 times if it returns None/empty
                while llm_retries < 5:
                    final_json = llm_parse_image(path)
                    if final_json:  # Successfully parsed by LLM
                        break
                    else:
                        llm_retries += 1
                        print(f"LLM fallback attempt {llm_retries} failed, retrying...")

            # Step 6: Print formatted results to console
            pretty_print(final_json)

            # Ensure output directory exists
            if not os.path.exists('output/'):
                os.mkdir('output/')

            # Step 7: Save results to JSON file in output directory
            filename = name.split(".")[0]  # Remove file extension
            output_path = os.path.join("output", filename + '.json')
            with open(output_path, "w") as f:
                f.write(json.dumps(final_json, indent=2, ensure_ascii=False))
            print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()