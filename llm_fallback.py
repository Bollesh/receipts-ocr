"""
LLM Fallback Module for OCR Receipt Processing

This module provides fallback functionality using Large Language Models (LLMs)
when traditional OCR yields low-confidence results. It uses Ollama with the
Gemma model to parse receipt images directly and extract structured data.

The module:
1. Encodes images to base64 for LLM consumption
2. Sends image + structured prompt to LLM via LangChain
3. Parses JSON response with error handling
4. Returns structured data matching the pipeline's output format
"""

import base64
import json
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def encode_image(image_path):
    """
    Encode an image file to base64 string for LLM consumption.

    Base64 encoding is required because LLMs cannot directly read image files.
    The encoded image is embedded in a data URL that the LLM can process.

    Args:
        image_path: Path to the image file to encode

    Returns:
        Base64-encoded string representation of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def llm_parse_image(image_path):
    """
    Parse a receipt image using LLM (Gemma via Ollama) as fallback for low-confidence OCR.

    This function is called when traditional OCR results have low confidence scores
    or missing fields. It sends the original image directly to the LLM with a
    structured prompt requesting JSON output.

    Args:
        image_path: Path to the original receipt image

    Returns:
        Dictionary containing structured receipt data in the format:
        {
            "store_name": "...",
            "date": "...",
            "items": [{"name": "...", "price": "..."}, ...],
            "total_amount": "..."
        }
        Returns None if JSON parsing fails

    Raises:
        FileNotFoundError: If image_path doesn't exist
        json.JSONDecodeError: If LLM response is not valid JSON (caught internally)
    """
    # Step 1: Encode image to base64 for LLM consumption
    base64_image = encode_image(image_path)

    # Step 2: Construct precise instruction prompt for structured JSON output
    instruction = """Perform OCR on this receipt. Return ONLY valid JSON.
    Constraints: No markdown, no preamble. Items must have 'name' and 'price'.
    Format:
    {
      "store_name": "...",
      "date": "MM/DD/YY",
      "items": [{"name": "...", "price": "..."}],
      "total_amount": "..."
    }"""

    # Step 3: Initialize LLM with deterministic settings (temperature=0)
    # Using Gemma 4 31B model via Ollama with JSON output format enforced
    llm = ChatOllama(
        model="gemma4:31b-cloud",  # 31B parameter model for detailed analysis
        temperature=0,              # Deterministic output for consistency
        format="json"               # Force JSON output format
    )

    # Step 4: Create message with multimodal content (text + image)
    message = HumanMessage(
        content=[
            {"type": "text", "text": instruction},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ],
    )

    # Step 5: Invoke LLM with the message
    response = llm.invoke([message])

    # Step 6: Parse and validate JSON response
    try:
        # result.content is a string, json.loads converts it to a dictionary
        data = json.loads(response.content)
        return data
    except json.JSONDecodeError as e:
        # Log error but don't crash - let main.py handle retry logic
        print(f"Failed to parse JSON from LLM response: {e}")
        return None