# OCR Receipt Processing Pipeline

An automated pipeline for extracting structured data from receipt images using OCR (Optical Character Recognition) with EasyOCR and fallback to LLM-based parsing.

## Overview

This system processes receipt images through a multi-stage pipeline:
1. **Preprocessing** - Image enhancement and deskewing
2. **OCR Extraction** - Text detection using EasyOCR
3. **Structured Parsing** - Regex-based field extraction with confidence scoring
4. **LLM Fallback** - Fallback to Gemma LLM via Ollama for low-confidence results

The pipeline outputs structured JSON with confidence scores for each extracted field and flags for low-confidence/missing data.

## Project Structure

```
.
├── main.py                    # Entry point - orchestrates the entire pipeline
├── preprocessor.py            # Image preprocessing (scaling, grayscale, deskewing)
├── result_formatter.py        # Structured parsing with confidence scoring
├── llm_fallback.py            # LLM-based OCR fallback using Gemma model
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file

# Generated directories during runtime:
├── input/                     # Place receipt images here (*.jpg, *.png, etc.)
├── preprocessed/              # Preprocessed images (generated)
└── output/                    # JSON output files (generated)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bollesh/receipts-ocr.git
   cd "receipts-ocr"
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install Ollama and pull the Gemma model (for LLM fallback):
   ```bash
   # Install Ollama (see https://ollama.com/)
   ollama pull gemma4:31b-cloud
   ```

4. Ensure you have CUDA-compatible GPU for optimal EasyOCR performance (optional but recommended).

## Usage

### Basic Usage

1. Place receipt images in the `input/` directory:
   ```bash
   cp your_receipt.jpg input/
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

### How It Works

1. **Preprocessing** (`preprocessor.py`):
   - Images are upscaled 2x for better small text detection
   - Converted to grayscale
   - Median blur applied to reduce noise
   - Deskewing using minimum area rectangle rotation

2. **OCR Processing** (`main.py`):
   - EasyOCR processes preprocessed images
   - Each text detection includes bounding box, text, and confidence score
   - Results are passed to the formatter

3. **Structured Extraction** (`result_formatter.py`):
   - Groups OCR tokens into rows based on vertical position
   - Extracts: store name, date, items, total amount
   - Applies regex patterns and heuristic rules
   - Calculates confidence scores based on:
     - OCR confidence (50%)
     - Pattern matching (30%)
     - Keyword proximity (15%)
     - Position bonus (5%)

4. **Fallback Logic** (`main.py` + `llm_fallback.py`):
   - If any flags are present OR average confidence < 0.7
   - Image is encoded to base64 and sent to Gemma LLM
   - LLM extracts structured JSON directly from image

### Output Format

The pipeline generates JSON files in `output/` with the following structure:

```json
{
  "store_name": {
    "value": "STORE NAME",
    "confidence": 0.85
  },
  "date": {
    "value": "2024-12-15",
    "confidence": 0.92
  },
  "items": [
    {
      "name": "ITEM ONE",
      "price": "12.50",
      "confidence": 0.78
    },
    {
      "name": "ITEM TWO",
      "price": "8.99",
      "confidence": 0.81
    }
  ],
  "total_amount": {
    "value": "21.49",
    "confidence": 0.95
  },
  "flags": []
}
```

**Flags** indicate issues:
- `"MISSING: store_name"` - Field not detected
- `"LOW_CONFIDENCE: date (0.45)"` - Confidence below threshold (0.6)

## Dependencies

Key packages (see `requirements.txt` for full list):
- `easyocr==1.7.2` - OCR engine
- `opencv-python==4.13.0.92` - Image preprocessing
- `langchain-ollama==1.1.0` - LLM integration
- `numpy==2.4.4` - Numerical operations
- `torch==2.11.0` - PyTorch for EasyOCR
- `pillow==12.2.0` - Image handling

## Configuration

### OCR Settings
- Language: English (`['en']`)
- Minimum OCR confidence: 0.30 (`MIN_OCR_PROB` in `result_formatter.py`)
- Low-confidence threshold: 0.60 (`LOW_CONF_THRESHOLD`)

### LLM Settings
- Model: `gemma4:31b-cloud`
- Temperature: 0 (deterministic output)

### Preprocessing Settings
- Scaling: 2x upscale
- Blur: Median blur with kernel size 3
- Deskewing: Automatic rotation correction

## Performance Considerations

1. **GPU Acceleration**: EasyOCR benefits significantly from CUDA-enabled GPUs
2. **LLM Fallback**: Gemma 4 31B requires substantial RAM/VRAM (16GB+ recommended)
3. **Batch Processing**: The pipeline processes all images in `input/` directory
4. **Memory Usage**: Large receipts or high-resolution images increase memory usage

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'cv2'"**
   ```bash
   pip install opencv-python-headless
   ```

2. **EasyOCR CUDA errors**
   - Check CUDA installation: `nvcc --version`
   - Install CPU-only version: `pip install easyocr --no-deps` then install dependencies manually

3. **Ollama connection refused**
   - Ensure Ollama is running: `ollama serve`
   - Check model is pulled: `ollama list`

4. **Low confidence results**
   - Ensure images are clear and properly oriented
   - Check lighting conditions in original images
   - Consider adjusting `MIN_OCR_PROB` threshold

### Debug Mode

Add debugging by modifying `main.py`:
```python
# Add after line 33:
print(f"Processing {name}: {len(results)} detections, avg conf: {average_conf:.3f}")
```

## Extending the Pipeline

### Adding New Fields
1. Add regex patterns in `result_formatter.py`
2. Add extraction logic in `_extract_*` functions
3. Update output dictionary structure
4. Add confidence calculation rules

### Supporting New Languages
1. Update EasyOCR reader: `reader = easyocr.Reader(['en', 'fr', 'es'])`
2. Add language-specific regex patterns
3. Consider language-specific LLM prompts

### Custom Preprocessing
Modify `preprocess_receipt()` in `preprocessor.py`:
- Add contrast enhancement
- Implement specific noise reduction
- Custom scaling factors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR capabilities
- [Ollama](https://ollama.com/) for local LLM hosting
- [OpenCV](https://opencv.org/) for image processing
- [LangChain](https://www.langchain.com/) for LLM integration