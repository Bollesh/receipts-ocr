#!/usr/bin/env python3
"""
ASCII-friendly README.md to PDF converter.

This script creates a PDF version of the README.md file using only ASCII characters
to avoid Unicode font issues.
"""

import os
import re
from fpdf import FPDF
from datetime import datetime


class ASCIIPDFGenerator:
    """Generate PDF from README.md using ASCII-only characters."""

    def __init__(self, readme_path="README.md", pdf_path="README.pdf"):
        self.readme_path = readme_path
        self.pdf_path = pdf_path
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def load_and_clean_readme(self):
        """Load README.md and replace all non-ASCII characters."""
        if not os.path.exists(self.readme_path):
            raise FileNotFoundError(f"README.md not found at {self.readme_path}")

        with open(self.readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace common Unicode characters with ASCII equivalents
        replacements = {
            # Bullets
            '•': '*',
            '─': '-',
            '├': '+',
            '│': '|',
            '└': '+',
            '┌': '+',
            '┐': '+',
            '┘': '+',
            # Quotes and other special characters
            '‘': "'",
            '’': "'",
            '“': '"',
            '”': '"',
            '…': '...',
        }

        for unicode_char, ascii_char in replacements.items():
            content = content.replace(unicode_char, ascii_char)

        # Also replace any remaining non-ASCII characters
        content = re.sub(r'[^\x00-\x7F]', '', content)

        return content.splitlines()

    def add_title_page(self):
        """Add title page."""
        self.pdf.add_page()

        # Title
        self.pdf.set_font("Helvetica", "B", 24)
        self.pdf.cell(0, 40, "OCR Receipt Processing Pipeline", align="C")
        self.pdf.ln(20)

        # Subtitle
        self.pdf.set_font("Helvetica", "I", 14)
        self.pdf.cell(0, 10, "Documentation", align="C")
        self.pdf.ln(20)

        # Description
        self.pdf.set_font("Helvetica", "", 12)
        desc = "This document contains the complete documentation for the OCR Receipt Processing Pipeline, an automated system for extracting structured data from receipt images using OCR and LLM-based parsing."
        self.pdf.multi_cell(0, 6, desc)
        self.pdf.ln(10)

        # Generated info
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.cell(0, 10, "Generated from README.md", align="C")
        self.pdf.ln(5)

        self.pdf.set_font("Helvetica", "", 10)
        current_date = datetime.now().strftime("%B %d, %Y")
        self.pdf.cell(0, 10, f"Generated on: {current_date}", align="C")
        self.pdf.ln(20)

    def process_and_render(self, lines):
        """Process and render content."""
        in_code_block = False
        code_block_content = []

        for line_num, line in enumerate(lines):
            line = line.rstrip('\n')

            # Handle code blocks
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    self.render_code_block(code_block_content)
                    code_block_content = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
                continue

            if in_code_block:
                code_block_content.append(line)
                continue

            # Handle headers
            if line.startswith('# '):
                self.render_header(line[2:], 1)
            elif line.startswith('## '):
                self.render_header(line[3:], 2)
            elif line.startswith('### '):
                self.render_header(line[4:], 3)

            # Handle bullet points (now using '*' after cleaning)
            elif line.strip().startswith('* ') or line.strip().startswith('- '):
                text = line.strip()[2:]
                self.render_bullet(text)

            # Handle numbered lists
            elif re.match(r'^\d+\.\s+', line.strip()):
                text = re.sub(r'^\d+\.\s+', '', line.strip())
                self.render_bullet(text, numbered=True)

            # Regular text
            elif line.strip():
                self.render_text(line)

            # Empty line
            else:
                self.pdf.ln(5)

    def render_header(self, text, level):
        """Render a header."""
        if level == 1:
            self.pdf.add_page()
            self.pdf.set_font("Helvetica", "B", 18)
            self.pdf.set_text_color(0, 0, 139)  # Dark blue
        elif level == 2:
            self.pdf.set_font("Helvetica", "B", 16)
            self.pdf.set_text_color(0, 0, 100)  # Medium blue
        else:
            self.pdf.set_font("Helvetica", "B", 14)
            self.pdf.set_text_color(0, 0, 50)  # Dark gray

        self.pdf.cell(0, 10, text, align="L")
        self.pdf.ln(8)
        self.pdf.set_text_color(0, 0, 0)  # Reset to black

    def render_text(self, text):
        """Render regular text."""
        if not text.strip():
            return

        self.pdf.set_font("Helvetica", "", 11)
        self.pdf.multi_cell(0, 5, text)
        self.pdf.ln(2)

    def render_bullet(self, text, numbered=False):
        """Render a bullet point using ASCII '*'."""
        if not text.strip():
            return

        self.pdf.set_font("Helvetica", "", 11)
        prefix = f"{'1.' if numbered else '*'} "
        self.pdf.cell(5)  # Indent
        self.pdf.multi_cell(0, 5, prefix + text)
        self.pdf.ln(1)

    def render_code_block(self, lines):
        """Render a code block."""
        if not lines:
            return

        code = '\n'.join(lines)

        self.pdf.set_font("Courier", "", 9)
        line_height = 4
        padding = 3

        # Calculate total height
        total_lines = len(lines)
        total_height = total_lines * line_height + 2 * padding

        x = self.pdf.get_x() + 5
        y = self.pdf.get_y()

        # Draw background rectangle
        self.pdf.set_fill_color(240, 240, 240)
        self.pdf.rect(x - 1, y, 185, total_height, style='F')

        # Add text lines
        self.pdf.set_xy(x, y + padding)
        for line in lines:
            line = line.rstrip('\r')
            self.pdf.cell(0, line_height, line)
            self.pdf.ln(line_height)

        self.pdf.ln(padding)

    def generate(self):
        """Generate the PDF."""
        print(f"Generating PDF from {self.readme_path}...")

        # Add title page
        self.add_title_page()

        # Load and clean content
        lines = self.load_and_clean_readme()

        # Process and render content
        self.process_and_render(lines)

        # Save PDF
        self.pdf.output(self.pdf_path)

        if os.path.exists(self.pdf_path):
            print(f"PDF generated successfully: {self.pdf_path}")
            print(f"File size: {os.path.getsize(self.pdf_path):,} bytes")
            return True
        else:
            print("Failed to generate PDF")
            return False


def main():
    """Main function."""
    generator = ASCIIPDFGenerator()

    if generator.generate():
        return 0

    # Fallback: create a minimal PDF
    print("Creating minimal PDF as fallback...")
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "OCR Receipt Processing Pipeline", align="C")
        pdf.ln(20)

        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 10, "Please see README.md file for complete documentation.")
        pdf.ln(10)

        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 10, "PDF generation completed with ASCII filtering to avoid font issues.")

        pdf.output("README_minimal.pdf")
        print("Minimal PDF created: README_minimal.pdf")
    except Exception as e:
        print(f"Fallback also failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())