"""
PDF Handler for MBCET documents.
Downloads PDFs and converts them to Markdown with page markers.
Supports both text-based and scanned (image) PDFs using OCR.
Enhanced with pdfplumber for table extraction.
"""

import logging
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from io import BytesIO

import requests
from pypdf import PdfReader

# Table extraction with pdfplumber
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

import config

logger = logging.getLogger(__name__)

# Optional OCR imports
try:
    from pdf2image import convert_from_path, convert_from_bytes
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR dependencies not installed. Install pdf2image, Pillow, pytesseract for OCR support.")


class PDFHandler:
    """
    Handles PDF downloading and conversion to Markdown.
    Supports OCR for scanned/image-based PDFs.
    """

    def __init__(self, use_ocr: bool = True):
        """
        Initialize PDF handler.
        
        Args:
            use_ocr: Whether to use OCR for image-based PDFs
        """
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})
        self.use_ocr = use_ocr and OCR_AVAILABLE
        
        if use_ocr and not OCR_AVAILABLE:
            logger.warning("OCR requested but dependencies not available")

    def download_pdf(self, url: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Download a PDF file.
        
        Args:
            url: URL of the PDF
            output_dir: Directory to save the PDF
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        output_dir = output_dir or config.RAW_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading PDF: {url}")
        
        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.session.get(
                    url,
                    timeout=config.REQUEST_TIMEOUT,
                    stream=True
                )
                response.raise_for_status()
                
                # Generate filename from URL
                filename = url.split('/')[-1]
                if not filename.endswith('.pdf'):
                    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                    filename = f"document_{url_hash}.pdf"
                
                output_path = output_dir / filename
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded: {output_path}")
                return output_path
                
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.REQUEST_DELAY * (attempt + 1))
        
        logger.error(f"Failed to download {url}")
        return None

    def _extract_text_pypdf(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        Extract text from PDF using pypdf.
        
        Returns:
            List of (page_number, text) tuples
        """
        pages = []
        try:
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append((i + 1, text.strip()))
        except Exception as e:
            logger.error(f"pypdf extraction failed for {pdf_path}: {e}")
        return pages

    def _extract_text_ocr(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        Extract text from PDF using OCR (for scanned documents).
        
        Returns:
            List of (page_number, text) tuples
        """
        if not self.use_ocr:
            return []
        
        pages = []
        try:
            # Convert PDF pages to images
            images = convert_from_path(str(pdf_path), dpi=300)
            
            for i, image in enumerate(images):
                # Perform OCR on each page image
                text = pytesseract.image_to_string(image, lang='eng')
                pages.append((i + 1, text.strip()))
                logger.debug(f"OCR completed for page {i + 1}")
                
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
        
        return pages

    def _is_text_based(self, pages: List[Tuple[int, str]], min_words: int = 50) -> bool:
        """
        Check if PDF has sufficient extracted text.
        
        Args:
            pages: List of (page_number, text) tuples
            min_words: Minimum word count to consider as text-based
            
        Returns:
            True if PDF appears to be text-based
        """
        total_words = sum(len(text.split()) for _, text in pages)
        return total_words >= min_words

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert a table (list of rows) to Markdown table format.
        
        Args:
            table: List of rows, each row is a list of cell values
            
        Returns:
            Markdown table string
        """
        if not table or len(table) < 1:
            return ""
        
        # Clean cell values
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                # Handle None and clean whitespace
                cell_str = str(cell) if cell is not None else ""
                cell_str = ' '.join(cell_str.split())  # Normalize whitespace
                cell_str = cell_str.replace('|', '\\|')  # Escape pipes
                cleaned_row.append(cell_str)
            cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return ""
        
        # Determine column count (max across all rows)
        max_cols = max(len(row) for row in cleaned_table)
        
        # Pad rows to have consistent column count
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        # Build markdown table
        lines = []
        
        # Header row
        header = cleaned_table[0]
        lines.append('| ' + ' | '.join(header) + ' |')
        
        # Separator row
        lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
        
        # Data rows
        for row in cleaned_table[1:]:
            lines.append('| ' + ' | '.join(row) + ' |')
        
        return '\n'.join(lines)

    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text and tables from PDF using pdfplumber.
        
        Returns:
            List of page data dictionaries with 'page_num', 'text', and 'tables'
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available, skipping table extraction")
            return []
        
        pages_data = []
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_data = {
                        'page_num': i + 1,
                        'text': '',
                        'tables': []
                    }
                    
                    # Extract tables first
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            if table and len(table) > 0:
                                md_table = self._table_to_markdown(table)
                                if md_table:
                                    page_data['tables'].append(md_table)
                    
                    # Extract text (this includes text not in tables)
                    text = page.extract_text() or ""
                    page_data['text'] = text.strip()
                    
                    pages_data.append(page_data)
                    logger.debug(f"pdfplumber: page {i + 1} - {len(page_data['tables'])} tables found")
                    
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
        
        return pages_data

    def pdf_to_markdown(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
        force_ocr: bool = False
    ) -> Optional[Path]:
        """
        Convert PDF to Markdown with page markers.
        Uses pdfplumber for table extraction when available.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save Markdown file
            force_ocr: Force OCR even if text extraction works
            
        Returns:
            Path to saved Markdown file or None if failed
        """
        output_dir = output_dir or config.MARKDOWN_PDFS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting PDF to Markdown: {pdf_path}")
        
        # Try pdfplumber first for better table extraction
        pdfplumber_data = []
        if PDFPLUMBER_AVAILABLE and not force_ocr:
            pdfplumber_data = self._extract_with_pdfplumber(pdf_path)
            total_tables = sum(len(p['tables']) for p in pdfplumber_data)
            logger.info(f"pdfplumber extracted {len(pdfplumber_data)} pages with {total_tables} tables")
        
        # Check if pdfplumber got enough content
        pdfplumber_has_content = False
        if pdfplumber_data:
            total_words = sum(len(p['text'].split()) for p in pdfplumber_data)
            pdfplumber_has_content = total_words >= 50
        
        # Fall back to pypdf/OCR if pdfplumber didn't work well
        use_legacy = False
        pages = []
        if not pdfplumber_has_content:
            logger.info("pdfplumber extraction insufficient, trying pypdf...")
            pages = self._extract_text_pypdf(pdf_path)
            
            if force_ocr or not self._is_text_based(pages):
                logger.info(f"Insufficient text extracted, trying OCR for {pdf_path}")
                if self.use_ocr:
                    ocr_pages = self._extract_text_ocr(pdf_path)
                    if self._is_text_based(ocr_pages):
                        pages = ocr_pages
                        logger.info("OCR extraction successful")
                    else:
                        logger.warning(f"OCR also failed to extract sufficient text from {pdf_path}")
                else:
                    logger.warning("OCR not available, using incomplete text extraction")
            use_legacy = True
        
        # Check if we have any content
        if not pdfplumber_data and not pages:
            logger.error(f"No text extracted from {pdf_path}")
            return None
        
        # Build Markdown content
        markdown_lines = []
        
        # Add frontmatter
        markdown_lines.append("---")
        markdown_lines.append(f'title: "{pdf_path.stem}"')
        markdown_lines.append(f'source_file: "{pdf_path.name}"')
        markdown_lines.append("source_type: pdf")
        markdown_lines.append("---")
        markdown_lines.append("")
        
        # Add title
        markdown_lines.append(f"# {pdf_path.stem}")
        markdown_lines.append("")
        
        if use_legacy:
            # Legacy mode: just text, no table preservation
            for page_num, text in pages:
                if text:
                    markdown_lines.append(f"<!-- page: {page_num} -->")
                    markdown_lines.append("")
                    
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        cleaned = ' '.join(para.split())
                        if cleaned:
                            markdown_lines.append(cleaned)
                            markdown_lines.append("")
        else:
            # pdfplumber mode: preserve tables
            for page_data in pdfplumber_data:
                page_num = page_data['page_num']
                text = page_data['text']
                tables = page_data['tables']
                
                if text or tables:
                    markdown_lines.append(f"<!-- page: {page_num} -->")
                    markdown_lines.append("")
                    
                    # Add tables first (they're usually the structured content)
                    for table_md in tables:
                        markdown_lines.append(table_md)
                        markdown_lines.append("")
                    
                    # Add remaining text
                    if text:
                        paragraphs = text.split('\n\n')
                        for para in paragraphs:
                            cleaned = ' '.join(para.split())
                            if cleaned:
                                markdown_lines.append(cleaned)
                                markdown_lines.append("")
        
        # Save to file
        output_filename = pdf_path.stem + ".md"
        output_path = output_dir / output_filename
        
        try:
            content = '\n'.join(markdown_lines)
            output_path.write_text(content, encoding='utf-8')
            logger.info(f"Saved Markdown: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            return None

    def process_pdf_url(
        self,
        url: str,
        pdf_dir: Optional[Path] = None,
        markdown_dir: Optional[Path] = None,
        force_ocr: bool = False
    ) -> Optional[Path]:
        """
        Download PDF from URL and convert to Markdown.
        
        Args:
            url: URL of the PDF
            pdf_dir: Directory to save raw PDF
            markdown_dir: Directory to save Markdown
            force_ocr: Force OCR processing
            
        Returns:
            Path to Markdown file or None if failed
        """
        # Download PDF
        pdf_path = self.download_pdf(url, pdf_dir)
        if pdf_path is None:
            return None
        
        # Convert to Markdown
        return self.pdf_to_markdown(pdf_path, markdown_dir, force_ocr)

    def process_pdf_urls(
        self,
        urls: List[str],
        pdf_dir: Optional[Path] = None,
        markdown_dir: Optional[Path] = None
    ) -> dict:
        """
        Process multiple PDF URLs.
        
        Returns:
            Dictionary mapping URLs to output Markdown paths
        """
        results = {}
        
        for i, url in enumerate(urls):
            logger.info(f"Processing PDF {i + 1}/{len(urls)}: {url}")
            
            output_path = self.process_pdf_url(url, pdf_dir, markdown_dir)
            if output_path:
                results[url] = str(output_path)
            
            # Rate limiting
            if i < len(urls) - 1:
                time.sleep(config.REQUEST_DELAY)
        
        logger.info(f"Processed {len(results)}/{len(urls)} PDFs successfully")
        return results


def download_pdf(url: str, output_path: Optional[str] = None) -> Optional[str]:
    """Convenience function to download a PDF."""
    handler = PDFHandler()
    output_dir = Path(output_path).parent if output_path else None
    result = handler.download_pdf(url, output_dir)
    return str(result) if result else None


def pdf_to_markdown(pdf_path: str) -> Optional[str]:
    """Convenience function to convert PDF to Markdown."""
    handler = PDFHandler()
    result = handler.pdf_to_markdown(Path(pdf_path))
    return str(result) if result else None


if __name__ == "__main__":
    # Test PDF handling
    logging.basicConfig(level=logging.INFO)
    
    # Check OCR availability
    if OCR_AVAILABLE:
        print("OCR is available")
    else:
        print("OCR is NOT available - install pdf2image, Pillow, pytesseract")
    
    # Test with a sample PDF URL if provided
    import sys
    if len(sys.argv) > 1:
        pdf_url = sys.argv[1]
        handler = PDFHandler()
        result = handler.process_pdf_url(pdf_url)
        if result:
            print(f"Converted to: {result}")
        else:
            print("Conversion failed")
