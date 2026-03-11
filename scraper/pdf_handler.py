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

    def _clean_cell(self, cell) -> str:
        """Clean a cell value: handle None, normalize whitespace, escape pipes."""
        cell_str = str(cell) if cell is not None else ""
        cell_str = ' '.join(cell_str.split())  # Normalize whitespace
        cell_str = cell_str.replace('|', '\\|')  # Escape pipes
        return cell_str

    def _rows_to_markdown(self, rows: List[List[str]], header_row: int = 0) -> str:
        """
        Convert a list of rows to Markdown table format.
        
        Args:
            rows: List of rows, each row is a list of cell values
            header_row: Index of the header row (default: 0)
            
        Returns:
            Markdown table string
        """
        if not rows or len(rows) < 1:
            return ""
        
        # Clean all cells
        cleaned = [[self._clean_cell(c) for c in row] for row in rows]
        
        # Determine column count (max across rows)
        max_cols = max(len(row) for row in cleaned)
        
        # Pad rows to consistent column count
        for row in cleaned:
            while len(row) < max_cols:
                row.append("")
        
        lines = []
        header = cleaned[header_row]
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
        for i, row in enumerate(cleaned):
            if i != header_row:
                lines.append('| ' + ' | '.join(row) + ' |')
        
        return '\n'.join(lines)

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert a table (list of rows) to Markdown table format."""
        return self._rows_to_markdown(table, header_row=0)

    def _is_timetable_pdf(self, pdf_path: Path) -> bool:
        """Check if a PDF is a timetable based on filename."""
        name = pdf_path.stem.lower()
        return 'tt_' in name or 'time' in name or 'timetable' in name

    def _extract_timetable_markdown(self, pdf_path: Path) -> Optional[str]:
        """
        Specialized extraction for timetable PDFs.
        
        Timetable PDFs have a complex layout with merged cells that pdfplumber
        extracts as a single large table. This method splits it into 3 logical
        sections:
        1. Schedule Grid (Day x Period)
        2. Course Legend (Slot, Category, Course Code, etc.)
        3. Advisor Info (Sl.No, Title, Name)
        
        Returns:
            Formatted Markdown string or None if extraction fails
        """
        if not PDFPLUMBER_AVAILABLE:
            return None
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page = pdf.pages[0]
                tables = page.extract_tables()
                
                if not tables:
                    logger.warning(f"No tables found in timetable PDF: {pdf_path}")
                    return None
                
                raw_table = tables[0]
                if not raw_table:
                    return None
                
                # Clean all rows
                cleaned_rows = []
                for row in raw_table:
                    cleaned = [self._clean_cell(c) for c in row]
                    cleaned_rows.append(cleaned)
                
                # --- Parse the table into logical sections ---
                # Find key section boundaries by scanning row content
                schedule_rows = []
                course_rows = []
                advisor_rows = []
                metadata_header = ""
                footer_text = ""
                
                section = 'header'  # header -> schedule -> note -> course -> advisor -> footer
                
                for row in cleaned_rows:
                    joined = ' '.join(row).strip()
                    
                    # Skip entirely empty rows
                    if not joined:
                        continue
                    
                    # Detect header/metadata (college name, department, etc.)
                    if 'MAR BASELIOS' in joined.upper() or 'DEPARTMENT' in joined.upper():
                        if 'Academic Year' in joined or 'DEPARTMENT' in joined:
                            metadata_header = joined
                            continue
                    
                    # Detect schedule grid rows (Time headers and day rows)
                    first_cell = row[0].strip() if row[0] else ""
                    if first_cell.startswith('Time') or first_cell.startswith('Day'):
                        schedule_rows.append(row)
                        section = 'schedule'
                        continue
                    if first_cell in ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'):
                        schedule_rows.append(row)
                        section = 'schedule'
                        continue
                    
                    # Detect the note about periods 1 and 8
                    if 'Periods 1 an' in joined or 'Honours/Minors' in joined:
                        section = 'note'
                        continue
                    
                    # Detect course legend section
                    if first_cell == 'Slot' and 'Category' in joined:
                        course_rows.append(row)
                        section = 'course'
                        continue
                    if section == 'course' and first_cell in ('A','B','C','D','E','F','G','H','S','T','M','R/M','R',''):
                        # Check if this is still a course row (has meaningful content)
                        non_empty = sum(1 for c in row if c.strip())
                        if non_empty >= 2 and first_cell != '':
                            course_rows.append(row)
                            continue
                        elif first_cell == '' and non_empty >= 3:
                            course_rows.append(row)
                            continue
                    
                    # Detect advisor section
                    if first_cell == 'Sl.No' or (first_cell.isdigit() and 'Advisor' in joined):
                        advisor_rows.append(row)
                        section = 'advisor'
                        continue
                    if section == 'advisor' and first_cell.isdigit():
                        advisor_rows.append(row)
                        continue
                    
                    # Detect footer (Published on, Prepared by)
                    if 'Published on' in joined or 'Prepared by' in joined or 'HoD' in joined:
                        footer_text = joined
                        continue
                
                # --- Build clean markdown ---
                lines = []
                
                # Extract metadata from the header blob
                if metadata_header:
                    # Parse out key fields
                    meta = metadata_header
                    semester = ""
                    room = ""
                    year = ""
                    dates = ""
                    
                    import re
                    sem_match = re.search(r'Semester:\s*([\w\-\s]+?)(?:\s*Room|$)', meta)
                    if sem_match:
                        semester = sem_match.group(1).strip()
                    room_match = re.search(r'Room\s*No:\s*(\S+)', meta)
                    if room_match:
                        room = room_match.group(1).strip()
                    year_match = re.search(r'Academic Year:\s*([\d\-]+\s*\w+\s*SEMESTER)', meta)
                    if year_match:
                        year = year_match.group(1).strip()
                    start_match = re.search(r'Start date:\s*([\d/]+)', meta)
                    end_match = re.search(r'End date:\s*([\d/]+)', meta)
                    if start_match and end_match:
                        dates = f"{start_match.group(1)} to {end_match.group(1)}"
                    
                    lines.append("**MAR BASELIOS COLLEGE OF ENGINEERING AND TECHNOLOGY (AUTONOMOUS)**")
                    lines.append("**Department of Computer Science and Engineering**")
                    lines.append("")
                    if year:
                        lines.append(f"**Academic Year:** {year}")
                    if semester:
                        lines.append(f"**Semester:** {semester}")
                    if room:
                        lines.append(f"**Room No:** {room}")
                    if dates:
                        lines.append(f"**Duration:** {dates}")
                    lines.append("")
                
                # Schedule Grid Section
                if schedule_rows:
                    lines.append("## Class Schedule")
                    lines.append("")
                    
                    # Build a clean schedule table
                    # Find the time header row and day rows
                    time_headers = []
                    day_rows = []
                    
                    for row in schedule_rows:
                        first = row[0].strip()
                        if first.startswith('Time') or first.startswith('Day'):
                            time_headers.append(row)
                        else:
                            day_rows.append(row)
                    
                    # Use the first time header as column headers
                    if time_headers:
                        # Build a simplified schedule table
                        # Columns: Day, Period 2, Period 3, Period 4, LUNCH, Period 5, Period 6, Period 7
                        header = time_headers[0]
                        
                        # Build the full table with header + day rows
                        all_rows = [header] + day_rows
                        md_table = self._rows_to_markdown(all_rows, header_row=0)
                        lines.append(md_table)
                    else:
                        md_table = self._rows_to_markdown(schedule_rows, header_row=0)
                        lines.append(md_table)
                    
                    lines.append("")
                    lines.append("*Periods 1 and 8 are used for Honours/Minors/Remedial/Extra Classes*")
                    lines.append("")
                
                # Course Legend Section
                if course_rows:
                    lines.append("## Course Details")
                    lines.append("")
                    
                    # Find meaningful columns from the header
                    # Typical columns: Slot, Category, Course Code, Course Name, TT code, Faculty Name, Remarks
                    # But pdfplumber splits them across 14 columns with many empty ones
                    
                    # Compact the rows: remove columns that are always empty
                    if course_rows:
                        num_cols = max(len(r) for r in course_rows)
                        # Check which columns have any content
                        col_has_content = [False] * num_cols
                        for row in course_rows:
                            for j, cell in enumerate(row):
                                if cell.strip():
                                    col_has_content[j] = True
                        
                        # Keep only columns that have content
                        compacted = []
                        for row in course_rows:
                            new_row = []
                            for j, cell in enumerate(row):
                                if j < len(col_has_content) and col_has_content[j]:
                                    new_row.append(cell)
                            compacted.append(new_row)
                        
                        md_table = self._rows_to_markdown(compacted, header_row=0)
                        lines.append(md_table)
                    lines.append("")
                
                # Advisor Section  
                if advisor_rows:
                    lines.append("## Class Advisors")
                    lines.append("")
                    
                    # Compact advisor rows too
                    if advisor_rows:
                        num_cols = max(len(r) for r in advisor_rows)
                        col_has_content = [False] * num_cols
                        for row in advisor_rows:
                            for j, cell in enumerate(row):
                                if cell.strip():
                                    col_has_content[j] = True
                        
                        compacted = []
                        for row in advisor_rows:
                            new_row = []
                            for j, cell in enumerate(row):
                                if j < len(col_has_content) and col_has_content[j]:
                                    new_row.append(cell)
                            compacted.append(new_row)
                        
                        md_table = self._rows_to_markdown(compacted, header_row=0)
                        lines.append(md_table)
                    lines.append("")
                
                # Footer
                if footer_text:
                    lines.append(f"*{footer_text}*")
                    lines.append("")
                
                return '\n'.join(lines)
                
        except Exception as e:
            logger.error(f"Timetable extraction failed for {pdf_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

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
        
        # Check if this is a timetable PDF — use specialized extraction
        if self._is_timetable_pdf(pdf_path) and PDFPLUMBER_AVAILABLE and not force_ocr:
            logger.info(f"Detected timetable PDF, using specialized extraction: {pdf_path.name}")
            tt_md = self._extract_timetable_markdown(pdf_path)
            if tt_md:
                # Build complete markdown with frontmatter
                markdown_lines = []
                markdown_lines.append("---")
                markdown_lines.append(f'title: "{pdf_path.stem}"')
                markdown_lines.append(f'source_file: "{pdf_path.name}"')
                markdown_lines.append("source_type: pdf")
                markdown_lines.append("---")
                markdown_lines.append("")
                markdown_lines.append(f"# {pdf_path.stem}")
                markdown_lines.append("")
                markdown_lines.append("<!-- page: 1 -->")
                markdown_lines.append("")
                markdown_lines.append(tt_md)
                
                output_filename = pdf_path.stem + ".md"
                output_path = output_dir / output_filename
                try:
                    content = '\n'.join(markdown_lines)
                    output_path.write_text(content, encoding='utf-8')
                    logger.info(f"Saved timetable Markdown: {output_path}")
                    return output_path
                except Exception as e:
                    logger.error(f"Failed to save timetable {output_path}: {e}")
                    # Fall through to regular extraction
            else:
                logger.warning(f"Timetable extraction returned no content, falling back to regular: {pdf_path.name}")
        
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
                    
                    # Only add remaining text if no tables were found
                    # (when tables exist, the text is usually a duplicate)
                    if text and not tables:
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
