#!/usr/bin/env python3
"""
MBCET CSE Semantic Chunking Pipeline

Main CLI entry point for scraping, chunking, and entity extraction.

Usage:
    python main.py --stage scrape    # Run web scraper
    python main.py --stage chunk     # Run semantic chunker
    python main.py --stage entities  # Build entity registry
    python main.py --stage all       # Run complete pipeline
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def run_scrape_stage():
    """Run the web scraping stage."""
    print("\n" + "=" * 50)
    print("🌐 Starting Web Scraping Stage")
    print("=" * 50)
    
    from scraper.url_discovery import discover_urls
    from scraper.html_scraper import HTMLScraper
    from scraper.pdf_handler import PDFHandler
    
    # Discover URLs
    print("\n📍 Discovering URLs...")
    discovered = discover_urls(max_depth=3, max_urls=100)
    
    print(f"Found {len(discovered['pages'])} pages and {len(discovered['pdfs'])} PDFs")
    
    # Scrape HTML pages
    if discovered['pages']:
        print(f"\n📄 Scraping {len(discovered['pages'])} HTML pages...")
        scraper = HTMLScraper()
        results = scraper.scrape_pages(discovered['pages'])
        print(f"Successfully scraped {len(results)} pages")
        print(f"Output: {config.MARKDOWN_PAGES_DIR}")
    
    # Process PDFs
    if discovered['pdfs']:
        print(f"\n📑 Processing {len(discovered['pdfs'])} PDFs...")
        handler = PDFHandler(use_ocr=True)
        results = handler.process_pdf_urls(discovered['pdfs'])
        print(f"Successfully processed {len(results)} PDFs")
        print(f"Output: {config.MARKDOWN_PDFS_DIR}")
    
    print("\n✅ Scraping stage complete!")


def run_entities_stage():
    """Build entity registry from scraped data."""
    print("\n" + "=" * 50)
    print("🏷️  Building Entity Registry")
    print("=" * 50)
    
    import json
    import re
    from scraper.html_scraper import HTMLScraper
    
    # Scrape CSE department page to extract faculty
    print("\n📍 Extracting faculty from CSE department...")
    scraper = HTMLScraper()
    faculty_list = scraper.extract_faculty_list(config.CSE_DEPARTMENT_URL)
    
    # Build faculty entities
    faculty_entities = []
    for faculty in faculty_list:
        name = faculty['name']
        
        # Generate ID
        name_slug = re.sub(r'[^\w\s]', '', name.lower())
        name_slug = '_'.join(name_slug.split())
        entity_id = f"faculty_{name_slug}"
        
        # Generate aliases
        aliases = [name]
        
        # Remove title for alias
        name_no_title = re.sub(r'^(Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s*', '', name, flags=re.IGNORECASE)
        if name_no_title != name:
            aliases.append(name_no_title.strip())
        
        faculty_entities.append({
            "id": entity_id,
            "name": name,
            "aliases": aliases,
            "type": "faculty",
            "url": faculty.get('url', '')
        })
    
    # Save faculty entities
    config.ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(config.FACULTY_FILE, 'w', encoding='utf-8') as f:
        json.dump(faculty_entities, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(faculty_entities)} faculty entities to {config.FACULTY_FILE}")
    
    # Create empty courses and programs files if they don't exist
    for filepath, name in [(config.COURSES_FILE, "courses"), (config.PROGRAMS_FILE, "programs")]:
        if not filepath.exists():
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print(f"Created empty {name} file: {filepath}")
    
    print("\n✅ Entity registry built!")


def run_chunk_stage():
    """Run the semantic chunking stage."""
    print("\n" + "=" * 50)
    print("✂️  Starting Semantic Chunking Stage")
    print("=" * 50)
    
    from chunker.semantic_chunker import run_chunking_pipeline
    
    chunks, report = run_chunking_pipeline()
    
    if not chunks:
        print("\n⚠️  No chunks generated. Make sure to run scraping first:")
        print("    python main.py --stage scrape")


def run_all_stages():
    """Run complete pipeline."""
    print("\n" + "=" * 50)
    print("🚀 Running Complete Pipeline")
    print("=" * 50)
    
    run_scrape_stage()
    run_entities_stage()
    run_chunk_stage()
    
    print("\n" + "=" * 50)
    print("🎉 Complete pipeline finished!")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="MBCET CSE Semantic Chunking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --stage scrape    # Scrape MBCET website
    python main.py --stage entities  # Build entity registry
    python main.py --stage chunk     # Run semantic chunker
    python main.py --stage all       # Run complete pipeline
        """
    )
    
    parser.add_argument(
        '--stage',
        choices=['scrape', 'entities', 'chunk', 'all'],
        required=True,
        help='Pipeline stage to run'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    config.ensure_directories()
    
    # Run selected stage
    if args.stage == 'scrape':
        run_scrape_stage()
    elif args.stage == 'entities':
        run_entities_stage()
    elif args.stage == 'chunk':
        run_chunk_stage()
    elif args.stage == 'all':
        run_all_stages()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
