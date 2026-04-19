#!/usr/bin/env python3
"""
MBCET CSE Semantic Chunking & RAG Pipeline

Main CLI entry point for scraping, chunking, entity extraction,
embedding generation, retrieval, and chatbot.

Usage:
    python main.py --stage scrape    # Run web scraper
    python main.py --stage chunk     # Run semantic chunker
    python main.py --stage entities  # Build entity registry
    python main.py --stage kg        # Build canonical knowledge graph (chunker)
    python main.py --stage graph     # Build phase-1 knowledge graph JSON
    python main.py --stage embed     # Embed chunks & ingest into ChromaDB
    python main.py --stage embed --force  # Force re-embedding
    python main.py --stage query --text "Who is the HOD?"
    python main.py --stage chat      # Launch Streamlit chatbot UI
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
    print(" Starting Web Scraping Stage")
    print("=" * 50)
    
    from scraper.url_discovery import discover_urls
    from scraper.html_scraper import HTMLScraper
    from scraper.pdf_handler import PDFHandler
    
    # Discover URLs
    print("\n Discovering URLs...")
    discovered = discover_urls(max_depth=3, max_urls=100)
    
    print(f"Found {len(discovered['pages'])} pages and {len(discovered['pdfs'])} PDFs")
    
    # Scrape HTML pages
    if discovered['pages']:
        print(f"\n Scraping {len(discovered['pages'])} HTML pages...")
        scraper = HTMLScraper()
        results = scraper.scrape_pages(discovered['pages'])
        print(f"Successfully scraped {len(results)} pages")
        print(f"Output: {config.MARKDOWN_PAGES_DIR}")
    
    # Process PDFs
    if discovered['pdfs']:
        print(f"\n Processing {len(discovered['pdfs'])} PDFs...")
        handler = PDFHandler(use_ocr=True)
        results = handler.process_pdf_urls(discovered['pdfs'])
        print(f"Successfully processed {len(results)} PDFs")
        print(f"Output: {config.MARKDOWN_PDFS_DIR}")
    
    print("\n[OK] Scraping stage complete!")


def run_entities_stage():
    """Build entity registry from scraped data."""
    print("\n" + "=" * 50)
    print("  Building Entity Registry")
    print("=" * 50)
    
    import json
    import re
    from scraper.html_scraper import HTMLScraper
    
    # Scrape CSE department page to extract faculty
    print("\n Extracting faculty from CSE department...")
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
    
    print("\n[OK] Entity registry built!")


def run_chunk_stage():
    """Run the semantic chunking stage."""
    print("\n" + "=" * 50)
    print("[*]  Starting Semantic Chunking Stage")
    print("=" * 50)
    
    from chunker.semantic_chunker import run_chunking_pipeline
    
    chunks, report = run_chunking_pipeline()
    
    if not chunks:
        print("\n[WARN]  No chunks generated. Make sure to run scraping first:")
        print("    python main.py --stage scrape")


def run_embed_stage(force: bool = False):
    """Run the embedding & ChromaDB ingestion stage."""
    print("\n" + "=" * 50)
    print(" Starting Embedding & Ingestion Stage")
    print("=" * 50)

    from rag_ingestion import run_ingestion_pipeline

    run_ingestion_pipeline(force_reembed=force)


def run_kg_stage():
    """Build the canonical knowledge graph from entity registries."""
    print("\n" + "=" * 50)
    print(" Building Canonical Knowledge Graph")
    print("=" * 50)

    from chunker.knowledge_graph import build_and_save_knowledge_graph

    graph = build_and_save_knowledge_graph(config.DATA_DIR)
    summary = graph.get("summary", {})
    print(f"Nodes: {summary.get('total_nodes', 0)}")
    print(f"Edges: {summary.get('total_edges', 0)}")
    print("\n[OK] Canonical knowledge graph built!")


def run_graph_stage():
    """Run the phase-1 knowledge graph stage."""
    print("\n" + "=" * 50)
    print(" Starting Phase-1 Knowledge Graph Stage")
    print("=" * 50)

    from knowledge_graph.builder import run_knowledge_graph_pipeline

    run_knowledge_graph_pipeline()


def run_query_stage(query_text: str):
    """Run a query against the ChromaDB knowledge base."""
    from rag_ingestion import run_query

    run_query(query_text)


def run_chat_stage():
    """Launch the Streamlit chatbot UI."""
    import subprocess

    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        print("[ERR] app.py not found. Cannot launch chatbot.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(" Launching MBCET CSE Chatbot")
    print("=" * 50)
    print(f"Starting Streamlit server...")
    print(f"Press Ctrl+C to stop.\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.headless", "true"],
        cwd=str(Path(__file__).parent),
    )


def run_serve_stage():
    """Run the FastAPI-based HTTP server for the chatbot using uvicorn.

    This provides a lightweight replacement for the Streamlit frontend
    and exposes `POST /chat` for the single-file HTML widget to call.
    """
    import subprocess

    print("\n" + "=" * 50)
    print(" Starting FastAPI server (api_server:app) on http://127.0.0.1:8000")
    print("=" * 50)

    try:
        import uvicorn
        uvicorn.run("api_server:app", host="127.0.0.1", port=8000, log_level="info")
    except Exception:
        # Fallback to subprocess invocation of uvicorn
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(Path(__file__).parent),
        )


def run_all_stages(force: bool = False):
    """Run complete pipeline."""
    print("\n" + "=" * 50)
    print(" Running Complete Pipeline")
    print("=" * 50)

    run_scrape_stage()
    run_entities_stage()
    run_kg_stage()
    run_chunk_stage()
    # Graph stage depends on entities + chunks but is independent of embeddings.
    run_graph_stage()
    run_embed_stage(force=force)

    print("\n" + "=" * 50)
    print(" Complete pipeline finished!")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="MBCET CSE Semantic Chunking & RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --stage scrape    # Scrape MBCET website
    python main.py --stage entities  # Build entity registry
    python main.py --stage chunk     # Run semantic chunker
    python main.py --stage kg        # Build canonical knowledge graph
    python main.py --stage graph     # Build phase-1 knowledge graph JSON
    python main.py --stage embed     # Embed chunks & ingest into ChromaDB
    python main.py --stage embed --force  # Force re-embedding
    python main.py --stage query --text "Who is the HOD?"
    python main.py --stage chat      # Launch Streamlit chatbot
    python main.py --stage all       # Run complete pipeline
        """
    )

    parser.add_argument(
        '--stage',
        choices=['scrape', 'entities', 'chunk', 'kg', 'graph', 'embed', 'query', 'chat', 'serve', 'all'],
        required=True,
        help='Pipeline stage to run'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-embedding (deletes cache and recreates ChromaDB collection)'
    )

    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Query text for --stage query'
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
    elif args.stage == 'kg':
        run_kg_stage()
    elif args.stage == 'embed':
        run_embed_stage(force=args.force)
    elif args.stage == 'graph':
        run_graph_stage()
    elif args.stage == 'query':
        if not args.text:
            print("[ERR] --text is required for query stage")
            print("   Example: python main.py --stage query --text \"Who is the HOD?\"")
            sys.exit(1)
        run_query_stage(args.text)
    elif args.stage == 'chat':
        run_chat_stage()
    elif args.stage == 'serve':
        run_serve_stage()
    elif args.stage == 'all':
        run_all_stages(force=args.force)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
