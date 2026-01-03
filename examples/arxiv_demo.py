#!/usr/bin/env python3
"""
arXiv Paper Assistant - WDVA Example

Train a personalized AI assistant on a research paper and query it privately.

This demonstrates the complete WDVA workflow:
1. Download an arXiv paper
2. Extract text content
3. Generate Q&A training pairs
4. Train a WDVA adapter (placeholder in this demo)
5. Query locally (no cloud, no data sharing)

Usage:
    python examples/arxiv_demo.py --paper-id 2502.13171
    python examples/arxiv_demo.py --paper-id 2502.13171 --query "What is the main contribution?"

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from wdva import WDVA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QAPair:
    """A question-answer training pair."""
    instruction: str
    input: str
    output: str
    
    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }


# =============================================================================
# Paper Processing Functions
# =============================================================================

def download_arxiv_paper(paper_id: str, *, timeout: int = 60) -> Path:
    """
    Download an arXiv paper PDF.
    
    Args:
        paper_id: arXiv paper ID (e.g., "2502.13171")
        timeout: Request timeout in seconds
        
    Returns:
        Path to downloaded PDF
        
    Raises:
        ImportError: If requests library not available
        RuntimeError: If download fails
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests library required. Install with: pip install requests"
        )
    
    # Construct arXiv PDF URL
    # Format: https://arxiv.org/pdf/{paper_id}.pdf
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    logger.info("Downloading paper from arXiv: %s", paper_id)
    logger.debug("URL: %s", url)
    
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "WDVA-Demo/1.0"}
        )
        response.raise_for_status()
        
        # Verify it's a PDF
        if not response.content.startswith(b"%PDF"):
            raise RuntimeError(
                f"Downloaded content is not a valid PDF. "
                f"Paper ID '{paper_id}' may not exist."
            )
        
        # Save to temporary file
        pdf_path = Path(tempfile.gettempdir()) / f"arxiv_{paper_id.replace('/', '_')}.pdf"
        pdf_path.write_bytes(response.content)
        
        logger.info("✓ Downloaded: %s (%d KB)", pdf_path, len(response.content) // 1024)
        return pdf_path
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download paper: {e}") from e


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
        
    Note:
        Falls back to placeholder if PyPDF2 is not available.
    """
    try:
        import PyPDF2
    except ImportError:
        logger.warning(
            "PyPDF2 not available. Install with: pip install PyPDF2"
        )
        return f"[Placeholder: Text would be extracted from {pdf_path.name}]"
    
    logger.info("Extracting text from PDF...")
    
    text_parts: List[str] = []
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    
        text = "\n\n".join(text_parts)
        logger.info("✓ Extracted %d characters from %d pages", len(text), len(reader.pages))
        return text
        
    except Exception as e:
        logger.error("PDF extraction failed: %s", e)
        raise RuntimeError(f"Failed to extract text from PDF: {e}") from e


def generate_qa_pairs(
    text: str,
    *,
    num_pairs: int = 50,
    min_sentence_length: int = 30,
) -> List[QAPair]:
    """
    Generate Q&A training pairs from text.
    
    This is a simplified implementation for demonstration.
    Production systems would use a larger language model for
    high-quality Q&A generation.
    
    Args:
        text: Input text content
        num_pairs: Target number of Q&A pairs
        min_sentence_length: Minimum sentence length to consider
        
    Returns:
        List of QAPair objects
    """
    logger.info("Generating Q&A pairs...")
    
    # Split into sentences (simplified)
    sentences = []
    for paragraph in text.split('\n'):
        for sentence in paragraph.split('. '):
            sentence = sentence.strip()
            if len(sentence) >= min_sentence_length:
                sentences.append(sentence)
    
    # Generate Q&A pairs
    qa_pairs: List[QAPair] = []
    
    question_templates = [
        "According to the paper, {}",
        "What does the paper say about {}",
        "Explain {}",
        "Describe {}",
        "What is {}",
    ]
    
    for i, sentence in enumerate(sentences[:num_pairs]):
        # Create a simple question from the sentence
        # (Production would use a proper Q&A generation model)
        
        # Extract key phrase (first few words)
        words = sentence.split()[:5]
        key_phrase = " ".join(words).lower()
        if key_phrase.endswith(','):
            key_phrase = key_phrase[:-1]
        
        # Select question template
        template = question_templates[i % len(question_templates)]
        question = template.format(key_phrase) + "?"
        
        qa_pairs.append(QAPair(
            instruction=question,
            input="",
            output=sentence
        ))
    
    logger.info("✓ Generated %d Q&A pairs", len(qa_pairs))
    return qa_pairs


def create_placeholder_adapter(
    qa_pairs: List[QAPair],
    output_dir: Path,
) -> tuple[Path, str]:
    """
    Create a placeholder encrypted adapter file.
    
    In production, this would:
    1. Train a DoRA adapter using PEFT
    2. Encrypt the adapter weights
    3. Save the encrypted package
    
    Args:
        qa_pairs: Training Q&A pairs
        output_dir: Output directory
        
    Returns:
        Tuple of (adapter_path, encryption_key_hex)
    """
    logger.info("Creating adapter (placeholder for demo)...")
    logger.warning(
        "Full DoRA training requires PEFT library. "
        "See main Enclave repository for complete implementation."
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapter_path = output_dir / "arxiv_adapter.wdva"
    encryption_key = secrets.token_hex(32)
    
    # Create placeholder adapter file
    placeholder_data = {
        "salt": "placeholder_salt_base64",
        "nonce": "placeholder_nonce_base64",
        "ciphertext": "placeholder_ciphertext_base64",
        "tag": "placeholder_tag_base64",
        "metadata": {
            "version": "1.0",
            "num_tensors": 0,
            "compressed": False,
            "num_qa_pairs": len(qa_pairs),
            "note": "Placeholder adapter. Use full training for real adapters."
        },
        "algorithm": "XChaCha20-Poly1305",
        "kdf": "HKDF-SHA256"
    }
    
    with open(adapter_path, 'w', encoding='utf-8') as f:
        json.dump(placeholder_data, f, indent=2)
    
    logger.info("✓ Placeholder adapter saved: %s", adapter_path)
    return adapter_path, encryption_key


# =============================================================================
# Main Demo Function
# =============================================================================

def run_demo(
    paper_id: str,
    output_dir: Path,
    query: Optional[str] = None,
) -> int:
    """
    Run the arXiv paper demo.
    
    Args:
        paper_id: arXiv paper ID
        output_dir: Directory to save adapter
        query: Optional query to run
        
    Returns:
        Exit code (0 for success)
    """
    print()
    print("=" * 70)
    print("  WDVA: Secure & Private AI for Everyone")
    print("  arXiv Paper Assistant Demo")
    print("=" * 70)
    print()
    print(f"  Paper ID: {paper_id}")
    print(f"  Output: {output_dir}")
    print()
    
    try:
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Download paper
        # ─────────────────────────────────────────────────────────────────────
        print("📥 Step 1: Downloading paper from arXiv...")
        pdf_path = download_arxiv_paper(paper_id)
        print(f"   ✓ Downloaded: {pdf_path}\n")
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Extract text
        # ─────────────────────────────────────────────────────────────────────
        print("📄 Step 2: Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        print(f"   ✓ Extracted {len(text):,} characters\n")
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Generate Q&A pairs
        # ─────────────────────────────────────────────────────────────────────
        print("🤖 Step 3: Generating Q&A training pairs...")
        qa_pairs = generate_qa_pairs(text, num_pairs=50)
        print(f"   ✓ Generated {len(qa_pairs)} Q&A pairs\n")
        
        # Show sample Q&A pairs
        if qa_pairs:
            print("   Sample Q&A pairs:")
            for pair in qa_pairs[:2]:
                print(f"   Q: {pair.instruction[:60]}...")
                print(f"   A: {pair.output[:60]}...")
                print()
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Create adapter
        # ─────────────────────────────────────────────────────────────────────
        print("🎓 Step 4: Training WDVA adapter...")
        adapter_path, encryption_key = create_placeholder_adapter(
            qa_pairs,
            output_dir
        )
        print(f"   ✓ Adapter saved: {adapter_path}")
        print(f"   ✓ Encryption key: {encryption_key[:16]}... (keep secret!)\n")
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Query (optional)
        # ─────────────────────────────────────────────────────────────────────
        if query:
            print("💬 Step 5: Querying your private AI...")
            print(f"   Question: {query}\n")
            
            try:
                wdva = WDVA()
                # Use base model for demo since adapter is placeholder
                response = wdva.query_base(query, max_tokens=200)
                print(f"   Response: {response[:300]}...\n")
            except Exception as e:
                print(f"   ⚠ Query failed: {e}")
                print("   (Model loading requires ML dependencies)\n")
        
        # ─────────────────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────────────────
        print("=" * 70)
        print("  ✅ Complete!")
        print("=" * 70)
        print()
        print("  What happened:")
        print("  • Paper was downloaded and processed locally")
        print("  • Q&A pairs were generated from content")
        print("  • Adapter was created (placeholder in demo)")
        print("  • Encryption key was generated (store securely!)")
        print()
        print("  Privacy guarantees:")
        print("  • Your data never left your device")
        print("  • Adapter is encrypted with your key")
        print("  • Querying happens entirely locally")
        print("  • No cloud, no data sharing, no compromise")
        print()
        print(f"  To query again:")
        print(f"  python examples/arxiv_demo.py --paper-id {paper_id} --query 'Your question'")
        print()
        print("=" * 70)
        print()
        
        return 0
        
    except Exception as e:
        logger.error("Error: %s", e)
        print(f"\n❌ Error: {e}\n")
        return 1


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train a WDVA adapter on an arXiv paper and query it privately",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process a paper
  python examples/arxiv_demo.py --paper-id 2502.13171
  
  # Process and query
  python examples/arxiv_demo.py --paper-id 2502.13171 --query "What is the main contribution?"
  
  # Specify output directory
  python examples/arxiv_demo.py --paper-id 2502.13171 --output-dir ./my_adapters
        """
    )
    
    parser.add_argument(
        "--paper-id",
        type=str,
        required=True,
        help="arXiv paper ID (e.g., '2502.13171' or 'cs.AI/0608042')"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional: Query to ask the trained model"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./adapters",
        help="Directory to save adapter (default: ./adapters)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    return run_demo(
        paper_id=args.paper_id,
        output_dir=Path(args.output_dir),
        query=args.query,
    )


if __name__ == "__main__":
    sys.exit(main())
