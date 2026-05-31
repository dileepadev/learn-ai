---
title: "Introduction to LlamaParse: Parsing Documents for LLM Applications"
description: "Learn how LlamaParse extracts structured data from complex documents — PDFs, images, tables — optimized for RAG pipelines and LLM document understanding."
---

LlamaParse is a document parsing service by Meta that extracts clean, structured content from complex documents. It's designed specifically for RAG pipelines, where clean text extraction is essential for retrieval quality.

## Why Document Parsing Matters for RAG

Most real-world data lives in unstructured documents:

- **PDFs**: Research papers, reports, contracts, invoices.
- **Presentations**: Slides with text, images, and tables.
- **Scanned documents**: Images that need OCR.
- **Mixed content**: Documents with text, tables, and figures combined.

Standard PDF libraries extract raw text without structure. LlamaParse preserves document semantics — headings, tables, figures, and their relationships.

## Getting Started with LlamaParse

```python
from llama_parse import LlamaParse

# Initialize the parser
parser = LlamaParse(
    api_key="your-api-key",
    num_workers=4,          # Parallel parsing workers
    verbose=True,           # Progress logging
    language="en",          # Document language
)

# Parse a single document
documents = parser.load_data("path/to/document.pdf")

# Parse multiple documents
documents = parser.load_and_split("path/to/documents/")
```

## Parsing Results

LlamaParse returns structured `Document` objects:

```python
for doc in documents:
    print(f"=== Page {doc.metadata['page_num']} ===")
    print(doc.text[:500])  # First 500 characters
    print(f"Elements: {doc.metadata['total_elements']}")
```

The output includes:
- Clean text with preserved headings and structure.
- Element types (text, table, figure, header, footer).
- Page numbers and positions.
- Table structures (not just plain text).

## Handling Tables

Tables are often the most valuable content in documents. LlamaParse extracts them in multiple formats:

```python
# Get tables separately
tables = parser.get_tables(documents)

for table in tables:
    print(f"Table on page {table.metadata['page_num']}")
    print(f"Rows: {table.rows}, Columns: {table.cols}")
    print(table.to_csv())  # Export as CSV
    print(table.to_markdown())  # Export as Markdown
```

| Format | Use Case |
|--------|----------|
| CSV | Processing in pandas |
| Markdown | LLM context |
| HTML | Web display |
| XML | Structured processing |

## Multimodal Parsing

LlamaParse can extract and describe images within documents:

```python
documents = parser.load_data(
    "document_with_images.pdf",
    extract_images=True,    # Extract embedded images
    image_output_path="./extracted_images"
)

for doc in documents:
    for img in doc.images:
        print(f"Image: {img.filename}")
        print(f"Description: {img.caption}")
        print(f"Page: {img.page_num}")
```

This is valuable for:
- Extracting charts and graphs for analysis.
- Processing visual documents like infographics.
- Building multimodal RAG systems.

## Optimizing for RAG

### Chunking Strategies

```python
from llama_parse.utils import SplitByPage, SplitByHeading, SplitBySize

# Split by pages (default)
documents = parser.load_and_split("doc.pdf", SplitByPage())

# Split by headings (preserves document structure)
documents = parser.load_and_split("doc.pdf", SplitByHeading())

# Split by size for consistent chunk lengths
documents = parser.load_and_split("doc.pdf", SplitBySize(max_size=1024))
```

### Preserving Context

```python
# Enable enhanced context for better chunk coherence
documents = parser.load_data(
    "doc.pdf",
    enhance_context=True,      # Add surrounding context to chunks
    window_size=3,             # Include 3 surrounding chunks
    combine_text_under_n_chars=200  # Merge short chunks
)
```

## Integration with RAG Frameworks

### LangChain

```python
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter

parser = LlamaParse(api_key="...", mode="text", result_type="markdown")

# Parse and split
raw_docs = parser.load_and_split("document.pdf")

# Further split for embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(raw_docs)
```

### LlamaIndex

```python
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex

parser = LlamaParse(api_key="...", result_type="markdown")

# Parse directly into LlamaIndex
documents = parser.load_data("document.pdf")
index = VectorStoreIndex.from_documents(documents)
```

## Handling Common Challenges

### Large Documents

```python
# Process large documents efficiently
parser = LlamaParse(
    api_key="...",
    num_workers=8,              # More workers for parallelism
    check_interval=100,         # Check progress every 100 pages
    max_consecutive_failures=5, # Handle OCR failures gracefully
)

# Resume from checkpoint if interrupted
documents = parser.load_data(
    "large_document.pdf",
    resume_on_failure=True
)
```

### Multilingual Documents

```python
# Specify document language for better OCR
documents = parser.load_data(
    "chinese_document.pdf",
    language="zh",
    ocr_strategy="auto",        # Auto-detect if mixed languages
)

# For Japanese, Korean, Chinese
parser = LlamaParse(
    language="ja",  # or "ko", "zh"
    detect_language_multi=True  # Handle mixed-language docs
)
```

## Performance and Cost

### Pricing Model

LlamaParse is priced per document page. Costs vary by:
- Document complexity (tables, images increase processing).
- Processing mode (enhanced parsing costs more).
- Volume commitments.

### Optimization Tips

```python
# Use faster mode when structure is simple
documents = parser.load_data(
    "simple_doc.pdf",
    mode="fast",           # Skip enhanced processing
    result_type="text"     # Plain text, no markdown
)

# For complex docs, the extra processing is worth it
documents = parser.load_data(
    "complex_report.pdf",
    mode="high_res",       # Better table/image extraction
    result_type="markdown",
    skip_diagonal_text=True  # Ignore marginal notes
)
```

## Alternative: Building Your Own Pipeline

For simple documents, standard PDF extraction may suffice:

```python
# PyMuPDF for simple text extraction
import fitz

def extract_simple_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

However, LlamaParse is superior for:
- Complex layouts with multiple columns.
- Tables that need structure preservation.
- Documents with images and figures.
- Large-scale document processing pipelines.

Document parsing is often overlooked but critical for RAG quality. Poor extraction leads to poor retrieval, regardless of how good your embedding model is. LlamaParse provides production-grade parsing that unlocks the data in your document repositories.