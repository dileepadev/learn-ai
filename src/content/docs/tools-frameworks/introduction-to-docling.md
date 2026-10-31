---
title: Introduction to Docling - Document Conversion for AI Pipelines
description: Learn how Docling converts PDFs, Office files, and images into structured, layout-aware representations that are ready for RAG and LLM pipelines.
---

Docling is an open-source document conversion toolkit that turns messy real-world files (PDF, DOCX, PPTX, HTML, images) into a clean, structured representation suitable for downstream LLM use. Unlike plain text extraction, it preserves document structure such as headings, tables, lists, and reading order.

## Why Structure Matters

Naive PDF text extraction flattens multi-column layouts, scrambles tables, and loses heading hierarchy. Chunking that text for retrieval then splits tables mid-row and separates headings from their content. A layout-aware converter keeps these relationships intact so chunks stay semantically coherent.

## What Docling Does

- **Layout analysis**: detects regions such as titles, paragraphs, figures, tables, and footnotes
- **Table structure recognition**: reconstructs rows, columns, and merged cells
- **Reading order**: restores the correct sequence for multi-column pages
- **OCR**: handles scanned pages and embedded images
- **Unified output**: a single document object that exports to Markdown, JSON, or HTML

## Basic Usage

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("annual_report.pdf")

markdown = result.document.export_to_markdown()
print(markdown[:500])
```

## Chunking with Structure

Because the converted document keeps its hierarchy, you can chunk along section boundaries instead of fixed character windows. Each chunk can carry its heading path as metadata, which improves both retrieval precision and the context given to the model.

## Practical Considerations

- Layout and table models add compute cost; run conversion offline as an ingestion step, not at query time.
- Always spot-check table output on your own documents, since complex financial or scientific tables remain the hardest case.
- Store the source page number with each chunk so answers can cite the original location.
