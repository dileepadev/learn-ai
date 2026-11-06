---
title: Introduction to Unstructured - Preprocessing Documents for LLMs
description: Learn how the Unstructured library partitions heterogeneous documents into typed elements and prepares them for chunking, embedding, and retrieval.
---

Unstructured is an open-source library for ingesting and preprocessing documents of many formats into a normalized list of typed elements. It is a common first stage in RAG pipelines where the corpus mixes PDFs, emails, slide decks, HTML pages, and spreadsheets.

## Core Idea: Partition into Elements

Rather than returning one blob of text, Unstructured *partitions* a file into elements such as `Title`, `NarrativeText`, `ListItem`, `Table`, and `Image`. Each element carries metadata (page number, filename, coordinates, parent section).

```python
from unstructured.partition.auto import partition

elements = partition(filename="handbook.pdf")
for el in elements[:5]:
    print(type(el).__name__, "|", el.text[:60])
```

`partition` inspects the file type and routes to the right parser automatically.

## Chunking

Elements can be grouped into chunks that respect structure:

```python
from unstructured.chunking.title import chunk_by_title

chunks = chunk_by_title(elements, max_characters=1000, combine_text_under_n_chars=200)
```

`chunk_by_title` starts a new chunk at each section heading and merges tiny fragments, so related content stays together.

## Processing Strategies

| Strategy | Behavior | Use when |
|----------|----------|----------|
| `fast` | Extracts text directly from digital PDFs | Clean, text-based files |
| `hi_res` | Uses layout detection models | Complex layouts, tables |
| `ocr_only` | Runs OCR on page images | Scanned documents |

## Tips

- Keep element metadata through embedding so you can filter and cite by source.
- Use `hi_res` selectively because it is far slower than `fast`.
- Deduplicate repeated headers and footers, which otherwise pollute retrieval results.
