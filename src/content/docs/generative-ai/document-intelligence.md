---
title: Document Intelligence
description: Learn how AI transforms unstructured documents into structured data. Explore the document understanding stack from OCR through layout analysis to semantic extraction, covering LayoutLM, Donut, DocVQA, table structure recognition, and modern multimodal approaches that process documents as images rather than text strings.
---

**Document intelligence** (also called document AI or document understanding) is the field of extracting structured information from the unstructured documents that dominate real-world information flows: invoices, contracts, forms, research papers, financial reports, and ID documents. Unlike plain text processing, documents carry meaning through both their textual content and their **spatial layout** — a number in the top-right corner of an invoice means something different from the same number in the middle of the page.

Effective document intelligence must integrate three modalities: **text** (what the document says), **layout** (where elements are positioned), and **visual appearance** (fonts, lines, tables, checkboxes, handwriting). Modern approaches use transformers that process all three simultaneously.

## The Document Understanding Stack

A complete document processing pipeline involves several layers:

```
Raw Document (PDF, scan, photo)
        ↓
  OCR Engine (text detection + recognition)
        ↓
  Layout Analysis (reading order, regions, tables)
        ↓
  Semantic Understanding (classification, extraction, VQA)
        ↓
  Structured Output (JSON, database record, API response)
```

Traditional systems built each layer independently, propagating errors downward. Modern end-to-end models like **Donut** bypass OCR entirely — treating documents as images and learning to read them directly.

## LayoutLM: Fusing Text and Spatial Position

**LayoutLM** (Xu et al., Microsoft Research, 2020) extends BERT with 2D positional embeddings derived from bounding box coordinates. Each text token gets not just a text embedding and 1D position, but also x-min, y-min, x-max, y-max embeddings normalized to the page dimensions:

$$\text{Input}_i = \text{TextEmbed}(w_i) + \text{1DPos}(i) + \text{2DPos}(x_1, y_1, x_2, y_2)$$

This gives the model awareness of *where* each word appears — a fundamental feature for distinguishing "Invoice Number:" from "Amount Due:" even if both are short strings.

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

def extract_form_fields_with_layoutlmv3(
    image_path: str,
    processor: LayoutLMv3Processor,
    model: LayoutLMv3ForTokenClassification,
    id2label: dict[int, str]
) -> list[dict[str, str]]:
    """
    Token classification with LayoutLMv3 for named entity recognition
    in documents — identifying key-value pairs like invoice fields.
    
    LayoutLMv3 improvements over v1/v2:
    - Uses patch-based image features (like ViT) instead of CNN features
    - Unified text-image alignment via masked language modeling + masked image modeling
    - No separate OCR required for image patches (though words still need OCR)
    - Supports both text and visual tokens in the same sequence
    
    Common label schema for form NER:
    - B-QUESTION / I-QUESTION: question/label text (e.g., "Invoice Number:")
    - B-ANSWER / I-ANSWER: answer text (e.g., "INV-2024-001")
    - O: other / background text
    """
    image = Image.open(image_path).convert("RGB")
    
    # Processor runs OCR (or accepts pre-computed OCR) and prepares tensors
    # words: list of strings (from OCR)
    # boxes: list of [x0, y0, x1, y1] normalized to [0, 1000]
    encoding = processor(
        image,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    logits = outputs.logits.squeeze(0)          # (seq_len, n_labels)
    predictions = logits.argmax(dim=-1).tolist()
    
    # Decode predictions back to word-level labels
    word_ids = encoding.word_ids(batch_index=0)
    
    current_field = {"label": None, "value_tokens": []}
    extracted_fields = []
    
    prev_word_id = None
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            prev_word_id = word_id
            continue
        
        label = id2label[predictions[token_idx]]
        word = encoding["input_ids"][0][token_idx]  # simplified
        
        if label.startswith("B-QUESTION"):
            if current_field["label"]:
                extracted_fields.append(current_field)
            current_field = {"label": label, "value_tokens": []}
        elif label.startswith("B-ANSWER") or label.startswith("I-ANSWER"):
            current_field["value_tokens"].append(token_idx)
        
        prev_word_id = word_id
    
    if current_field["label"]:
        extracted_fields.append(current_field)
    
    return extracted_fields


def prepare_document_inputs_with_bboxes(
    words: list[str],
    boxes: list[list[int]],     # [[x0, y0, x1, y1], ...] normalized 0-1000
    image: "PIL.Image",
    processor: LayoutLMv3Processor
) -> dict:
    """
    Prepare LayoutLMv3 inputs when OCR output is already available
    (e.g., from a specialized OCR service like Azure Document Intelligence).
    
    boxes must be in LayoutLM's coordinate system: normalized to page size
    where page_width = page_height = 1000. Each box is [x_min, y_min, x_max, y_max].
    
    Example: a token in the top-left quadrant of an A4 page:
        physical box: [10mm, 15mm, 60mm, 25mm]
        normalized:   [47, 71, 283, 118]   (if page = 210mm × 297mm)
    """
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    return encoding
```

## Donut: Document Understanding Transformer (No OCR)

**Donut** (Kim et al., NAVER CLOVA, 2022) takes a radical approach: skip OCR entirely. The model receives the document image as pixel patches (like ViT) and generates structured JSON output autoregressively:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
import json
import re

class DonutDocumentParser:
    """
    End-to-end document parsing with Donut (Document Understanding Transformer).
    
    Architecture:
    - Encoder: Swin Transformer (processes 2560×1920 document images as patches)
    - Decoder: BART-style autoregressive decoder that generates structured text/JSON
    
    Key advantage: No OCR dependency
    - Traditional pipeline errors (low-quality scans, unusual fonts, rotated text)
      don't accumulate before reaching the understanding layer
    - Can handle handwritten text naturally
    - Supports multi-lingual documents without separate OCR models per language
    
    Pre-trained tasks:
    - DocVQA: answer questions about document contents
    - Document classification: categorize document type
    - Key information extraction: parse invoices, receipts, forms to JSON
    """
    
    def __init__(self, model_name: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def answer_document_question(
        self,
        image: Image.Image,
        question: str
    ) -> str:
        """
        DocVQA: answer a natural language question about a document image.
        
        Example questions:
        - "What is the invoice number?"
        - "What is the total amount due?"
        - "Who signed this document?"
        - "What date was this contract executed?"
        """
        # Donut uses a task-specific prompt format
        task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        
        decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids
        
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")
            decoder_input_ids = decoder_input_ids.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=4
            )
        
        sequence = self.processor.batch_decode(outputs)[0]
        
        # Extract answer from between tags
        answer = re.search(r"<s_answer>(.*?)</s_answer>", sequence, re.DOTALL)
        return answer.group(1).strip() if answer else ""

    def parse_invoice_to_json(self, image: Image.Image) -> dict:
        """
        Parse an invoice image to structured JSON without any OCR step.
        
        Uses a Donut model fine-tuned on CORD (Consolidated Receipt Dataset)
        or a custom invoice dataset. Output schema depends on fine-tuning.
        
        Typical output structure:
        {
          "menu": [{"nm": "Item 1", "price": "10.00"}, ...],
          "sub_total": {"subtotal_price": "10.00", "tax_price": "1.00"},
          "total": {"total_price": "11.00"}
        }
        """
        task_prompt = "<s_cord-v2>"  # CORD receipt parsing task
        
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")
            decoder_input_ids = decoder_input_ids.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        sequence = self.processor.batch_decode(outputs)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").strip()
        
        # Convert Donut's markup output to a Python dict
        result = self.processor.token2json(sequence)
        return result
```

## Table Detection and Structure Recognition

Tables in documents present a unique challenge: the structure is visual (grid lines, merged cells, column headers) rather than textual. **TATR** (Table Transformer, Smock et al., 2022) adapts DETR (Detection Transformer) for two tasks:

- **Table detection**: find where tables are on the page (bounding boxes)
- **Table structure recognition**: identify rows, columns, and spanning cells within a detected table

The resulting structure enables converting visual tables to machine-readable formats (CSV, pandas DataFrame) by using OCR on the individual cell regions.

## Benchmarks and Datasets

| Benchmark | Task | State of Art | Key Challenge |
| --- | --- | --- | --- |
| DocVQA | Document visual QA | ~95% ANLS | Long documents, tables, charts |
| InfoVQA | Infographic QA | ~78% ANLS | Complex visual+textual reasoning |
| FUNSD | Form NER (key-value) | ~93% F1 | Irregular layouts, noise |
| CORD | Receipt parsing | ~99% F1 | Multi-currency, line items |
| PubTables-1M | Table structure recognition | ~96% TEDS | Complex merged cells |

## The Shift to Multimodal LLMs

Modern large multimodal models (GPT-4V, Claude 3.5, Gemini 1.5) can directly process document images and perform DocVQA, extraction, and classification via prompting — without fine-tuning. This shifts the tradeoff:

- **Specialized models** (Donut, LayoutLMv3): faster, cheaper, deployable locally, higher accuracy on specific tasks, require labeled data for fine-tuning
- **General MLLMs** (GPT-4o): zero-shot capable, no training data needed, handles unusual layouts, but latency, cost, and data privacy concerns for sensitive documents

Production document intelligence systems today often use specialized models for high-volume structured extraction (invoices, forms) and general MLLMs for low-volume, high-complexity documents (contracts, research papers, correspondence).
