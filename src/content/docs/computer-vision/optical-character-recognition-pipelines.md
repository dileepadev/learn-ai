---
title: Optical Character Recognition Pipelines
description: Learn how modern OCR systems detect text regions, recognize characters, and assemble results, and how end-to-end vision-language models are changing the pipeline.
---

Optical character recognition (OCR) converts images of text, such as scans, photos, and screenshots, into machine-readable characters. It is the entry point to document digitization, invoice processing, and many RAG ingestion pipelines.

## The Classic Two-Stage Pipeline

```text
image -> preprocessing -> text detection -> text recognition -> post-processing -> text + layout
```

### 1. Preprocessing
Deskewing, denoising, contrast normalization, and binarization improve difficult scans.

### 2. Text Detection
Locate where text appears:
- **Segmentation-based**: DBNet and CRAFT predict per-pixel text probability maps, then extract polygons (handling curved text)
- **Box regression**: EAST and detector-style models predict rotated boxes

### 3. Text Recognition
Read the content of each cropped line or word:
- **CRNN + CTC**: a CNN extracts features, a recurrent network models the sequence, and CTC loss aligns predictions without character-level boxes
- **Attention / transformer decoders**: autoregressive models such as TrOCR use a vision encoder and text decoder

### 4. Post-processing
Language-model correction, dictionary checks, and format validation (dates, IDs) fix recognition errors.

## Usage Example

```python
import easyocr

reader = easyocr.Reader(["en"])
for box, text, conf in reader.readtext("receipt.jpg"):
    print(f"{conf:.2f}  {text}")
```

## Layout and Structure

Plain text is often not enough. Document understanding adds layout analysis, table recognition, and reading order, so headings, columns, and tables are preserved. Models like LayoutLM combine text, position, and image features for key-information extraction.

## Evaluation Metrics

- **Character error rate (CER)** and **word error rate (WER)**: edit distance relative to ground truth
- **Detection F1** at an IoU threshold
- **Field-level accuracy** for extraction tasks, which is what business workflows care about

## Modern Alternatives

Vision-language models can read text directly from an image and answer questions about it, without a separate detector and recognizer. They handle messy layouts and context well but are slower and costlier, can hallucinate characters, and lack precise coordinates. Many production systems combine both: a dedicated OCR engine for accuracy and coordinates, and a VLM for interpretation.

## Practical Tips

- Resolution matters: 300 DPI is a common target for scanned documents
- Fine-tune recognition on your domain's fonts and vocabulary when error rates are high
- Always keep confidence scores and route low-confidence fields to human review
