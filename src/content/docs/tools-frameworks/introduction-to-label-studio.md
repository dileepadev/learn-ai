---
title: Introduction to Label Studio
description: Get started with Label Studio — the open-source data annotation platform for creating high-quality ML training datasets — covering labeling interfaces, label config XML, ML backend integration for pre-annotation, active learning, quality control, and export formats for vision, NLP, and audio tasks.
---

High-quality labeled data is the foundation of every supervised ML model, yet annotation is slow, expensive, and error-prone. **Label Studio** is an open-source, self-hosted data annotation platform that supports labeling across modalities — text, images, audio, video, and time series — with a flexible XML-based configuration system, an ML backend API for pre-annotation and active learning, and integrations with popular dataset formats.

## Installation

```bash
# Install via pip
pip install label-studio

# Start the server
label-studio start

# Or run with Docker
docker run -it -p 8080:8080 \
  -v $(pwd)/label-studio-data:/label-studio/data \
  heartexlabs/label-studio:latest label-studio
```

Label Studio runs at `http://localhost:8080`. Create an account and log in to access the project dashboard.

## Core Concepts

**Projects** organize related labeling tasks. Each project has:

- A **labeling configuration** (XML) defining the annotation interface
- A set of **tasks** (items to annotate, e.g., images or text snippets)
- One or more **annotators** assigned to label tasks
- An optional **ML backend** for pre-annotation

**Annotations** are the completed labels on a task. **Predictions** are model-generated pre-labels.

## Labeling Interfaces and Label Config

The label config is an XML document specifying what data to show and what annotation tools to expose. Label Studio ships with pre-built templates for common tasks.

### Text Classification

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="label" toName="text" choice="single-radio">
    <Choice value="Positive"/>
    <Choice value="Negative"/>
    <Choice value="Neutral"/>
  </Choices>
</View>
```

### Named Entity Recognition

```xml
<View>
  <Labels name="label" toName="text">
    <Label value="Person" background="#FFA39E"/>
    <Label value="Organization" background="#D4B896"/>
    <Label value="Location" background="#96C4D4"/>
    <Label value="Date" background="#B7EB8F"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```

### Image Bounding Box Detection

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Car" background="#1890FF"/>
    <Label value="Person" background="#13C2C2"/>
    <Label value="Traffic Light" background="#52C41A"/>
  </RectangleLabels>
</View>
```

### Image Segmentation (Polygon)

```xml
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image" strokeWidth="3">
    <Label value="Building" background="#FF7A45"/>
    <Label value="Road" background="#9254DE"/>
    <Label value="Vegetation" background="#52C41A"/>
  </PolygonLabels>
</View>
```

### Audio Transcription

```xml
<View>
  <Audio name="audio" value="$audio"/>
  <TextArea name="transcription" toName="audio"
    placeholder="Type transcription here..."
    rows="4" editable="true" maxSubmissions="1"/>
</View>
```

## Importing Tasks

Import tasks via the UI, CLI, or SDK. Tasks are JSON documents where keys match the `$variable` placeholders in the label config:

```python
import label_studio_sdk

# Connect to running Label Studio instance
ls = label_studio_sdk.Client(url="http://localhost:8080", api_key="your-api-key")

project = ls.get_project(project_id=1)

# Import text classification tasks
tasks = [
    {"data": {"text": "The product is excellent and shipping was fast."}},
    {"data": {"text": "Terrible customer service, never buying again."}},
    {"data": {"text": "Average experience, nothing special."}},
]
project.import_tasks(tasks)

# Import image tasks
image_tasks = [
    {"data": {"image": "https://example.com/image1.jpg"}},
    {"data": {"image": "s3://my-bucket/images/image2.jpg"}},
]
project.import_tasks(image_tasks)
```

## ML Backend for Pre-Annotation

Label Studio's ML backend API lets you connect a model to generate pre-annotations that annotators review and correct — dramatically reducing labeling time:

```python
# ml_backend.py — A simple ML backend server
from label_studio_ml import LabelStudioMLBase
from label_studio_ml.model import ModelResponse
from transformers import pipeline


class SentimentClassifier(LabelStudioMLBase):
    def setup(self):
        self.classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

    def predict(self, tasks, **kwargs):
        from_name = "label"
        to_name = "text"
        task_type = "choices"
        predictions = []

        for task in tasks:
            text = task["data"]["text"]
            result = self.classifier(text)[0]

            # Map model label to Label Studio label
            label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
            predicted_label = label_map.get(result["label"], result["label"])

            predictions.append({
                "result": [{
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": task_type,
                    "value": {"choices": [predicted_label]},
                }],
                "score": result["score"],
                "model_version": "roberta-sentiment-v1",
            })

        return ModelResponse(predictions=predictions)
```

```bash
# Start the ML backend server
label-studio-ml start ml_backend.py --port 9090

# Connect it to Label Studio project via Settings > Model
```

## Active Learning Loop

Active learning uses model uncertainty to select the most informative unlabeled samples for annotation:

```python
from label_studio_sdk import Client
import numpy as np

ls = Client(url="http://localhost:8080", api_key="your-api-key")
project = ls.get_project(project_id=1)

# 1. Get predictions with uncertainty scores from ML backend
tasks_with_preds = project.get_tasks(filters={"completed_at__isnull": True})

# 2. Sort by uncertainty (lowest confidence = highest uncertainty)
def get_uncertainty(task):
    if not task.get("predictions"):
        return 1.0  # Maximum uncertainty if no prediction
    return 1.0 - max(p["score"] for p in task["predictions"])

tasks_sorted = sorted(tasks_with_preds, key=get_uncertainty, reverse=True)

# 3. Prioritize top-N uncertain tasks for human review
priority_task_ids = [t["id"] for t in tasks_sorted[:100]]
print(f"Prioritized {len(priority_task_ids)} uncertain tasks for labeling")
```

## Project and Annotation Management

```python
# Create a new project
project = ls.create_project(
    title="Medical Image Segmentation",
    label_config="""
    <View>
      <Image name="image" value="$image"/>
      <PolygonLabels name="label" toName="image">
        <Label value="Tumor" background="#FF4D4F"/>
        <Label value="Healthy Tissue" background="#52C41A"/>
      </PolygonLabels>
    </View>
    """,
)

# List all annotations in a project
annotations = project.get_labeled_tasks()
print(f"Completed tasks: {len(annotations)}")

# Filter unlabeled tasks
unlabeled = project.get_tasks(filters={"completed_at__isnull": True})
print(f"Remaining tasks: {len(unlabeled)}")
```

## Quality Control and Review

Label Studio supports multi-annotator workflows and consensus labeling:

- Assign multiple annotators to the same task to measure inter-annotator agreement
- Enable **Review mode** to have a senior annotator accept or reject each annotation
- Use **Agreement Score** in project settings to filter low-consensus annotations
- Set a **minimum annotations per task** threshold before a task is considered complete

Configure in project settings:

```text
Annotations per task: 2
Show agreement: Yes
Agreement threshold: 0.8
```

## Exporting Annotations

Label Studio supports a range of export formats:

```python
# Export in JSON format (full annotation structure)
project.export_tasks(export_type="JSON", path="./annotations.json")

# Export in JSON-MIN format (simplified for downstream training)
project.export_tasks(export_type="JSON_MIN", path="./annotations_min.json")

# Export in COCO format (for object detection)
project.export_tasks(export_type="COCO", path="./coco_dataset.json")

# Export in CoNLL format (for NER)
project.export_tasks(export_type="CONLL2003", path="./ner_annotations.conll")

# Export in YOLO format (for object detection training)
project.export_tasks(export_type="YOLO", path="./yolo_dataset/")
```

Available export formats:

- **JSON / JSON-MIN**: universal format for all task types
- **COCO**: bounding boxes and segmentation for computer vision
- **YOLO**: object detection with `.txt` label files and `data.yaml`
- **CoNLL 2003**: NER token-level annotations
- **spaCy**: `.spacy` binary format for NER training
- **CSV**: tabular export for classification and regression tasks
- **Pascal VOC XML**: classic object detection format

## Label Studio vs Label Studio Cloud

| Feature | Label Studio OSS | Label Studio Cloud |
| --- | --- | --- |
| Hosting | Self-hosted | Managed SaaS |
| Storage | Local / S3 / GCS | Cloud-native |
| Team collaboration | Manual user management | Roles and organizations |
| SSO | Not included | SAML/OIDC |
| Support | Community | Enterprise SLA |
| Cost | Free | Paid per seat |

## Summary

Label Studio is a flexible, self-hostable annotation platform for building high-quality ML training datasets:

- **XML-based label configs** define annotation interfaces for text, images, audio, video, and time series without writing frontend code
- **The ML backend API** connects any model to pre-annotate tasks, reducing annotator time by surfacing model predictions for review rather than labeling from scratch
- **Active learning integration** prioritizes the most uncertain samples, focusing expensive human labeling where it has the most impact on model improvement
- **Review mode and agreement scoring** enable quality control workflows that catch annotation errors before they corrupt training data
- **Rich export formats** (COCO, YOLO, CoNLL, spaCy, CSV) cover the common input formats of popular training frameworks and downstream tools
- **The Python SDK** provides programmatic access to all project, task, and annotation operations, enabling fully automated dataset curation pipelines
