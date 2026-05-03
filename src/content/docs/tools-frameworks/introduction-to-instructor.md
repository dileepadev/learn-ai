---
title: Introduction to Instructor
description: Learn how to use the Instructor library to extract structured Pydantic models from LLM responses — covering client wrapping, automatic retry on validation errors, nested model extraction, list extraction, multi-provider support (OpenAI, Anthropic, Gemini, Ollama), streaming partial responses, and real-world use cases for structured RAG and entity extraction.
---

**Instructor** is a Python library that makes it straightforward to extract structured, validated data from LLM responses using Pydantic models. Instead of parsing freeform LLM output with string manipulation, you define a Pydantic model describing the data you want, pass it to the LLM, and Instructor handles the prompt injection, response parsing, and validation — retrying automatically if the model returns data that fails validation.

## Installation

```bash
pip install instructor
# Provider-specific: Instructor uses the official SDKs underneath
pip install openai anthropic google-generativeai
```

## Basic Usage: Wrapping the Client

```python
import instructor
from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel, Field

# ── Wrap OpenAI client ─────────────────────────────────────────────────────
client = instructor.from_openai(OpenAI())

# ── Wrap Anthropic client ──────────────────────────────────────────────────
anthropic_client = instructor.from_anthropic(Anthropic())

# ── Wrap Ollama (local models) ─────────────────────────────────────────────
from openai import OpenAI as OllamaClient
ollama_client = instructor.from_openai(
    OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON   # Ollama doesn't support function calling
)
```

## Defining Output Models with Pydantic

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ProductReview(BaseModel):
    """
    Structured extraction target for product review analysis.
    
    Field descriptions are injected into the prompt by Instructor,
    guiding the model to populate each field correctly.
    """
    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(
        ge=1, le=5,
        description="Rating from 1 (worst) to 5 (best)"
    )
    sentiment: Sentiment = Field(description="Overall sentiment of the review")
    pros: list[str] = Field(
        description="List of positive aspects mentioned, each as a short phrase"
    )
    cons: list[str] = Field(
        description="List of negative aspects mentioned, each as a short phrase"
    )
    summary: str = Field(
        max_length=200,
        description="One-sentence summary of the review"
    )
    would_recommend: bool = Field(description="Whether the reviewer would recommend the product")
    
    @field_validator("rating")
    @classmethod
    def rating_must_match_sentiment(cls, v, info):
        """Cross-field validation: high rating shouldn't be negative sentiment."""
        if "sentiment" in info.data:
            if v >= 4 and info.data["sentiment"] == Sentiment.NEGATIVE:
                raise ValueError("Rating 4-5 inconsistent with negative sentiment")
        return v


# ── Extract structured data from review text ──────────────────────────────
review_text = """
I bought this mechanical keyboard three months ago and it's been amazing.
The tactile feedback is superb and typing feels so satisfying. Build quality
is excellent — solid metal frame. The RGB lighting is beautiful. Only downside:
it's quite loud (hard to use in an office) and the software for RGB config
is a bit buggy. Overall a 4/5 for an enthusiast keyboard at home.
"""

review = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ProductReview,
    messages=[
        {"role": "user", "content": f"Extract structured data from this review:\n\n{review_text}"}
    ]
)

print(type(review))              # <class '__main__.ProductReview'>
print(review.product_name)       # "mechanical keyboard"
print(review.rating)             # 4
print(review.sentiment)          # Sentiment.POSITIVE
print(review.pros)               # ['tactile feedback', 'build quality', 'RGB lighting']
print(review.cons)               # ['loud', 'buggy software']
print(review.would_recommend)    # True
```

## Automatic Retry on Validation Failure

When the LLM returns data that fails Pydantic validation, Instructor automatically retries — injecting the validation error into the context so the model can correct its response:

```python
from instructor import patch
from pydantic import BaseModel, field_validator
import instructor

class EmailAddress(BaseModel):
    email: str
    domain: str
    is_corporate: bool
    
    @field_validator("email")
    @classmethod
    def must_be_valid_email(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"'{v}' is not a valid email address")
        return v.lower()
    
    @field_validator("domain")
    @classmethod
    def domain_must_match_email(cls, v: str, info) -> str:
        if "email" in info.data:
            expected = info.data["email"].split("@")[-1]
            if v != expected:
                raise ValueError(f"domain '{v}' doesn't match email domain '{expected}'")
        return v


# max_retries=3: if model returns invalid JSON or fails validation, retry 3 times
# On each retry, the validation error is appended to the conversation history
result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=EmailAddress,
    max_retries=3,
    messages=[
        {"role": "user", "content": "Extract the email from: Contact us at support@acmecorp.com"}
    ]
)

print(result.email)         # "support@acmecorp.com"
print(result.domain)        # "acmecorp.com"
print(result.is_corporate)  # True
```

## Nested Model Extraction

```python
from typing import Optional

class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None

class ContactInfo(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Address] = None

class Company(BaseModel):
    name: str
    industry: str
    founded_year: Optional[int] = Field(None, ge=1800, le=2024)
    headquarters: Optional[Address] = None
    key_contacts: list[ContactInfo] = Field(
        default_factory=list,
        description="List of key people at the company with their contact details"
    )
    annual_revenue_usd_millions: Optional[float] = None

# Extract complex nested structure from unstructured text
company_text = """
OpenAI was founded in 2015 and is headquartered at 3180 18th St, San Francisco, 
California 94110, US. Their CEO is Sam Altman (sam@openai.com) and CTO is Mira Murati.
The company operates in the artificial intelligence industry.
"""

company = client.chat.completions.create(
    model="gpt-4o",
    response_model=Company,
    messages=[
        {"role": "user", "content": f"Extract company information:\n\n{company_text}"}
    ]
)

print(company.name)                          # "OpenAI"
print(company.founded_year)                  # 2015
print(company.headquarters.city)             # "San Francisco"
print(company.key_contacts[0].name)          # "Sam Altman"
print(company.key_contacts[0].email)         # "sam@openai.com"
```

## List Extraction with `Iterable`

```python
from typing import Iterable

class Entity(BaseModel):
    text: str = Field(description="Exact text span from the document")
    entity_type: str = Field(description="Type: PERSON, ORG, LOCATION, DATE, or PRODUCT")
    confidence: float = Field(ge=0.0, le=1.0)

# Extract a variable-length list of entities
document = """
On March 15, 2024, Apple CEO Tim Cook announced a partnership with Google
to integrate Gemini into iPhone 17 devices. The deal was signed in Cupertino, California.
"""

# Iterable[Entity] streams entities one at a time as they are generated
entities = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Iterable[Entity],   # returns a generator
    messages=[
        {"role": "user", "content": f"Extract all named entities:\n\n{document}"}
    ]
)

for entity in entities:
    print(f"{entity.entity_type:10} | {entity.text:20} | {entity.confidence:.2f}")
# PERSON     | Tim Cook            | 0.99
# ORG        | Apple               | 0.99
# ORG        | Google              | 0.98
# PRODUCT    | Gemini              | 0.95
# PRODUCT    | iPhone 17           | 0.96
# LOCATION   | Cupertino           | 0.94
# LOCATION   | California          | 0.93
# DATE       | March 15, 2024      | 0.99
```

## Classification with Union Types

```python
from typing import Union, Literal, Annotated

class TechQuery(BaseModel):
    intent: Literal["code_help", "debugging", "explanation"]
    language: Optional[str] = Field(None, description="Programming language if applicable")
    complexity: Literal["beginner", "intermediate", "advanced"]

class GeneralQuery(BaseModel):
    intent: Literal["general_question", "recommendation", "comparison"]
    domain: str = Field(description="Subject domain of the query")

class SupportQuery(BaseModel):
    intent: Literal["bug_report", "feature_request", "billing"]
    urgency: Literal["low", "medium", "high", "critical"]
    product: str

# Union type: model chooses which schema best fits the input
QueryType = Annotated[
    Union[TechQuery, GeneralQuery, SupportQuery],
    Field(discriminator="intent")
]

def classify_query(user_message: str) -> Union[TechQuery, GeneralQuery, SupportQuery]:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=QueryType,
        messages=[
            {"role": "system", "content": "Classify the user's query into the appropriate category."},
            {"role": "user", "content": user_message}
        ]
    )

result = classify_query("My Python script crashes with a KeyError on line 42")
# Returns TechQuery(intent='debugging', language='Python', complexity='intermediate')
```

## Streaming Partial Responses

```python
from instructor.dsl.partial import Partial

# Stream partial model as it's generated — useful for progressive UI updates
class ReportSection(BaseModel):
    title: str
    content: str
    key_takeaways: list[str]

for partial_report in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=ReportSection,
    stream=True,
    messages=[
        {"role": "user", "content": "Write a brief section on transformer architecture."}
    ]
):
    # partial_report is populated progressively; access available fields
    if partial_report.title:
        print(f"\rTitle: {partial_report.title}", end="", flush=True)
```

## Multi-Provider Support Summary

| Provider | Wrapper | Best mode |
| --- | --- | --- |
| OpenAI | `instructor.from_openai(client)` | `TOOLS` (default) |
| Anthropic | `instructor.from_anthropic(client)` | `TOOLS` |
| Google Gemini | `instructor.from_gemini(client)` | `TOOLS` |
| Groq | `instructor.from_groq(client)` | `TOOLS` |
| Ollama | `instructor.from_openai(ollama_client)` | `JSON` |
| Mistral | `instructor.from_mistral(client)` | `TOOLS` |
| Cohere | `instructor.from_cohere(client)` | `TOOLS` |
| LiteLLM | `instructor.from_litellm(completion)` | `TOOLS` |

Instructor's core value is removing the friction between LLM outputs and typed application code. Every time you'd otherwise write a fragile regex or a chain of `json.loads` + key lookups, Instructor lets you replace that with a Pydantic model — giving you validation, documentation, IDE autocompletion, and automatic retry for free.
