---
title: Introduction to Streamlit - Building ML Demos and Data Apps
description: Learn how Streamlit turns Python scripts into interactive web apps for model demos, with notes on its rerun model, caching, and chat interface components.
---

Streamlit is a Python framework for building interactive web apps with no front-end code. It is widely used to demo models, explore datasets, and wrap LLM prototypes in a usable interface.

## The Execution Model

A Streamlit app is a script that reruns from top to bottom whenever the user interacts with a widget. This makes the code simple but requires care with expensive operations.

```python
import streamlit as st

st.title("Sentiment Demo")
text = st.text_area("Enter some text")

if st.button("Analyze") and text:
    result = classifier(text)[0]
    st.metric("Label", result["label"], f"{result['score']:.2%}")
```

## Caching

Because the script reruns, load models and data once using caching decorators:

```python
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

@st.cache_data
def load_data(path):
    return pd.read_parquet(path)
```

`cache_resource` is for unserializable singletons such as models and connections; `cache_data` is for serializable values such as DataFrames.

## Session State

`st.session_state` persists values across reruns, which is essential for chat history and multi-step flows.

## Chat Interfaces

Built-in chat components make LLM front ends short to write:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if prompt := st.chat_input("Ask something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
```

## Deployment Notes

Streamlit suits internal tools and prototypes. For high-traffic production products with custom UX requirements, teams typically move to a dedicated front end backed by an API.
