---
title: Introduction to Amazon Bedrock
description: Learn how Amazon Bedrock provides unified, serverless access to multiple foundation model providers within AWS's security and networking perimeter.
---

Amazon Bedrock is AWS's managed service for accessing foundation models from multiple providers (Anthropic, Meta, Mistral, Amazon's own Titan and Nova models, and others) through one API, without managing any inference infrastructure yourself.

## Why Bedrock Instead of Calling Providers Directly

Enterprises already operating inside AWS often need model access within their existing VPC, IAM permission model, and compliance boundary (data never leaving AWS's network, consistent audit logging through CloudTrail) rather than making outbound calls to each provider's separate API and managing separate credentials and network egress rules per provider.

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.invoke_model(
    modelId="anthropic.claude-sonnet-4-5-20250929-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Summarize this incident report."}]
    })
)
```

Model identifiers and request formats still vary somewhat by underlying provider, since Bedrock exposes each model's native API shape rather than fully normalizing every provider into one universal schema.

## Knowledge Bases and Agents

Bedrock includes managed Knowledge Bases, which handle chunking, embedding, and vector storage for RAG pipelines using integrated AWS services (typically OpenSearch or a supported vector store) without assembling that pipeline manually, and Bedrock Agents, which orchestrate multi-step tool use and API calls on top of a chosen foundation model, similar in concept to agent frameworks but managed within AWS's infrastructure.

## Provisioned Throughput vs. On-Demand

Bedrock offers on-demand, pay-per-token pricing for variable workloads, and provisioned throughput for reserving guaranteed capacity at a fixed cost, which matters for latency-sensitive or high-volume production workloads where on-demand rate limits could otherwise throttle traffic during peak usage.

## Practical Guidance

Choose Bedrock when AWS-native compliance, networking, and billing consolidation matter more than always having access to a specific provider's very latest model on day one, since new model availability on Bedrock sometimes lags a provider's own direct API by weeks. For teams not otherwise committed to AWS, calling providers directly usually gives more immediate access to new capabilities with less abstraction between you and the underlying API.
