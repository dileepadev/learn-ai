---
title: Introduction to Vanna
description: Learn how Vanna enables natural-language-to-SQL workflows for analytics and business intelligence applications.
---

Vanna is a framework for building text-to-SQL experiences, allowing users to ask questions in natural language and get SQL-generated answers from structured databases. It is useful for analytics assistants, internal BI copilots, and self-service data exploration tools.

## Why Text-to-SQL Matters

Business users often need answers from data but cannot write SQL fluently. Text-to-SQL systems can reduce this gap by translating intent into queries that run against existing data warehouses or databases.

## What Vanna Provides

### Natural Language Querying

Users can ask questions like:

- "What were monthly sales by region in Q2?"
- "Show churn trend for enterprise customers this year."

Vanna maps this intent to SQL queries.

### Schema-Aware Query Generation

To generate useful SQL, Vanna uses metadata and schema context. Good schema grounding is essential for reducing incorrect joins and ambiguous filters.

### Feedback-Driven Improvement

Teams can refine performance by incorporating corrected queries and examples, improving system behavior on recurring business questions.

## Typical Workflow

1. Connect Vanna to the target database
2. Provide schema information and usage examples
3. Let users submit natural-language questions
4. Review generated SQL and results
5. Add feedback loops for accuracy improvements

## Common Use Cases

- Internal analytics chatbots
- Product and growth dashboards
- Finance and operations Q&A
- Customer-support data assistants

## Key Risks and Controls

Text-to-SQL systems can generate expensive or incorrect queries if not governed carefully. Useful controls include:

- Read-only database roles
- Query timeout and row-limit policies
- Query approval for high-risk operations
- Guardrails for sensitive tables/columns

## Best Practices

- Start with high-value, low-risk analytics domains
- Maintain a clean data model and naming conventions
- Build evaluation sets from real analyst questions
- Track precision of generated SQL and execution success rate

## Benefits

- **Faster insights:** Less dependency on SQL specialists
- **Improved accessibility:** More teams can use data directly
- **Reusable analytics interface:** Natural language front end for multiple data sources

## When to Use Vanna

Vanna is most useful when your team has mature structured data and wants to broaden data access without forcing every user to become a SQL expert.

It works best when paired with strong governance, schema discipline, and iterative evaluation. With these in place, Vanna can become a practical bridge between business intent and database-level analytics.
