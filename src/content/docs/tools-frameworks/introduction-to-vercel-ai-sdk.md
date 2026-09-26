---
title: Introduction to the Vercel AI SDK
description: Learn how the Vercel AI SDK standardizes streaming, tool calling, and UI state management across LLM providers for JavaScript and React apps.
---

The Vercel AI SDK is a TypeScript library for building AI-powered applications, providing a consistent interface for text generation, streaming, structured output, and tool calling across many different LLM providers.

## Provider-Agnostic Generation

```typescript
import { generateText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

const { text } = await generateText({
  model: anthropic('claude-sonnet-4-5'),
  prompt: 'Write a haiku about distributed systems.',
});
```

Switching providers is a matter of importing a different provider package and passing a different model instance, while the surrounding `generateText`, `streamText`, and `generateObject` function calls stay the same — this insulates application code from provider-specific SDK differences.

## Streaming and the useChat Hook

The SDK's React hooks (`useChat`, `useCompletion`) handle the client-side complexity of streaming responses token-by-token into a chat UI, managing message state, loading states, and error handling, so building a responsive streaming chat interface doesn't require hand-rolling a server-sent-events or WebSocket client from scratch:

```typescript
'use client';
import { useChat } from 'ai/react';

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat();
  // renders messages and a form wired to the streaming chat endpoint
}
```

## Structured Output and Tool Calling

`generateObject` and `streamObject` constrain model output to match a provided schema (typically defined with Zod), returning validated, typed objects instead of raw text that needs separate parsing, and the SDK's tool-calling interface lets you define callable functions with typed parameters that work consistently across providers that support function calling.

## Practical Guidance

Use the Vercel AI SDK when building a web application (particularly with Next.js or React) that needs to remain flexible about which LLM provider it uses, since the abstraction cost is low and the provider-switching flexibility is high. For non-JavaScript backends or applications that need very provider-specific features not yet abstracted by the SDK, calling a provider's native SDK directly may still be necessary for that specific capability.
