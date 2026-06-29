---
title: "Secure Tool Use for AI Agents"
description: "How to let AI agents call tools without giving them unsafe levels of power."
---

Tool use makes AI agents dramatically more useful, but it also expands the attack surface. The moment an agent can send emails, query databases, or execute commands, prompt quality is no longer the only concern.

## Core Risks

- Prompt injection through retrieved or web content
- Over-privileged tools that allow dangerous actions
- Missing validation on tool arguments
- Lack of audit trails for what the agent actually did

## Safer Tool Design

Prefer narrow tools over general ones. A tool that creates a draft email is safer than a tool that sends arbitrary messages. Validation, permission checks, and human confirmation for irreversible actions are all important.

## Security Principle

Treat AI agents like untrusted operators with useful skills, not like perfectly aligned employees. That mindset leads to better permissions, better logging, and safer tool interfaces.
