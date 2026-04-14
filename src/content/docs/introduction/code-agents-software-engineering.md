---
title: Code Agents and AI for Software Engineering
description: Explore how AI agents tackle full software engineering tasks — from repository-level bug fixing and test generation to autonomous code review. Learn how SWE-bench evaluates these systems and how architectures like SWE-agent, OpenHands, and GitHub Copilot Workspace work.
---

**Code agents** are AI systems that autonomously plan, write, execute, and iterate on code to complete software engineering tasks — going far beyond autocomplete to handle full bug fixes, feature implementations, test generation, and repository navigation. This represents a shift from AI as a **typing assistant** to AI as a **collaborative software engineer**.

## Beyond Code Completion

Traditional AI coding tools (GitHub Copilot, Tabnine) operate at the **token or line level**: they predict the next token given the current context. Code agents operate at the **task level**: given a natural language specification, they interact with a repository over multiple steps to deliver a working solution.

| Capability | Code Completion | Code Agent |
|---|---|---|
| Scope | Single function or snippet | Full task across files |
| Tool use | No | File read/write, bash, tests |
| Iteration | No | Runs tests, fixes errors |
| Planning | Implicit | Explicit multi-step planning |
| Memory | Context window | Repository + scratchpad |

## SWE-bench: The Standard Benchmark

**SWE-bench** (Jimenez et al., 2023) is the canonical benchmark for evaluating repository-level software engineering agents. It consists of **2,294 real GitHub issues** from 12 popular Python repositories (Django, Flask, scikit-learn, matplotlib, numpy, etc.), each with:
- A natural language issue description
- A failing test that will pass after the fix
- A ground-truth patch

Agents are scored by **% resolved**: the fraction of issues where the submitted patch passes the tests. SWE-bench Verified (a human-validated subset of 500 issues) is the standard leaderboard.

### 2024–2025 SOTA Performance

| System | SWE-bench Verified (%) |
|---|---|
| SWE-agent (Claude 3.5 Sonnet) | ~48% |
| OpenHands (Claude 3.5) | ~53% |
| Devin 2.0 | ~53% |
| Amazon Q Developer | ~50% |
| Human developers (expert) | ~94% |

Performance has grown from ~4% (GPT-4 baseline, 2023) to over 50% in 18 months.

## Architectural Components

Code agents share a common architecture:

### Environment Interface
The agent interacts with a **sandboxed software environment** containing:
- The repository's file system
- A bash shell for executing commands
- Python interpreter for running tests and scripts
- Git for tracking changes

### Tools
Agents use tools to interact with the environment:
- `read_file(path)` — Read file contents
- `write_file(path, content)` — Create or overwrite a file
- `bash(command)` — Execute shell commands
- `search(pattern)` — Search for code patterns across the repository
- `python_repl(code)` — Execute Python snippets interactively

### Planning and Reasoning
Agents typically follow a **think → act → observe** loop:

```
1. Think: Analyze the issue, identify relevant files
2. Act: Read files, search codebase, write fixes
3. Observe: Run tests, check output
4. Repeat until tests pass or budget exhausted
```

## SWE-agent

SWE-agent (Yang et al., 2024, from Princeton NLP) introduced a specialized **Agent-Computer Interface (ACI)** — a set of tools purposefully designed for software engineering tasks, distinct from general tool use:

- **File viewer:** Shows file contents with line numbers, scrolling, and search
- **Code editor:** Applies precise line-range edits without rewriting entire files
- **Test runner:** Runs the failing test and returns pass/fail and output

The ACI design insight: generic bash is too noisy for LLMs; structured, purpose-built tools with tight action/observation formats outperform raw shell access.

## OpenHands (formerly OpenDevin)

**OpenHands** is an open-source software engineering agent platform that provides:
- A sandboxed Docker environment for each task
- A browser-based UI for interacting with agents in real time
- Multiple agent implementations (CodeAct, browsing-capable agents)
- A plugin system for adding new tools

**CodeAct** — OpenHands' primary agent architecture — represents all actions as **executable code** rather than structured tool calls. The agent writes Python or bash code that gets executed in the sandbox, and the output is appended to the context. This unifies tools and actions into a single interface.

## Devin

**Devin** (Cognition Labs, 2024) was the first publicly announced "AI software engineer" to demonstrate end-to-end autonomous software task completion. Key features:
- Long-horizon task memory across multi-hour work sessions
- Integration with IDEs, terminals, browsers, and deployment tools
- Learns and iterates from test failures and error messages
- Capable of reading documentation and installing dependencies

Devin sparked industry-wide investment in autonomous software engineering agents.

## GitHub Copilot Workspace

GitHub Copilot Workspace (2024) integrates deep repository context into a task-completion workflow within GitHub:
- Starts from a GitHub issue
- Plans which files must be changed and why
- Implements the changes in a side-by-side editor
- Creates a PR directly from the workspace

Unlike SWE-agent or OpenHands (which run in isolated sandboxes), Copilot Workspace is tightly integrated into the GitHub UI/UX, making it more accessible to non-researchers.

## Challenges and Open Problems

### Repository Navigation
Real repositories contain hundreds of files; the agent must efficiently identify which 3-5 files are relevant to the issue without reading everything.

### Test Reliability
Some codebases have flaky or environment-dependent tests; an agent may incorrectly conclude its fix is wrong when the test failure is unrelated.

### Long-Context Handling
Bug fixes may require understanding deeply nested call chains across many files, straining context windows.

### Security
Code agents with shell access can execute arbitrary commands in their environment. Careful sandboxing and resource limits are critical in any deployment.

### Evaluation Validity
SWE-bench measures pass/fail on a specific test oracle; a submitted patch may pass the test but introduce new bugs, not generalize to similar issues, or be unmaintainable.

## Code Review and Automated PR Analysis

Beyond issue resolution, AI is being applied to:
- **PR review:** Automatically identify bugs, security issues, and style violations in pull requests
- **Test generation:** Generate unit tests for existing code using coverage analysis
- **Documentation generation:** Write docstrings and README sections from code analysis
- **Dependency vulnerability detection:** Flag outdated or vulnerable dependencies

Tools like GitHub Copilot code review, CodeRabbit, and Amazon CodeGuru implement these capabilities at scale.

## Further Reading

- Jimenez et al. (2023), *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- Yang et al. (2024), *SWE-agent: Agent Computer Interfaces Enable Software Engineering Language Models*
- Wang et al. (2024), *OpenDevin: An Open Platform for AI Software Developers as Generalist Agents*
- Github Copilot Workspace: https://githubnext.com/projects/copilot-workspace
- SWE-bench Leaderboard: https://www.swebench.com
