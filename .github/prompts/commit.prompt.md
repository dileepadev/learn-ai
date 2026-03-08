---
agent: "agent"
model: GPT-5 mini (copilot)
tools: ["execute", "read", "search"]
description: "Generate a new commit message based on the provided code changes."
---

Your goal is to generate the most appropriate and effective commit message based on the provided code changes.

## Commit Message Format

Follow this format strictly:
`<type>(<scope>): <short message> (<optional issue references>) (<optional PR reference>)`

`<optional longer description>`

### Type (`<type>`)

- `feat`: A new feature or enhancement.
- `fix`: A bug fix.
- `docs`: Documentation updates.
- `style`: Formatting, missing semi colons, etc; no logic change.
- `refactor`: Refactoring code without changing external behavior.
- `perf`: Performance improvements.
- `test`: Adding or modifying tests.
- `chore`: Routine tasks, maintenance, or tooling changes.

### Scope (`<scope>`)

- Optional but recommended (e.g., `repo`, `config`, `auth`, `api`, `ui`, `db`).

### Short Message (`<short message>`)

- Use imperative mood (e.g., "Add" not "Added").
- Capitalize first letter.
- No period at the end.
- Keep under 50 characters.

### References

- Issues: `(refs #2)` or `(fixes #2)`.
- PRs: `(#123)` at the end of the header.

## Instructions

- **Check the changes first**: Identify added, modified, or deleted files using `git diff --cached`.
- **Provide the final output as a Zsh-ready command**: You MUST wrap the `git commit` command in a Zsh code block so it can be copied and pasted directly into the terminal.
- **Example output format**:
  ```zsh
  git commit -m "feat(ui): Add new navigation menu (refs #2)" -m "Detailed description of changes..."
  ```
- **Current references**: None. Do not include any issue or PR references unless explicitly mentioned in the code changes.

## Examples

- `feat(ui): Add new navigation menu (refs #2)`
- `fix(api): Resolve memory leak in user service (#45)`
- `chore(repo): Update dependencies`

Ensure the commit message is professional, concise, and follows these guidelines exactly.
