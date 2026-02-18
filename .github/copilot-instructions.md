# Copilot Instructions

## Guidelines

- Follow the project's code style.
- Update documentation if necessary.
- Refer to the following templates and guidelines before submitting your changes:
  - [learn-ai/](../) - Root directory of the repository
    - [.github/](./) - GitHub-specific files (workflows, templates, etc.)
      - [ISSUE_TEMPLATE/](./ISSUE_TEMPLATE) - Contains all issue templates
        - [bug_report.md](./ISSUE_TEMPLATE/bug_report.md) - Template for reporting bugs
        - [documentation_update.md](./ISSUE_TEMPLATE/documentation_update.md) - Template for documentation updates
        - [feature_request.md](./ISSUE_TEMPLATE/feature_request.md) - Template for suggesting new features
        - [feedback.md](./ISSUE_TEMPLATE/feedback.md) - Template for general feedback
        - [other.md](./ISSUE_TEMPLATE/other.md) - Template for other types of issues
      - [PULL_REQUEST_TEMPLATE.md](./PULL_REQUEST_TEMPLATE.md) - Template for pull request submissions
    - [BRANCH_NAMING_GUIDELINES.md](../BRANCH_NAMING_GUIDELINES.md) - Branch naming rules
    - [CHANGELOG.md](../CHANGELOG.md) - Record of project changes
    - [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - Contributor behavior guidelines
    - [COMMIT_MESSAGE_GUIDELINES.md](../COMMIT_MESSAGE_GUIDELINES.md) - Rules for writing commit messages
    - [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute to the project
    - [LICENSE](../LICENSE) - Project license
    - [PULL_REQUEST_GUIDELINES.md](../PULL_REQUEST_GUIDELINES.md) - Pull request submission guidelines
    - [README.md](../README.md) - Project overview
    - [SECURITY.md](../SECURITY.md) - Security policy and reporting
    - [TODO.md](../TODO.md) - Tasks planned for future releases
    - [VERSIONING.md](../VERSIONING.md) - Versioning strategy for the project

## Commit Message Guidelines

Commit messages should be written clearly and consistently, following the project’s template.

- **Follow the template exactly** — do not modify or “improve” it.
- **Provide the commit message as a Zsh-ready `git commit -m "..."` command**, so it can be run directly in the terminal.
- **Check the changes first**: identify added, modified, or deleted files.
- **Describe the changes according to the template**:
  - For **multiple files**, summarize changes at a high level.
  - For **a single file**, describe the exact modification.
- Include **code snippets or terminal commands in Zsh format** where indicated by the template.
- **Consistency is key**: ensure every commit follows the template to keep project history clear and uniform.
