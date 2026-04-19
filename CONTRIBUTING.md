# Contributing

Thank you for contributing to this project.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Workflow

1. Create a feature branch from `main`.
2. Keep changes focused and easy to review.
3. Update documentation and tests when behavior changes.
4. Open a pull request using the repository template.

## Quality Checks

Run these commands before opening a pull request:

```bash
pytest -q
black --check .
isort --check-only .
flake8 .
```

## Commit Style

Use clear, descriptive commit messages such as:

```text
Improve training CLI and stabilize GitHub Actions workflow
```

## Reporting Issues

Use the GitHub issue templates for bugs and feature requests so maintainers can reproduce problems quickly.
