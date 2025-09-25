# Contributing

Please follow these guidelines to contribute to the Activation Probing Toolkit.

- Code style: follow PEP8. Use `black` for formatting and `ruff`/`flake8` for linting.
- Tests: All new functionality must include tests. Run tests with `pytest`.
- Type checks: Maintain type annotations; run `mypy` in PRs when feasible.
- Branches: `feature/<short-desc>`, `fix/<short-desc>`, `ci/<short-desc>`.
- PRs: Small and focused. Include a description, the test plan, and impact.

CI should run: lint, mypy, pytest.
