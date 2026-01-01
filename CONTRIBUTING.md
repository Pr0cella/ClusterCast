# Contributing

Thanks for your interest in improving this project! We welcome issues, discussions, and pull requests.

## Ways to contribute
- Report bugs with clear reproduction steps and environment details
- Propose features or enhancements via issues or discussions
- Improve docs: READMEs, examples, comments, and diagrams
- Add models, calibrators, optimizers, evaluators, or reporters via the plugin-friendly framework
- Create example plugins under `src/plugins/models/`

## Getting started
1. Fork the repo and create a feature branch
2. Make your changes with clear, focused commits
3. If you add public behavior, include or update small tests/examples
4. Run linters/formatters if present; keep style consistent
5. Open a PR describing motivation, approach, and validation

## Code guidelines
- Keep changes minimal and focused; avoid unrelated refactors
- Prefer small, well-named functions and clear docstrings
- Add types where practical; keep APIs stable
- Include acceptance criteria or before/after snippets when changing outputs

## Plugin development
- Put plugin models in `src/plugins/models/` and subclass `ModelInterface`
- Use `framework.modular_runner --list-models` to verify discovery
- Add a short README section or docstring explaining your model’s idea and parameters

## Discussions and support
- Use issues for bugs and concrete feature requests
- Use discussions (if enabled) for design ideas and Q&A

We appreciate your contributions and time. Thank you for helping make the framework better!