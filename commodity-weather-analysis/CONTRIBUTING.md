# Contributing

Thank you for your interest in contributing to this project!

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or suggest features
- Clearly describe the problem and steps to reproduce
- Include Python version and OS

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for new functionality in `tests/`
4. Ensure all tests pass: `pytest tests/`
5. Format code with Black: `black src/`
6. Commit with a clear message: `git commit -m "Add: Granger causality rolling window"`
7. Push and open a Pull Request

### Code Style
- Follow PEP 8
- Use Black for formatting (`pip install black && black src/`)
- Add docstrings to all public functions
- Keep functions focused and < 50 lines where possible

### Areas for Contribution
- Additional commodity pairs (e.g., Sugar/Ethanol, Natural Gas/Coal)
- Extended weather variables (humidity, extreme event indices)
- Additional ML architectures (Transformer, N-BEATS)
- Dashboard enhancements
- Additional cointegration tests

## Questions?
Open a GitHub Discussion or contact the maintainer via the repo.
