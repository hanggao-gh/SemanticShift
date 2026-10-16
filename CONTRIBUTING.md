# Contributing

1. Create a feature branch.
2. Keep changes focused and add minimal documentation.
3. Run formatting/lint:
   - `pip install -r requirements-dev.txt`
   - `ruff check .`
   - `black .`

If you add new experiments, please:
- document how to run them in `README.md`
- save outputs under `results/` with descriptive filenames
- record key hyperparameters (model/dataset/window size/etc.)
