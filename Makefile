.PHONY: install run clean lint

install:
		uv sync
run:
		uv run python3 -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/output/*.json

lint:
	uv run python3 -m flake8 --exclude=llm_sdk/llm_sdk/__init__.py
	uv run python3 -m mypy .

# export HF_HOME=~/goinfre/huggingface
# export UV_CACHE_DIR=~/goinfre/uv_cache
# export UV_PROJECT_ENVIRONMENT=~/goinfre/uv_venv
# export TMPDIR=~/goinfre/tmp
# export XDG_CACHE_HOME=~/goinfre/.cache