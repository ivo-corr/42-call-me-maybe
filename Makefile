INTRA		= icorrale
VENV_NAME	= call_me_maybe-venv
SDK_PATH	= ./llm_sdk
MAIN_PATH	= main.py

export UV_CACHE_DIR=/goinfre/$(INTRA)/uv-cache
export UV_PROJECT_ENVIRONMENT=/goinfre/$(INTRA)/$(VENV_NAME)
export HF_HOME=/goinfre/$(INTRA)/huggingface-cache

install:
	@mkdir -p /goinfre/$(INTRA)/uv-cache
	@mkdir -p /goinfre/$(INTRA)/$(VENV_NAME)	
	@mkdir -p /goinfre/$(INTRA)/huggingface-cache
	@uv sync --project $(SDK_PATH)

run: install
	@uv run --project $(SDK_PATH) python3 $(MAIN_PATH)