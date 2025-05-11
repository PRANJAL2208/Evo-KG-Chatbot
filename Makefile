SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

install:
	git submodule update --init --recursive
	poetry lock --no-update
	poetry install

streamlit-dev:
	poetry run streamlit run streamlit_app.py --server.port 8501
