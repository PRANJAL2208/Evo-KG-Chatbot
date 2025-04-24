# 💬 EvoKG Chatbot

An AI chatbot designed to answer queries about the EvoKG knowledge graph using OpenAI's GPT models and the Kani framework, served via Streamlit.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evokg.ahujalab.iiitd.edu.in/)

## Features

*   Query the EvoKG knowledge graph about entities like Gene, Protein, Disease, etc.
*   Predict relationships between biological entities.
*   Utilizes the `evo-utils` submodule for Streamlit integration with Kani agents.
*   Powered by OpenAI's GPT models (configurable).

## Setup Instructions

### Prerequisites

*   [Git](https://git-scm.com/downloads)
*   [Python](https://www.python.org/downloads/) (>=3.11, <4.0 recommended, check `pyproject.toml`)
*   [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)

### Installation

1.  **Clone the repository with submodules:**

    ```bash
    git clone --recurse-submodules https://github.com/zakmii/Evo-KG-Chatbot.git
    cd Evo-KG-Chatbot
    ```

    *If you cloned without `--recurse-submodules`, run this inside the project directory:*
    ```bash
    git submodule update --init --recursive
    ```

2.  **Install Poetry (if you haven't already):**

    Follow the official [Poetry installation guide](https://python-poetry.org/docs/#installation).

3.  **Install project dependencies using Poetry:**

    This command reads the `pyproject.toml` file, resolves dependencies (including the local `evo-utils` submodule), and installs them into a virtual environment managed by Poetry.

    ```bash
    poetry install
    ```

4.  **Set up environment variables:**

    Create a `.env` file in the root directory (`Evo-KG-Chatbot/`) and add your OpenAI API key:

    ```env
    # .env
    OPENAI_API_KEY="sk-..."
    ```

    *Optional: For chat sharing functionality via Upstash Redis:*
    ```env
    # UPSTASH_REDIS_REST_URL="https://..."
    # UPSTASH_REDIS_REST_TOKEN="..."
    ```

## How to Run

1.  **Activate the Poetry virtual environment:**

    ```bash
    poetry shell
    ```

2.  **Run the Streamlit app:**

    ```bash
    streamlit run streamlit_app.py
    ```

    *Alternatively, you can run it directly using `poetry run`:*
    ```bash
    poetry run streamlit run streamlit_app.py
    ```

The application should now be running and accessible in your web browser (usually at `http://localhost:8501`).
