"""
setup.py for Smart Shopping Negotiator
--------------------------------------
Allows the project to be installed as a local package with:
    pip install -e .          (development / editable install)
    pip install .             (regular install)

Running `pip install -e .` is the recommended way to work on this project
locally — it adds `src/` to Python's module search path, so every import
like `from src.shopping_agent.graph.state import ShoppingState` resolves
correctly from any working directory, without manually setting PYTHONPATH.
"""

from setuptools import setup, find_packages

setup(
    # -----------------------------------------------------------------
    # Project identity
    # -----------------------------------------------------------------
    name="smart-shopping-agent",
    version="1.0.0",
    author="Nishit Kumar",
    author_email="nishitkumaroll12@gmail.com",
    description=(
        "An agentic AI-powered shopping assistant built with LangGraph, "
        "Groq (Llama 3.3 70B), and SerpAPI. Classifies intent, generates "
        "dynamic clarifying questions, fetches live product data, validates "
        "deals, detects fake discounts, and recommends the best option via "
        "a Streamlit chat interface."
    ),
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="NA",

    # -----------------------------------------------------------------
    # Package discovery
    # -----------------------------------------------------------------
    # find_packages(where="src") tells setuptools to look inside src/
    # for any folder that contains an __init__.py, and treat it as a
    # Python package. This picks up:
    #   src/shopping_agent/
    #   src/shopping_agent/graph/
    #   src/shopping_agent/nodes/
    #   src/shopping_agent/services/
    #   src/shopping_agent/utils/
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # -----------------------------------------------------------------
    # Python version requirement
    # -----------------------------------------------------------------
    python_requires=">=3.10",

    # -----------------------------------------------------------------
    # Runtime dependencies
    # (mirrors requirements.txt — keep both in sync)
    # -----------------------------------------------------------------
    install_requires=[
        # Orchestration & LLM
        "langgraph>=0.0.20",
        "langchain-groq>=0.1.0",
        "langchain-core>=0.1.50",

        # External APIs
        "google-search-results>=2.4.2",   # Official SerpAPI client

        # UI
        "streamlit>=1.32.0",

        # Environment / secrets management
        "python-dotenv>=1.0.1",

        # Data validation (used for structured LLM output in nodes)
        "pydantic>=2.0.0",
    ],

    # -----------------------------------------------------------------
    # Optional / development dependencies
    # Install with: pip install -e ".[dev]"
    # -----------------------------------------------------------------
    extras_require={
        "dev": [
            "pytest>=8.0.0",          # Unit testing
            "vulture>=2.0",           # Dead code detection
            "pyflakes>=3.0",          # Unused imports / variables
            "ruff>=0.4.0",            # Fast linter (replaces flake8)
        ],
    },

    # -----------------------------------------------------------------
    # CLI entry point
    # Lets you run `shopping-agent` from the terminal instead of
    # `streamlit run app.py` — optional, useful for demos.
    # -----------------------------------------------------------------
    entry_points={
        "console_scripts": [
            "shopping-agent=shopping_agent.__main__:main",
        ],
    },

    # -----------------------------------------------------------------
    # Metadata for PyPI (if ever published)
    # -----------------------------------------------------------------
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "agentic-ai", "langgraph", "groq", "llm", "shopping-assistant",
        "multi-agent", "streamlit", "serpapi", "langchain"
    ],
)