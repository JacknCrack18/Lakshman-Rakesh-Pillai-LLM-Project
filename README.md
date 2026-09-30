# Lakshman-Rakesh-Pillai-LLM-Project
# Lakshman LLM Project

A small movie recommendation demo that searches a built-in collection of 100 movies using semantic similarity.

## What it does

- Stores movie titles, directors, years, genres, moods, and descriptions in `movies2.py`.
- Encodes the movie information with the `all-MiniLM-L6-v2` Sentence Transformers model.
- Ranks movies for a user's natural-language query using cosine similarity.
- Includes example query results in `demo_results.txt`.

## Requirements

Python dependencies are listed in `requirements.txt`:

- NumPy
- Sentence Transformers
- scikit-learn

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first run may download the sentence-transformer model.

## Run

```powershell
python movies2.py
```

Enter a movie mood, theme, or description when prompted. The program returns the closest matches from the included dataset.

## Files

- `movies2.py` — movie dataset and semantic search demo.
- `demo_results.txt` — saved sample results and relevance notes.
- `requirements.txt` — Python package dependencies.
