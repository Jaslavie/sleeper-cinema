# Sleeper Cinema

Inspired by the recent successes of low-budget/non-hollywood films "Backrooms" and "Obsession", we wanted to investigate Out-of-distribution box office hits and wether or not there were unseen patterns in their success. 

We detect "sleeper" movies that end up topping box-offices with a two-step approach:

1. Get the top-performing films by box office and **isolate anomalies** (Generative Semi-supervised Graph Anomaly Detection Approach)
2. Run a simple algorithm on top of those anomalies to uncover hidden trends in success

Definitionally, sleepers are not necessarily extra-ordinary films with extremely low budgets, rather they can exist as unremarkable/ordinary films on paper and still outperform its original expectations

## Dataset

We keep a focused set of columns from the two source datasets, combined into a single dataset:

**Dataset 1: `enhanced_box_office_data(2000-2024).csv`**

- `Rank`: box-office rank within the dataset.
- `Release Group`: movie title used as the film identifier.
- `$Worldwide`: total worldwide box-office gross.
- `$Domestic`: domestic box-office gross.
- `Domestic %`: domestic share of worldwide gross.
- `$Foreign`: international box-office gross.
- `Foreign %`: international share of worldwide gross.
- `Year`: release year.
- `Genres`: movie genres for profile similarity.
- `Rating`: audience rating score.
- `Production_Countries`: production country metadata.

**Dataset 2: `tmdb_5000_movies.csv`**

- `budget`: reported production budget.
- `production_companies`: production company metadata.

## Installation

Create a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable development with test dependencies from `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

To run training:

```
python src/train.py
```

## Architecture

Beforehand, we construct the film attribution graph. This graph is constructed from the dataset and then used to compare each film against similar films.

1. **Graph construction**: We connect movies whose genre profiles overlap by at least 60% Jaccard similarity. Each movie is a node, and its first-degree neighbors are its closest peer films.

2. **Success labeling**: We define a sleeper as a film whose budget-adjusted gross is in the top 10% relative to its graph neighbors. This makes success peer-relative instead of only box-office absolute.

3. **Unary scoring**: We train a gradient boosted tree on pre-release features only. This gives each film an individual sleeper score before considering its neighbors.

4. **Graph smoothing**: We run mean-field inference over the graph so similar films can influence each other's sleeper probability. Known training labels stay fixed, and validation films receive final ranked scores.

**Run:** `python -m src.model`. Rankings are written to `artifacts/crf_ranked.csv`.

## References

Related material, not necessarily implemented into this architecture.

- [Credal Graph Neural Networks](https://arxiv.org/abs/2512.02722)
- [Generative Semi-supervised Graph Anomaly Detection]()

