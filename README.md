# Movie Recommender System Analysis

This project implements and analyzes different recommender system algorithms using the MovieLens dataset.

## Setup

1. Download the MovieLens dataset from [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset)
2. Place the `ratings_small.csv` file in the project directory
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Features

- Implements and compares three recommendation algorithms:
  - Probabilistic Matrix Factorization (PMF)
  - User-based Collaborative Filtering
  - Item-based Collaborative Filtering
- Evaluates performance using MAE and RMSE metrics
- Analyzes impact of different similarity metrics (cosine, MSD, Pearson)
- Studies effect of neighborhood size on performance
- Includes visualization of results

## Usage

Run the analysis:
```bash
python recommender_analysis.py
```

## Output

The script will generate:
- Performance metrics for each algorithm
- Plots comparing similarity metrics
- Plots showing impact of neighborhood size
- Best neighborhood size for both user-based and item-based CF
