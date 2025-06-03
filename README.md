# Movie Recommender System Analysis

A comprehensive analysis of different recommendation algorithms using the MovieLens dataset. This project explores and compares multiple approaches to movie recommendation systems, providing insights into their performance and characteristics.

## Project Overview

This project aims to:
- Implement and compare various recommendation algorithms
- Analyze their performance metrics
- Study the impact of different parameters on recommendation quality
- Provide visual insights into the recommendation process

## Dataset

The project uses the MovieLens dataset, which contains movie ratings from real users. The dataset includes:
- User ratings for movies
- Movie metadata
- User demographics
- Timestamps of ratings

## Setup Instructions

1. **Dataset Preparation**
   - Download the MovieLens dataset from [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset)
   - Place the `ratings_small.csv` file in the project's root directory

2. **Environment Setup**
   - Create a virtual environment (recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Unix/macOS
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```

## Implemented Algorithms

The project implements and compares three main recommendation algorithms:

1. **Probabilistic Matrix Factorization (PMF)**
   - A latent factor model that decomposes the user-item rating matrix
   - Uses probabilistic approach to handle missing data
   - Provides probabilistic predictions of user preferences

2. **User-based Collaborative Filtering**
   - Finds similar users based on their rating patterns
   - Recommends items that similar users have liked
   - Supports multiple similarity metrics:
     - Cosine similarity
     - Mean Squared Difference (MSD)
     - Pearson correlation

3. **Item-based Collaborative Filtering**
   - Finds similar items based on user ratings
   - Recommends items similar to what the user has liked
   - Also supports multiple similarity metrics

## Performance Evaluation

The project evaluates algorithms using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Additional metrics for deeper analysis

## Parameter Analysis

The analysis includes:
- Impact of different similarity metrics
- Effect of neighborhood size on performance
- Optimal parameter tuning for each algorithm
- Comparative analysis of algorithm performance

## Usage

To run the complete analysis:
```bash
python recommender_analysis.py
```

## Output Analysis

The script generates comprehensive outputs including:

1. Performance Metrics
   - Detailed MAE and RMSE scores for each algorithm
   - Comparative performance analysis
   - Statistical significance testing

2. Visualization Results
   - Plots comparing similarity metrics
   - Impact of neighborhood size on performance
   - Algorithm performance comparison charts
   - Optimal parameter visualizations

3. Recommendations
   - Best performing algorithm for different scenarios
   - Optimal parameter configurations
   - Practical recommendations for implementation

## Technical Details

- Built with Python 3.x
- Uses NumPy for numerical computations
- Utilizes Pandas for data manipulation
- Implements Matplotlib and Seaborn for visualization
- Follows clean code principles and best practices

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Add new recommendation algorithms
- Enhance visualization capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.
