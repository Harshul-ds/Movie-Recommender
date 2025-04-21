import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, accuracy
from surprise import SVD  # for PMF
from surprise import KNNBasic
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
import os

def load_data(file_path):
    """Load the ratings data."""
    df = pd.read_csv(file_path)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    return data

def evaluate_algorithms(data):
    """Evaluate PMF, User-based CF, and Item-based CF using 5-fold CV."""
    # Initialize algorithms
    pmf = SVD()
    user_cf = KNNBasic(sim_options={'user_based': True})
    item_cf = KNNBasic(sim_options={'user_based': False})
    
    algorithms = [
        ('PMF', pmf),
        ('User-based CF', user_cf),
        ('Item-based CF', item_cf)
    ]
    
    results = {}
    kf = KFold(n_splits=5, random_state=42)
    
    for name, algo in algorithms:
        mae_scores = []
        rmse_scores = []
        
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            mae_scores.append(accuracy.mae(predictions))
            rmse_scores.append(accuracy.rmse(predictions))
        
        results[name] = {
            'MAE': np.mean(mae_scores),
            'RMSE': np.mean(rmse_scores)
        }
    
    return results

def evaluate_similarity_metrics(data):
    """Evaluate different similarity metrics for CF methods."""
    sim_options = [
        {'name': 'cosine', 'user_based': True},
        {'name': 'msd', 'user_based': True},
        {'name': 'pearson', 'user_based': True},
        {'name': 'cosine', 'user_based': False},
        {'name': 'msd', 'user_based': False},
        {'name': 'pearson', 'user_based': False}
    ]
    
    results = {}
    kf = KFold(n_splits=5, random_state=42)
    
    for sim_option in sim_options:
        algo = KNNBasic(sim_options={
            'name': sim_option['name'],
            'user_based': sim_option['user_based']
        })
        
        mae_scores = []
        rmse_scores = []
        
        for trainset, testset in kf.split(data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            mae_scores.append(accuracy.mae(predictions))
            rmse_scores.append(accuracy.rmse(predictions))
        
        key = f"{'User' if sim_option['user_based'] else 'Item'}-based {sim_option['name']}"
        results[key] = {
            'MAE': np.mean(mae_scores),
            'RMSE': np.mean(rmse_scores)
        }
    
    return results

def evaluate_n_neighbors(data, k_values):
    """Evaluate impact of number of neighbors."""
    results = {}
    kf = KFold(n_splits=5, random_state=42)
    
    for k in k_values:
        for user_based in [True, False]:
            algo = KNNBasic(k=k, sim_options={'user_based': user_based})
            mae_scores = []
            rmse_scores = []
            
            for trainset, testset in kf.split(data):
                algo.fit(trainset)
                predictions = algo.test(testset)
                mae_scores.append(accuracy.mae(predictions))
                rmse_scores.append(accuracy.rmse(predictions))
            
            key = f"{'User' if user_based else 'Item'}-based k={k}"
            results[key] = {
                'MAE': np.mean(mae_scores),
                'RMSE': np.mean(rmse_scores)
            }
    
    return results

def plot_similarity_results(results):
    """Plot similarity metric comparison."""
    metrics = ['MAE', 'RMSE']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        user_results = {k: v[metric] for k, v in results.items() if k.startswith('User')}
        item_results = {k: v[metric] for k, v in results.items() if k.startswith('Item')}
        
        x = np.arange(len(user_results))
        width = 0.35
        
        plt.bar(x - width/2, user_results.values(), width, label='User-based')
        plt.bar(x + width/2, item_results.values(), width, label='Item-based')
        
        plt.xlabel('Similarity Metric')
        plt.ylabel(metric)
        plt.title(f'{metric} by Similarity Metric')
        plt.xticks(x, ['Cosine', 'MSD', 'Pearson'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'similarity_{metric.lower()}_comparison.png')
        plt.close()

def plot_neighbors_results(results, k_values):
    """Plot number of neighbors comparison."""
    metrics = ['MAE', 'RMSE']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        user_results = []
        item_results = []
        for k in k_values:
            user_results.append(results[f'User-based k={k}'][metric])
            item_results.append(results[f'Item-based k={k}'][metric])
        
        plt.plot(k_values, user_results, 'o-', label='User-based')
        plt.plot(k_values, item_results, 'o-', label='Item-based')
        
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel(metric)
        plt.title(f'{metric} vs Number of Neighbors')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'neighbors_{metric.lower()}_comparison.png')
        plt.close()

def main():
    # Check if ratings_small.csv exists
    if not os.path.exists('ratings_small.csv'):
        print("Please download the MovieLens dataset and place ratings_small.csv in the current directory")
        return
    
    # Load data
    data = load_data('ratings_small.csv')
    
    # Basic algorithm comparison
    print("\nEvaluating basic algorithms...")
    basic_results = evaluate_algorithms(data)
    print("\nBasic Algorithm Results:")
    for algo, metrics in basic_results.items():
        print(f"{algo}:")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
    
    # Similarity metrics comparison
    print("\nEvaluating similarity metrics...")
    sim_results = evaluate_similarity_metrics(data)
    plot_similarity_results(sim_results)
    
    # Number of neighbors comparison
    print("\nEvaluating impact of number of neighbors...")
    k_values = [5, 10, 20, 30, 40, 50, 60, 80, 100]
    neighbor_results = evaluate_n_neighbors(data, k_values)
    plot_neighbors_results(neighbor_results, k_values)
    
    # Find best k
    user_best_k = min(k_values, key=lambda k: neighbor_results[f'User-based k={k}']['RMSE'])
    item_best_k = min(k_values, key=lambda k: neighbor_results[f'Item-based k={k}']['RMSE'])
    
    print(f"\nBest number of neighbors (k):")
    print(f"User-based CF: {user_best_k}")
    print(f"Item-based CF: {item_best_k}")

if __name__ == "__main__":
    main()
