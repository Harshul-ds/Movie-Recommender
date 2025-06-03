# Movie Recommender System Analysis  
A comprehensive recommendation system analyzing framework implementing Probabilistic Matrix Factorization (PMF), User-based Collaborative Filtering, and Item-based Collaborative Filtering algorithms. The system evaluates performance using MAE, RMSE, Precision@K and Recall@K metrics with visualization capabilities for comparative analysis.  

**Key Features**  
- Three implemented algorithms: PMF (Bayesian probabilistic approach with latent factor modeling), User-based CF (dynamic neighborhood selection with multiple similarity metrics), and Item-based CF (optimized for large catalogs)  
- Evaluation metrics including accuracy (MAE, RMSE), ranking (Precision@K, Recall@K), and efficiency (training/prediction speed)  
- Automated visualization generation for algorithm comparison, learning curves, and parameter tuning  
- Clean data pipeline supporting MovieLens dataset (ratings_small.csv and movies_metadata.csv)  

**Installation**  
```bash  
git clone https://github.com/Harshul-ds/Movie-Recommender.git  
cd Movie-Recommender  
python -m venv venv  
source venv/bin/activate  # Linux/Mac | venv\Scripts\activate for Windows  
pip install -r requirements.txt  
