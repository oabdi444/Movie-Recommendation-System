# Intelligent Movie Recommendation System

A sophisticated machine learning system that provides personalised movie recommendations using multiple algorithmic approaches including content-based filtering, collaborative filtering, and hybrid ensemble methods.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Performance Metrics](#performance-metrics)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a production-ready movie recommendation system that combines multiple machine learning approaches to deliver accurate and diverse recommendations. The system processes the TMDB 5000 Movie Dataset and utilises advanced natural language processing techniques to understand movie content and user preferences.

### Key Achievements
- **Multi-Modal Approach**: Combines content-based, collaborative, and hybrid filtering
- **High Performance**: RMSE of 1.639 and MAE of 1.319 on collaborative filtering
- **Production Ready**: Comprehensive REST API with Docker deployment
- **Scalable Architecture**: Modular design supporting enterprise-level deployment

## Features

### Core Recommendation Engines
- **Content-Based Filtering**: TF-IDF vectorisation with cosine similarity analysis
- **Collaborative Filtering**: Singular Value Decomposition (SVD) implementation
- **Hybrid System**: Weighted ensemble combining multiple approaches

### Advanced Capabilities
- **Fuzzy Movie Search**: Intelligent search with partial matching
- **Genre-Based Recommendations**: Filter by preferred genres
- **Trending Analysis**: Dynamic trending movie identification
- **Director-Specific Recommendations**: Curated lists by filmmaker
- **Decade-Based Discovery**: Historical movie exploration
- **Similar User Analysis**: Recommendations based on user similarity

### Production Features
- **RESTful API**: Comprehensive endpoints for all functionality
- **Error Handling**: Robust validation and error management
- **Performance Monitoring**: Built-in analytics and logging
- **Docker Support**: Containerised deployment ready

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Algorithm Layer │    │   API Layer     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ TMDB Dataset    │───▶│ Content-Based   │───▶│ REST API        │
│ User Ratings    │    │ Collaborative   │    │ Search Engine   │
│ Movie Metadata  │    │ Hybrid Ensemble │    │ Recommendation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Backend**: Python 3.8+
- **Machine Learning**: scikit-learn, Surprise
- **Data Processing**: pandas, NumPy
- **NLP**: TF-IDF Vectorisation, NLTK
- **API**: Custom REST implementation
- **Deployment**: Docker, pickle serialisation
- **Visualisation**: matplotlib, seaborn

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Docker for containerised deployment

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/intelligent-movie-recommendation-system.git
   cd intelligent-movie-recommendation-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

### Docker Installation

```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

## Usage

### Quick Start

```python
from movie_recommender import MovieRecommendationAPI

# Initialize the system
api = MovieRecommendationAPI(movies_df, content_recommender, 
                            collaborative_recommender, hybrid_recommender)

# Get content-based recommendations
recommendations = api.recommend(
    movie_title="Inception", 
    method="content", 
    n_recommendations=5
)

# Get collaborative filtering recommendations
user_recs = api.recommend(
    user_id=123, 
    method="collaborative", 
    n_recommendations=5
)

# Get hybrid recommendations
hybrid_recs = api.recommend(
    movie_title="Avatar", 
    user_id=456, 
    method="hybrid", 
    n_recommendations=10
)
```

### Advanced Features

```python
from advanced_features import AdvancedFeatures

advanced = AdvancedFeatures(movies_df, api)

# Get trending movies
trending = advanced.trending_movies(n_movies=10)

# Genre-specific recommendations
action_movies = advanced.genre_recommendations(['Action', 'Adventure'], n_movies=5)

# Director filmography
nolan_films = advanced.director_recommendations('Christopher Nolan', n_movies=5)
```

## API Documentation

### Core Endpoints

#### Search Movies
```python
api.search_movie(query="dark knight")
# Returns: ['The Dark Knight', 'The Dark Knight Rises', ...]
```

#### Get Movie Information
```python
api.get_movie_info("Inception")
# Returns: Detailed movie metadata including cast, genres, rating
```

#### Generate Recommendations
```python
api.recommend(movie_title="Movie", user_id=123, method="hybrid", n_recommendations=5)
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `movie_title` | string | No* | Movie title for content-based recommendations |
| `user_id` | integer | No* | User ID for collaborative filtering |
| `method` | string | Yes | Recommendation method: 'content', 'collaborative', 'hybrid' |
| `n_recommendations` | integer | No | Number of recommendations (default: 5) |

*At least one of `movie_title` or `user_id` required depending on method

### Response Format

```json
{
  "method": "hybrid",
  "input_movie": "Inception",
  "user_id": 123,
  "recommendations": [
    {
      "title": "Interstellar",
      "hybrid_score": 0.856,
      "genres": "Sci-Fi, Drama",
      "rating": 8.1,
      "year": 2014
    }
  ],
  "count": 5
}
```

## Performance Metrics

### Model Performance
- **Collaborative Filtering RMSE**: 1.639
- **Collaborative Filtering MAE**: 1.319
- **Content-Based Similarity**: Cosine similarity with TF-IDF
- **Dataset Coverage**: 4,803 movies processed
- **Feature Dimensionality**: 5,000 TF-IDF features

### System Performance
- **API Response Time**: <200ms average
- **Memory Usage**: ~2GB for full model loading
- **Recommendation Generation**: <50ms per request
- **Concurrent Users**: Scalable with containerisation

## Dataset

### TMDB 5000 Movie Dataset
- **Source**: [Kaggle TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Size**: 4,803 movies with comprehensive metadata
- **Features**: Genres, keywords, cast, crew, budgets, revenues, ratings
- **Preprocessing**: JSON parsing, text cleaning, feature engineering

### Synthetic User Ratings
- **Users**: 1,000 synthetic users
- **Ratings**: 50,000 user-movie interactions
- **Scale**: 1-10 rating system
- **Distribution**: Gaussian distribution around movie averages

## Model Details

### Content-Based Filtering
- **Algorithm**: TF-IDF Vectorisation + Cosine Similarity
- **Features**: Movie genres, keywords, cast, director
- **Preprocessing**: Text normalisation, stop word removal
- **Similarity Metric**: Cosine similarity matrix

### Collaborative Filtering
- **Algorithm**: Singular Value Decomposition (SVD)
- **Parameters**: 100 factors, 20 epochs
- **Optimisation**: Grid search for hyperparameter tuning
- **Validation**: Train-test split with cross-validation

### Hybrid System
- **Combination Method**: Weighted linear combination
- **Default Weights**: 60% content-based, 40% collaborative
- **Normalisation**: Min-max scaling for score alignment
- **Ensemble Strategy**: Score aggregation with ranking

## Deployment

### Local Development
```bash
python app.py  # Starts local development server
```

### Production Deployment
```bash
# Using Docker
docker build -t movie-recommender .
docker run -d -p 8000:8000 --name recommender-app movie-recommender

# Using Docker Compose
docker-compose up -d
```

### Environment Variables
```bash
export MODEL_PATH=/path/to/saved/models
export API_PORT=8000
export LOG_LEVEL=INFO
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Project Structure

```
intelligent-movie-recommendation-system/
├── src/
│   ├── content_based.py          # Content-based filtering implementation
│   ├── collaborative.py          # Collaborative filtering with SVD
│   ├── hybrid.py                 # Hybrid recommendation system
│   ├── api.py                    # REST API implementation
│   ├── advanced_features.py      # Additional recommendation features
│   └── utils.py                  # Utility functions
├── data/
│   ├── tmdb_5000_movies.csv      # Movie dataset
│   ├── tmdb_5000_credits.csv     # Credits dataset
│   └── processed/                # Preprocessed data files
├── models/
│   ├── content_model.pkl         # Serialised content-based model
│   ├── collaborative_model.pkl   # Serialised collaborative model
│   └── tfidf_vectorizer.pkl      # TF-IDF vectoriser
├── notebooks/
│   ├── exploration.ipynb         # Data exploration
│   ├── model_development.ipynb   # Model development
│   └── evaluation.ipynb          # Performance evaluation
├── tests/
│   ├── test_api.py               # API unit tests
│   ├── test_models.py            # Model unit tests
│   └── test_utils.py             # Utility function tests
├── docker/
│   ├── Dockerfile                # Container configuration
│   └── docker-compose.yml        # Multi-container setup
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── app.py                        # Main application entry point
```

## Contributing

We welcome contributions to improve the recommendation system. Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with appropriate tests
4. Run the test suite (`python -m pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Include docstrings for all functions and classes
- Add unit tests for new functionality
- Update documentation as needed

### Areas for Contribution
- Deep learning model integration (neural collaborative filtering)
- Real-time recommendation updates
- A/B testing framework
- Enhanced evaluation metrics
- UI/UX improvements

## Future Enhancements

### Planned Features
- **Deep Learning Integration**: Neural collaborative filtering and autoencoders
- **Real-Time Updates**: Streaming recommendation updates
- **A/B Testing**: Built-in experimentation framework
- **Multi-Language Support**: International movie recommendations
- **Social Features**: Friend-based recommendations
- **Explanation System**: Interpretable recommendation reasons

### Technical Improvements
- **Distributed Computing**: Spark integration for larger datasets
- **Caching Layer**: Redis for improved response times
- **Monitoring**: Comprehensive logging and alerting
- **Auto-Scaling**: Kubernetes deployment configuration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **TMDB**: For providing the comprehensive movie dataset
- **Surprise Library**: For collaborative filtering algorithms
- **scikit-learn**: For machine learning utilities
- **Open Source Community**: For the excellent Python ecosystem

Osman Hassan Abdi

---

**Built with passion for machine learning and recommendation systems**
