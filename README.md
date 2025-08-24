# 🎬 IMDB Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.6%2B-green)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced machine learning system for analyzing sentiment in movie reviews. This project implements and compares multiple classification algorithms to determine whether movie reviews express positive or negative sentiment, achieving **90.05% accuracy** with the best-performing model and consistent **87%+ accuracy** across all implementations.

## ✨ Features

- 🚀 **Multiple ML Models**: Logistic Regression, Naive Bayes, and Support Vector Machine
- 🧹 **Advanced Text Preprocessing**: HTML cleaning, stopword removal, and text normalization
- 📊 **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- 🎯 **Interactive Interface**: Real-time sentiment prediction for custom reviews
- 📈 **Model Comparison**: Side-by-side performance analysis with statistical insights
- 💾 **Batch Processing**: Analyze multiple reviews simultaneously
- 🎨 **Rich Visualizations**: Feature importance, confusion matrices, and performance charts

## 🎯 Performance Results

| Model | Accuracy | F1-Score | Precision | Recall | Key Strengths |
|-------|----------|----------|-----------|--------|---------------|
| **🏆 Logistic Regression** | **90.05%** | **0.9014** | 0.90 | 0.90 | Best overall performance, interpretability |
| **SVM (Linear)** | 89.16% | 0.8919 | 0.89 | 0.89 | Robust decision boundaries, consistent |
| **Naive Bayes** | 87.07% | 0.8725 | 0.87 | 0.87 | Fast training, excellent for text |

> 🎉 **Outstanding Results**: All models achieved **87%+ accuracy** with balanced precision and recall across both sentiment classes!

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/imdb-sentiment-analyzer.git
   cd imdb-sentiment-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download the IMDB Dataset from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Place `IMDB_Dataset.csv` in the project root directory

4. **Run the analysis**
   ```bash
   python sentiment_analysis.py
   ```

### Quick Demo

```python
from sentiment_analysis import predict_sentiment

# Analyze a single review
result = predict_sentiment("This movie was absolutely fantastic! Great acting and storyline.")
print(result['predictions'])

# Interactive mode
interactive_interface()
```

## 📊 Dataset Information

- **Source**: IMDB Movie Reviews Dataset
- **Size**: 50,000 movie reviews
- **Balance**: 50% positive, 50% negative reviews
- **Features**: Review text and sentiment labels
- **Preprocessing**: HTML removal, text normalization, stopword filtering

## 🧠 Model Architecture

### Text Preprocessing Pipeline
1. **Text Cleaning**: Remove HTML tags, URLs, and special characters
2. **Normalization**: Convert to lowercase and handle whitespace
3. **Filtering**: Remove stopwords and short words (<3 characters)
4. **Vectorization**: TF-IDF with unigrams and bigrams (max 10,000 features)

### Machine Learning Models
- **Logistic Regression**: Linear model with L2 regularization
- **Multinomial Naive Bayes**: Probabilistic classifier optimized for text
- **Linear SVM**: Support Vector Machine with linear kernel

## 📈 Key Features & Analysis

### 🔍 Feature Importance Analysis
The system identifies the most influential words for sentiment classification:

**Top Positive Indicators:**
- "excellent", "amazing", "brilliant", "outstanding"
- "highly recommend", "masterpiece"

**Top Negative Indicators:**  
- "terrible", "awful", "boring", "waste"
- "completely disappointed", "poorly made"

### 📊 Comprehensive Visualizations
- Model performance comparison charts
- Confusion matrices for error analysis
- Feature importance plots
- Statistical distribution analysis

## 🛠️ Usage Examples

### Basic Sentiment Analysis
```python
# Single review prediction
review = "The cinematography was breathtaking and the story was compelling."
result = predict_sentiment(review)

for model_name, prediction in result['predictions'].items():
    print(f"{model_name}: {prediction['sentiment']} ({prediction['confidence']:.1%})")
```

### Batch Analysis
```python
# Analyze multiple reviews
reviews = [
    "Great movie with excellent acting!",
    "Boring plot and terrible dialogue.",
    "Average film, nothing special."
]

results = batch_analyze_reviews(reviews)
save_predictions_to_file(results, "analysis_results.txt")
```

### Interactive Mode
```python
# Launch interactive interface
interactive_interface()
# Enter reviews in real-time and get instant sentiment predictions
```

## 📁 Project Structure

```
imdb-sentiment-analyzer/
│
├── sentiment_analysis.py      # Main analysis script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT license
│
├── data/
│   └── IMDB_Dataset.csv      # Dataset (download separately)
│
├── results/
│   ├── model_performance.png  # Performance visualizations
│   ├── confusion_matrix.png   # Model accuracy analysis
│   └── feature_importance.png # Top sentiment indicators
│
└── reports/
    └── analysis_report.md     # Detailed technical report
```

## 🔧 Advanced Configuration

### TF-IDF Parameters
```python
tfidf = TfidfVectorizer(
    max_features=10000,      # Top 10K most important features
    ngram_range=(1, 2),      # Unigrams and bigrams
    min_df=2,                # Ignore rare terms
    max_df=0.95,             # Ignore very common terms
    sublinear_tf=True        # Apply sublinear scaling
)
```

### Model Hyperparameters
- **Logistic Regression**: C=1.0, max_iter=1000
- **Naive Bayes**: alpha=1.0 (Laplace smoothing)
- **SVM**: C=1.0, max_iter=2000

## 🎯 Business Applications

- **Movie Studios**: Analyze audience feedback and reviews
- **Streaming Platforms**: Content recommendation systems
- **Marketing Teams**: Campaign sentiment monitoring
- **Film Critics**: Automated review classification
- **Research**: Sentiment analysis methodology studies

## 📊 Performance Metrics

### Model Evaluation
- **Accuracy**: Overall prediction correctness
- **F1-Score**: Balanced precision and recall
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **Confusion Matrix**: Detailed error analysis

### Feature Analysis
- **TF-IDF Vectorization**: 10,000 most informative features
- **N-gram Analysis**: Unigrams and bigrams for context
- **Coefficient Analysis**: Linear model interpretability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IMDB Dataset**: Thanks to the creators of the IMDB movie reviews dataset
- **scikit-learn**: For providing excellent machine learning tools
- **NLTK**: For natural language processing capabilities
- **Kaggle Community**: For dataset hosting and ML resources

---

⭐ **Star this repository if you found it helpful!** ⭐
