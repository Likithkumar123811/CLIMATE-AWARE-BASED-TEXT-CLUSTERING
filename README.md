# News Category Classification System

A comprehensive news category classification system that combines vector embeddings, ChromaDB, DeepSeek reasoning, and clustering analysis to classify news articles and provide detailed explanations.

##  Features

- **Data Preparation**: Load and clean Kaggle News Category Dataset
- **Vector Embeddings**: Generate embeddings using ChromaDB + LangChain
- **Vector Storage & Retrieval**: Advanced similarity search and retrieval
- **DeepSeek Integration**: AI-powered category prediction and explanations
- **Clustering Analysis**: KMeans and Agglomerative clustering
- **Comprehensive Analysis**: Results collection and evaluation
- **Rich Visualizations**: Interactive plots and dashboards
- **Example Cases**: Three demonstration cases with detailed analysis

##  Requirements

### Python Dependencies
```
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
chromadb==0.4.18
langchain==0.1.0
langchain-community==0.0.10
openai==1.6.1
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
requests==2.31.0
tqdm==4.66.1
jupyter==1.0.0
ipykernel==6.29.0
```

### External Requirements
- **Kaggle News Category Dataset**: Download from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **DeepSeek API Key**: For AI-powered explanations (optional but recommended)

##  Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd ML_DL_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download `News_Category_Dataset_v3.csv` from Kaggle
   - Place it in the project root directory

4. **Set up environment variables** (optional)
   ```bash
   export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
   ```

##  Quick Start

### Option 1: Run Complete Pipeline
```python
python main_pipeline.py
```

### Option 2: Run Individual Components
```python
# Data preparation
from data_preparation import DataPreparator
preparator = DataPreparator("News_Category_Dataset_v3.csv")
preparator.load_data()
preparator.clean_data()
preparator.split_data()

# Embedding generation
from embedding_generator import EmbeddingGenerator
generator = EmbeddingGenerator()
generator.generate_embeddings_from_dataframe(preparator.train_df)

# And so on...
```

##  Project Structure

```
ML_DL_project/
├── 1_data_preparation.py          # Data loading and cleaning
├── 2_embedding_generator.py       # ChromaDB + LangChain embeddings
├── 3_vector_storage_retrieval.py  # Vector storage and retrieval
├── 4_deepseek_integration.py      # DeepSeek API integration
├── 5_clustering_pipeline.py       # Clustering analysis
├── 6_results_analysis.py          # Results collection and analysis
├── 7_visualization.py             # Visualization generation
├── 8_example_cases.py             # Example cases demonstration
├── main_pipeline.py               # Main integration script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── output/                        # Generated results (created automatically)
    ├── results/
    │   ├── classification_results.csv
    │   ├── analysis_summary.json
    │   └── detailed_report.txt
    ├── visualizations/
    │   ├── category_distribution.png
    │   ├── confusion_matrix.png
    │   ├── accuracy_by_category.png
    │   └── ...
    └── example_cases_results.json
```

##  Component Details

### 1. Data Preparation (`1_data_preparation.py`)
- Loads Kaggle News Category Dataset
- Keeps only 'short_description' and 'category' columns
- Performs text cleaning (lowercase, special characters)
- Splits into train/test (80/20) with stratification

### 2. Embedding Generation (`2_embedding_generator.py`)
- Uses ChromaDB for vector storage
- Generates embeddings for news descriptions
- Stores embeddings with metadata (category, length, etc.)
- Supports batch processing

### 3. Vector Storage & Retrieval (`3_vector_storage_retrieval.py`)
- Advanced similarity search
- Top-k most similar articles retrieval
- Category-based filtering
- Metadata-based search
- Export/import functionality

### 4. DeepSeek Integration (`4_deepseek_integration.py`)
- Category prediction using DeepSeek API
- Natural language explanations
- Mismatch analysis
- Batch processing with rate limiting
- Response caching

### 5. Clustering Pipeline (`5_clustering_pipeline.py`)
- KMeans and Agglomerative clustering
- Optimal cluster number detection
- Cluster-category comparison
- Dimensionality reduction (PCA, t-SNE)
- Comprehensive evaluation metrics

### 6. Results Analysis (`6_results_analysis.py`)
- Collects all prediction results
- Calculates accuracy metrics
- Error pattern analysis
- Category-wise performance
- Detailed reporting

### 7. Visualization (`7_visualization.py`)
- Category distribution plots
- Clustering visualizations
- Confusion matrices
- Accuracy analysis
- Error pattern visualization
- Interactive dashboards

### 8. Example Cases (`8_example_cases.py`)
- Three demonstration cases:
  1. **Mismatch**: TRAVEL vs ENTERTAINMENT
  2. **Reasonable but Incorrect**: CRIME vs WORLD NEWS
  3. **Correct Match**: MEDIA
- Detailed analysis and explanations

## 📊 Example Cases

The system demonstrates three specific cases:

### Case 1: Mismatch (TRAVEL vs ENTERTAINMENT)
**Input**: "The new theme park in Orlando features the world's largest roller coaster..."
**Ground Truth**: TRAVEL
**Expected Behavior**: Should be classified as TRAVEL but might be misclassified as ENTERTAINMENT

### Case 2: Reasonable but Incorrect (CRIME vs WORLD NEWS)
**Input**: "International authorities have arrested a major drug trafficking ring..."
**Ground Truth**: CRIME
**Expected Behavior**: Could reasonably be classified as WORLD NEWS due to international scope

### Case 3: Correct Match (MEDIA)
**Input**: "The streaming service announced record-breaking viewership numbers..."
**Ground Truth**: MEDIA
**Expected Behavior**: Should be correctly classified as MEDIA

##  Output and Results

The pipeline generates:

1. **Classification Results** (`results/classification_results.csv`)
   - Input text, ground truth, predictions
   - Similarity scores and confidence levels
   - DeepSeek explanations and reasoning

2. **Analysis Summary** (`results/analysis_summary.json`)
   - Overall accuracy metrics
   - Category-wise performance
   - Error patterns and confusion matrices

3. **Detailed Report** (`results/detailed_report.txt`)
   - Human-readable analysis report
   - Performance insights and recommendations

4. **Visualizations** (`visualizations/`)
   - Category distribution plots
   - Clustering visualizations
   - Accuracy analysis charts
   - Error pattern visualizations

5. **Example Cases** (`example_cases_results.json`)
   - Detailed analysis of three demonstration cases
   - Vector vs DeepSeek comparison
   - Explanation quality assessment

##  Usage Examples

### Basic Usage
```python
from main_pipeline import NewsClassificationPipeline

# Initialize pipeline
pipeline = NewsClassificationPipeline(
    data_path="News_Category_Dataset_v3.csv",
    deepseek_api_key="your_api_key",
    sample_size=100  # For demo
)

# Run complete pipeline
pipeline.run_complete_pipeline()
```

### Individual Component Usage
```python
# Data preparation
from data_preparation import DataPreparator
preparator = DataPreparator("News_Category_Dataset_v3.csv")
preparator.load_data()
preparator.clean_data()
preparator.split_data()

# Vector similarity search
from vector_storage_retrieval import VectorStorageRetrieval
storage = VectorStorageRetrieval()
similar_articles = storage.retrieve_top_k_similar("Your news text here", k=5)

# DeepSeek prediction
from deepseek_integration import DeepSeekReasoning
deepseek = DeepSeekReasoning(api_key="your_api_key")
prediction = deepseek.predict_category("Your text", ["POLITICS", "SPORTS", "TECH"])
```

##  Configuration

### Environment Variables
```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

### Pipeline Parameters
```python
pipeline = NewsClassificationPipeline(
    data_path="path/to/dataset.csv",
    deepseek_api_key="your_api_key",  # Optional
    output_dir="output",              # Output directory
    sample_size=100                   # None for full dataset
)
```

## Troubleshooting

### Common Issues

1. **Dataset not found**
   - Ensure `News_Category_Dataset_v3.csv` is in the project root
   - Update the `data_path` parameter if needed

2. **ChromaDB errors**
   - Check if ChromaDB is properly installed
   - Ensure write permissions in the project directory

3. **DeepSeek API errors**
   - Verify API key is correct
   - Check API rate limits
   - Ensure internet connectivity

4. **Memory issues**
   - Use `sample_size` parameter to limit dataset size
   - Process data in smaller batches

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Reference

### DataPreparator
- `load_data()`: Load dataset from CSV
- `clean_data()`: Clean and preprocess data
- `split_data()`: Split into train/test sets

### EmbeddingGenerator
- `generate_embeddings_from_dataframe()`: Generate embeddings
- `get_similar_documents()`: Retrieve similar documents
- `get_collection_stats()`: Get database statistics

### VectorStorageRetrieval
- `retrieve_top_k_similar()`: Find similar articles
- `find_most_similar_category()`: Predict category
- `get_category_statistics()`: Category analysis

### DeepSeekReasoning
- `predict_category()`: Predict news category
- `generate_explanation()`: Generate detailed explanation
- `analyze_mismatch()`: Analyze incorrect predictions

### NewsClusteringPipeline
- `find_optimal_clusters()`: Find best number of clusters
- `perform_clustering()`: Run clustering algorithm
- `analyze_clusters()`: Analyze clustering results

### ResultsAnalyzer
- `collect_prediction_results()`: Collect all results
- `analyze_results()`: Perform comprehensive analysis
- `save_results()`: Save results to files

### NewsClassificationVisualizer
- `plot_category_distribution()`: Category distribution plot
- `plot_clustering_results()`: Clustering visualization
- `plot_confusion_matrix()`: Confusion matrix plot
- `create_interactive_dashboard()`: Interactive dashboard

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request



##  Acknowledgments

- [Kaggle News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [DeepSeek](https://www.deepseek.com/) for AI reasoning
- [LangChain](https://langchain.com/) for embeddings
- [scikit-learn](https://scikit-learn.org/) for clustering

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the example cases
3. Open an issue on GitHub
4. Contact the development team

---


