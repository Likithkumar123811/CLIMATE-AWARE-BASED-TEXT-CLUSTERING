# Climate Aware Based Text Clustering

## Overview
This project implements a Climate Aware Text Clustering system using Machine Learning and NLP techniques.

The system processes climate-related text data, generates embeddings, performs clustering, and evaluates clustering performance.

---

## Features

- Climate text processing
- Text embedding generation
- ChromaDB integration
- Clustering and classification
- CSV report generation
- Evaluation metrics

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- ChromaDB
- NLP

---

## Project Structure

```text
soft computing/
│
├── main_pipeline.py
├── clustering_pipeline.py
├── embedding_generator.py
├── data_preparation.py
├── deepseek_integration.py
├── config.py
├── demo.py
├── climate_dataset_10_categories_5000.csv
│
├── chroma_db/
├── output/
└── README.md
```

---

## Installation

Install required libraries:

```bash
pip install pandas numpy scikit-learn chromadb sentence-transformers matplotlib
```

---

## Run the Project

```bash
python main_pipeline.py
```

---

## Output Files

- final_results.csv
- confusion_matrix.csv
- class_accuracy.csv
- category_distribution.csv
- per_class_metrics.csv

---

## Evaluation Metrics

- Accuracy
- Confusion Matrix
- ARI
- NMI

---

## Future Enhancements

- Real-time climate analysis
- Advanced NLP models
- Dashboard integration
- Multi-language support

---

## Author

Likith

---

## License

Academic project for educational purposes.
