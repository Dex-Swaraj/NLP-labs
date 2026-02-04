# Advanced NLP Practicals - Comprehensive Guide

This repository contains **practicals** for Natural Language Processing that go beyond basic requirements with cutting-edge features and professional visualizations.

---

## Table of Contents

1. [Practical 1: Advanced Tokenization & Stemming](#practical-1)
2. [Practical 2: BOW, TF-IDF & Word2Vec](#practical-2)
3. [Practical 3: Text Preprocessing & Feature Engineering](#practical-3)
4. [Practical 4: Named Entity Recognition](#practical-4)

---

## Practical 1: Advanced Tokenization & Stemming

### Standout Features
- **6 tokenization methods** including custom emoji tokenizer
- **Porter & Snowball stemmer** comparison with statistical analysis
- **Interactive dashboards** with 6+ visualizations
- **Performance benchmarking** (speed comparison)
- **Compression ratio analysis** for stemming effectiveness

### What You Get
- Tokenization comparison across 6 methods
- 3D visualization of stemming patterns
- Performance metrics (time taken per method)
- Detailed stemmer agreement analysis
- Beautiful comparative dashboards

### Unique Additions
- Custom regex tokenizer for social media (preserves emojis & hashtags)
- Token length distribution analysis
- Stemming compression ratio by word length category
- Agreement rate visualization between stemmers

### Output Files
```
tokenization_comparison.png      # Interactive dashboard
stemming_analysis.png            # Deep stemming insights
stemmer_comparison.csv           # Detailed comparison table
stemmer_differences.csv          # Difference analysis
```

### Run It
```bash
python practical1_tokenization_stemming.py
```

---

## Practical 2: BOW, TF-IDF & Word2Vec

### Standout Features
- **Word2Vec CBOW & Skip-gram** models
- **3D embedding visualization** with PCA
- **Semantic clustering** (K-means on embeddings)
- **Document similarity heatmaps** (3 methods compared)
- **Comparative analysis** dashboard

### What You Get
- Count & normalized BOW representations
- TF-IDF with IDF value analysis
- 100-dimensional Word2Vec embeddings
- Word similarity search functionality
- Sparsity comparison across methods

###  Unique Additions
- 3D interactive Word2Vec space (color-coded by word length)
- Semantic word clustering with visualization
- Comprehensive 6-panel comparison dashboard
- Document similarity matrices for all 3 methods
- Top terms analysis (BOW vs TF-IDF overlay)

### Output Files
```
bow_counts.csv                   # Count occurrence
bow_normalized.csv               # Normalized counts
tfidf_scores.csv                 # TF-IDF matrix
idf_values.csv                   # IDF scores per term
word2vec_cbow.model              # Trained CBOW model
word2vec_skipgram.model          # Trained Skip-gram model
document_vectors.npy             # Document embeddings
word2vec_3d.png                  # 3D visualization
word_clusters.png                # Semantic clustering
similarity_*.png                 # 3 heatmaps
comparative_analysis.png         # Comprehensive comparison
```

###  Run It
```bash
python practical2_bow_tfidf_word2vec.py
```

---

##  Practical 3: Text Preprocessing & Feature Engineering

### Standout Features
- **Text quality scoring system** (8 metrics)
- **Data augmentation** (synonym replacement)
- **Before/after analysis** with 6 visualizations
- **Feature importance analysis** for TF-IDF
- **Label encoding** with mapping

###  What You Get
- Comprehensive cleaning (URLs, emails, HTML, etc.)
- POS-aware lemmatization
- Smart stopword removal
- TF-IDF feature extraction (100 features)
- Quality metrics tracking

### Unique Additions
- Text quality scoring (lexical diversity, readability, etc.)
- Augmented dataset generation (2x original size)
- 6-panel preprocessing impact dashboard
- Feature correlation heatmap
- Compression ratio analysis (before/after)

### Output Files
```
preprocessed_texts.csv           # Complete pipeline results
augmented_texts.csv              # Augmented dataset
tfidf_features.csv               # TF-IDF representation
label_encoding.csv               # Encoded labels
quality_metrics.csv              # Before/after metrics
preprocessing_impact.png         # 6-panel dashboard
feature_importance.png           # Feature analysis
```

### Run It
```bash
python practical3_preprocessing_tfidf.py
```

---

## Practical 4: Named Entity Recognition

### Standout Features
- **Multi-model NER** (spaCy + NLTK + Custom)
- **Custom entity recognition** (tech companies, AI terms, languages)
- **Intelligent entity merging** (deduplication)
- **Confusion matrix** with normalization
- **4-panel performance dashboard**

###  What You Get
- Entities from 3 different systems
- Accuracy, Precision, Recall, F1-Score
- Per-class detailed metrics
- Entity distribution analysis
- Source comparison (which model found what)

###  Unique Additions
- Custom domain-specific entity recognizer (TECH_COMPANY, AI_TERM, etc.)
- Smart entity combination (removes duplicates intelligently)
- Entity length distribution analysis
- Top 15 most frequent entities
- Precision-Recall scatter plot
- Normalized confusion matrix

### Output Files
```
extracted_entities.csv           # All entities found
ner_predictions.csv              # Per-document predictions
ner_metrics.csv                  # Performance summary
classification_report.csv        # Detailed per-class metrics
entity_distribution.png          # 4-panel distribution
confusion_matrix.png             # Raw & normalized
performance_metrics.png          # 4-panel dashboard
```

###  Run It
```bash
python practical4_ner_advanced.py
```

---

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Install Dependencies
```bash
# Core NLP libraries
pip install nltk spacy gensim --break-system-packages

# Data science libraries
pip install numpy pandas scikit-learn --break-system-packages

# Visualization libraries
pip install matplotlib seaborn --break-system-packages

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data (done automatically in scripts)
```

### One-Line Install
```bash
pip install nltk spacy gensim numpy pandas scikit-learn matplotlib seaborn --break-system-packages && python -m spacy download en_core_web_sm
```

---

## Quick Start

### Run All Practicals
```bash
# Practical 1
python practical1_tokenization_stemming.py

# Practical 2
python practical2_bow_tfidf_word2vec.py

# Practical 3
python practical3_preprocessing_tfidf.py

# Practical 4
python practical4_ner_advanced.py
```