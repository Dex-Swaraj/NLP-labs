# Advanced NLP Practicals - Comprehensive Guide

This repository contains **practicals** for Natural Language Processing that go beyond basic requirements with cutting-edge features and professional visualizations.

---

## Table of Contents

1. [Practical 1: Advanced Tokenization & Stemming](#practical-1)
2. [Practical 2: BOW, TF-IDF & Word2Vec](#practical-2)
3. [Practical 3: Text Preprocessing & Feature Engineering](#practical-3)
4. [Practical 4: Named Entity Recognition](#practical-4)
5. [Practical 5: WordNet Semantic Relationships](#practical-5)
6. [Practical 6: Machine Translation](#practical-6)
7. [Practical 7: NLTK Preprocessing Pipeline](#practical-7)
8. [Practical 8: Word Sense Disambiguation](#practical-8)
9. [Practical 9: Indian Language Sentiment Analysis](#practical-9)
10. [Practical 10: N-gram Auto-Complete](#practical-10)

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

## Practical 5: WordNet Semantic Relationships

### Standout Features
- **Multi-relationship analysis** (synonymy, antonymy, hypernymy, hyponymy, meronymy, holonymy)
- **3 similarity metrics** (Path, Wu-Palmer, Leacock-Chodorow)
- **Interactive semantic network graph** with color-coded relationships
- **Hierarchical taxonomy visualization** (hypernym trees)
- **Similarity matrices** with heatmaps
- **7-panel comprehensive dashboard**

### What You Get
- Complete semantic relationship extraction
- WordNet synset definitions and examples
- Pairwise similarity scores (multiple algorithms)
- Network visualization of word relationships
- Hierarchy trees showing concept generalization

### Unique Additions
- Semantic network graph with 4 relationship types
- Three different similarity algorithms compared
- Relationship richness scoring per word
- Synset distribution analysis
- Hierarchical depth visualization

### Output Files
```
semantic_relationships.csv       # Main relationship data
synset_definitions.csv          # Detailed synset information
similarity_scores.csv           # Pairwise similarity metrics
semantic_dashboard.png          # 7-panel analysis dashboard
semantic_network.png            # Interactive network graph
similarity_matrices.png         # 3 similarity heatmaps
hierarchy_tree.png              # Hypernym taxonomy tree
```

### Run It
```bash
python practical5_wordnet_semantics.py
```

---

## Practical 6: Machine Translation

### Standout Features
- **Multi-model translation** (Google Translate API, MarianMT)
- **5+ Indian languages** (Hindi, Bengali, Tamil, Telugu, Marathi)
- **BLEU score evaluation** (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
- **Back-translation quality checking**
- **n-gram precision analysis**
- **8-panel performance dashboard**

### What You Get
- Forward and backward translation
- Translation quality metrics (BLEU scores)
- Multi-language comparison
- Confidence scoring
- Similarity analysis

### Unique Additions
- Back-translation for quality assessment
- Multiple BLEU metrics (1-4 grams)
- Translation to 5+ Indian languages simultaneously
- Sentence-level similarity scoring (Jaccard)
- Quality categorization (Excellent/Good/Fair/Poor)

### Output Files
```
translations.csv                # Full translation results
multi_language_translations.csv # Translations to 5 Indian languages
bleu_analysis.csv               # Detailed BLEU scores
translation_dashboard.png       # 8-panel analysis dashboard
language_comparison.png         # Multi-language comparison
```

### Run It
```bash
python practical6_machine_translation.py
```

---

## Practical 7: NLTK Preprocessing Pipeline

### Standout Features
- **8-stage preprocessing pipeline** with quality tracking
- **Advanced POS tagging** with category analysis
- **POS transition probabilities** (bigram patterns)
- **Named entity recognition** with NLTK
- **Text complexity metrics** (Flesch Reading Ease, TTR)
- **Aspect-based analysis**

### What You Get
- Sentence and word tokenization
- Lemmatization with POS awareness
- Stopword filtering
- Named entity extraction
- Readability scoring
- POS distribution analysis

### Unique Additions
- Text complexity scoring (6 metrics)
- POS bigram transition analysis
- Syllable counting for readability
- 8-category POS classification
- Pipeline stage visualization
- Lexical diversity measurement

### Output Files
```
preprocessing_results.csv       # Complete preprocessing output
pos_analysis.csv               # POS tag distribution
named_entities.csv             # Extracted entities
pipeline_stages.png            # 4-panel pipeline visualization
complexity_metrics.png         # 6-panel complexity dashboard
pos_transitions.png            # POS bigram transitions
```

### Run It
```bash
python practical7_nltk_preprocessing.py
```

---

## Practical 8: Word Sense Disambiguation

### Standout Features
- **4 WSD algorithms** (Lesk, Adapted Lesk, Path Similarity, Wu-Palmer)
- **Context window optimization**
- **Confidence scoring** for predictions
- **Algorithm agreement analysis**
- **Ambiguity degree tracking**
- **7-panel comparison dashboard**

### What You Get
- Multiple disambiguation algorithms
- Sense definitions and explanations
- Confidence scores per prediction
- Algorithm consensus
- Ambiguity metrics

### Unique Additions
- Adapted Lesk with extended context
- Similarity-based WSD (2 algorithms)
- Confidence calculation (3 factors)
- Algorithm agreement matrix with heatmap
- Quality categorization by BLEU-like metrics
- Ambiguity vs confidence correlation

### Output Files
```
wsd_results.csv                 # Complete disambiguation results
algorithm_agreement.csv         # Algorithm agreement matrix
wsd_dashboard.png               # 7-panel analysis dashboard
algorithm_agreement.png         # Agreement heatmap
```

### Run It
```bash
python practical8_word_sense_disambiguation.py
```

---

## Practical 9: Indian Language Sentiment Analysis

### Standout Features
- **Multi-model sentiment** (VADER, TextBlob, Lexicon-based)
- **5+ Indian languages** (Hindi, Bengali, Tamil, Telugu, Marathi)
- **Translation-based approach** for Indian languages
- **Emotion classification** (joy, anger, sadness, fear, surprise)
- **Aspect-based sentiment** extraction
- **8-panel comprehensive dashboard**

### What You Get
- Sentiment classification (Positive/Negative/Neutral)
- Polarity and subjectivity scores
- Emotion detection
- Translation to English for analysis
- Consensus sentiment across models

### Unique Additions
- Custom Hindi sentiment lexicon
- Emotion keyword detection (5 emotions)
- Aspect-based sentiment (nouns as aspects)
- Multi-algorithm consensus
- Algorithm agreement visualization
- Translation quality for Indian languages

### Output Files
```
sentiment_results.csv          # Complete sentiment analysis
aspect_sentiments.csv          # Aspect-based sentiment analysis
sentiment_dashboard.png        # 8-panel analysis dashboard
```

### Run It
```bash
python practical9_sentiment_indian_lang.py
```

---

## Practical 10: N-gram Auto-Complete

### Standout Features
- **Multi-order N-grams** (Unigram to 5-gram)
- **MLE and Laplace smoothing**
- **Perplexity evaluation** for model quality
- **Context-aware predictions** with confidence
- **Text generation** capability
- **7-panel analysis dashboard**

### What You Get
- N-gram frequency analysis
- Auto-complete suggestions
- Next word probability distributions
- Text generation from seed
- Model quality metrics

### Unique Additions
- Dynamic model order selection (best available)
- Confidence scoring for predictions
- Sparsity analysis (singleton ratio)
- Top N-gram visualizations
- Text generation with greedy selection
- Vocabulary growth visualization

### Output Files
```
top_bigrams.csv                # Most frequent bigrams
top_trigrams.csv               # Most frequent trigrams
autocomplete_results.csv       # Auto-complete predictions
text_generation.csv            # Generated text examples
ngram_analysis.png             # 7-panel analysis dashboard
prediction_*.png               # Prediction visualizations
```

### Run It
```bash
python practical10_ngram_autocomplete.py
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

# Translation & Sentiment
pip install googletrans==4.0.0rc1 textblob vaderSentiment --break-system-packages

# Transformers for advanced models
pip install transformers sentencepiece sacremoses --break-system-packages

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data (done automatically in scripts)
```

### One-Line Install
```bash
pip install nltk spacy gensim numpy pandas scikit-learn matplotlib seaborn googletrans==4.0.0rc1 textblob vaderSentiment transformers sentencepiece sacremoses --break-system-packages && python -m spacy download en_core_web_sm
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

# Practical 5
python practical5_wordnet_semantics.py

# Practical 6
python practical6_machine_translation.py

# Practical 7
python practical7_nltk_preprocessing.py

# Practical 8
python practical8_word_sense_disambiguation.py

# Practical 9
python practical9_sentiment_indian_lang.py

# Practical 10
python practical10_ngram_autocomplete.py
```

---

## Output Structure

```
NLP/
├── practical1_tokenization_stemming.py
├── practical2_bow_tfidf_word2vec.py
├── practical3_preprocessing_tfidf.py
├── practical4_ner_advanced.py
├── practical5_wordnet_semantics.py
├── practical6_machine_translation.py
├── practical7_nltk_preprocessing.py
├── practical8_word_sense_disambiguation.py
├── practical9_sentiment_indian_lang.py
├── practical10_ngram_autocomplete.py
├── output_pract_1/
│   ├── tokenization_comparison.png
│   ├── stemming_analysis.png
│   └── *.csv
├── output_pract_2/
│   ├── word2vec_3d.png
│   ├── comparative_analysis.png
│   └── *.csv
├── output_pract_3/
│   ├── preprocessing_impact.png
│   └── *.csv
├── output_pract_4/
│   ├── performance_metrics.png
│   └── *.csv
├── output_pract_5/
│   ├── semantic_dashboard.png
│   ├── semantic_network.png
│   └── *.csv
├── output_pract_6/
│   ├── translation_dashboard.png
│   └── *.csv
├── output_pract_7/
│   ├── pipeline_stages.png
│   ├── complexity_metrics.png
│   └── *.csv
├── output_pract_8/
│   ├── wsd_dashboard.png
│   └── *.csv
├── output_pract_9/
│   ├── sentiment_dashboard.png
│   └── *.csv
└── output_pract_10/
    ├── ngram_analysis.png
    └── *.csv
```

---

## Key Features Across All Practicals

### Technical Excellence
- ✅ **Professional code structure** with classes and modular design
- ✅ **Comprehensive documentation** with docstrings
- ✅ **Error handling** and graceful degradation
- ✅ **Performance optimization** where applicable

### Visualization Quality
- ✅ **Multi-panel dashboards** (4-8 panels per practical)
- ✅ **Color-coded visualizations** for clarity
- ✅ **Statistical annotations** on plots
- ✅ **High-resolution outputs** (300 DPI)

### Extra Features That Stand Out
- ✅ **Custom algorithms** beyond basic requirements
- ✅ **Comparative analysis** across multiple approaches
- ✅ **Quality metrics** and confidence scoring
- ✅ **Interactive demonstrations** with real examples

---

## What Makes This Different

### 1. Indian Language Support
- Practical 6: Translation to/from 5+ Indian languages
- Practical 9: Sentiment analysis with Hindi lexicon

### 2. Multi-Algorithm Comparisons
- Practical 5: 3 similarity algorithms
- Practical 8: 4 WSD algorithms
- Practical 9: 3 sentiment models

### 3. Advanced Metrics
- BLEU scores (n-gram precision)
- Perplexity evaluation
- Confidence scoring
- Quality categorization

### 4. Network Visualizations
- Semantic relationship graphs
- Hierarchical taxonomies
- POS transition patterns

### 5. Real-World Applications
- Auto-complete system (Practical 10)
- Machine translation (Practical 6)
- Sentiment analysis (Practical 9)

---

## Troubleshooting

### Common Issues

**Issue**: ModuleNotFoundError
```bash
# Solution: Install missing package
pip install <package-name> --break-system-packages
```

**Issue**: NLTK data not found
```python
# Solution: Download manually
import nltk
nltk.download('all')
```

**Issue**: spaCy model missing
```bash
# Solution: Download model
python -m spacy download en_core_web_sm
```

**Issue**: Google Translate issues
```bash
# Solution: Install specific version
pip install googletrans==4.0.0rc1 --break-system-packages
```

---

## Credits & References

- **NLTK**: Natural Language Toolkit
- **spaCy**: Industrial-strength NLP
- **Gensim**: Topic modeling and Word2Vec
- **TextBlob**: Simplified text processing
- **VADER**: Sentiment analysis
- **WordNet**: Lexical database
- **MarianMT**: Neural machine translation

---

## License

Educational use only. Please cite if you use this code in your work.

---

## Author

Created for NLP coursework with enhanced features for academic excellence.

**Last Updated**: March 2026