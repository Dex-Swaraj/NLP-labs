"""
PRACTICAL 3: Advanced Text Preprocessing with Data Augmentation & Quality Analysis
STANDOUT FEATURES:
- Text quality scoring system
- Data augmentation techniques (synonym replacement, back-translation simulation)
- Before/After preprocessing comparison dashboard
- Feature importance analysis for TF-IDF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
print("Downloading NLTK resources...")
resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 
             'averaged_perceptron_tagger_eng', 'omw-1.4', 'punkt_tab']
for resource in resources:
    nltk.download(resource, quiet=True)

class AdvancedTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = None
        self.quality_metrics = []
        
    def calculate_text_quality(self, text):
        """EXTRA: Calculate quality metrics for text"""
        metrics = {}
        
        # Length metrics
        metrics['char_count'] = len(text)
        metrics['word_count'] = len(text.split())
        metrics['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Lexical diversity
        words = text.lower().split()
        metrics['unique_words'] = len(set(words))
        metrics['lexical_diversity'] = len(set(words)) / len(words) if words else 0
        
        # Special character ratio
        special_chars = sum(1 for c in text if c in string.punctuation)
        metrics['special_char_ratio'] = special_chars / len(text) if text else 0
        
        # Uppercase ratio
        metrics['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Digit ratio
        metrics['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        
        # Readability score (simplified)
        sentences = text.split('.')
        metrics['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        return metrics
    
    def clean_text(self, text):
        """Comprehensive text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers (optional, based on use case)
        # text = re.sub(r'\d+', '', text)
        
        return text
    
    def get_wordnet_pos(self, word):
        """Map POS tag to WordNet POS tag"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def lemmatize_text(self, text):
        """Advanced lemmatization with POS tagging"""
        words = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) 
                     for word in words]
        return ' '.join(lemmatized)
    
    def remove_stopwords(self, text):
        """Remove stopwords"""
        words = text.split()
        filtered = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered)
    
    def augment_text_synonym_replacement(self, text, n=2):
        """EXTRA: Data augmentation using synonym replacement"""
        words = text.split()
        augmented_texts = [text]  # Original text
        
        for _ in range(n):
            new_words = words.copy()
            random_word_list = list(set([word for word in words 
                                        if word not in self.stop_words]))
            
            if not random_word_list:
                continue
                
            # Replace random words with synonyms
            num_replacements = max(1, len(random_word_list) // 5)
            random_words = np.random.choice(random_word_list, 
                                          min(num_replacements, len(random_word_list)), 
                                          replace=False)
            
            for random_word in random_words:
                synonyms = []
                for syn in wordnet.synsets(random_word):
                    for lemma in syn.lemmas():
                        if lemma.name() != random_word:
                            synonyms.append(lemma.name().replace('_', ' '))
                
                if synonyms:
                    synonym = np.random.choice(synonyms)
                    new_words = [synonym if word == random_word else word 
                               for word in new_words]
            
            augmented_texts.append(' '.join(new_words))
        
        return augmented_texts
    
    def preprocess_pipeline(self, texts, augment=False):
        """Complete preprocessing pipeline with quality tracking"""
        print("\nRUNNING PREPROCESSING PIPELINE...")
        
        results = {
            'original': [],
            'cleaned': [],
            'lemmatized': [],
            'stopwords_removed': [],
            'quality_before': [],
            'quality_after': []
        }
        
        all_augmented = []
        
        for i, text in enumerate(texts):
            print(f"  Processing document {i+1}/{len(texts)}...", end='\r')
            
            # Track quality before
            quality_before = self.calculate_text_quality(text)
            results['quality_before'].append(quality_before)
            results['original'].append(text)
            
            # Clean
            cleaned = self.clean_text(text)
            results['cleaned'].append(cleaned)
            
            # Lemmatize
            lemmatized = self.lemmatize_text(cleaned)
            results['lemmatized'].append(lemmatized)
            
            # Remove stopwords
            final = self.remove_stopwords(lemmatized)
            results['stopwords_removed'].append(final)
            
            # Track quality after
            quality_after = self.calculate_text_quality(final)
            results['quality_after'].append(quality_after)
            
            # Augmentation
            if augment:
                augmented = self.augment_text_synonym_replacement(final, n=2)
                all_augmented.extend(augmented)
        
        print("\n   Preprocessing completed!")
        
        if augment:
            results['augmented'] = all_augmented
            print(f"  Generated {len(all_augmented)} augmented texts")
        
        return results
    
    def visualize_preprocessing_impact(self, results, save_path='preprocessing_impact.png'):
        """EXTRA: Comprehensive visualization of preprocessing effects"""
        print("\nCreating preprocessing impact visualization...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Text Preprocessing Impact Analysis', fontweight='bold', fontsize=16)
        
        quality_before_df = pd.DataFrame(results['quality_before'])
        quality_after_df = pd.DataFrame(results['quality_after'])
        
        # 1. Word count comparison
        axes[0, 0].plot(quality_before_df['word_count'], 'o-', label='Before', 
                       linewidth=2, markersize=8)
        axes[0, 0].plot(quality_after_df['word_count'], 's-', label='After', 
                       linewidth=2, markersize=8)
        axes[0, 0].set_title('Word Count: Before vs After', fontweight='bold')
        axes[0, 0].set_xlabel('Document Index')
        axes[0, 0].set_ylabel('Word Count')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Lexical diversity
        axes[0, 1].bar(range(len(quality_before_df)), quality_before_df['lexical_diversity'],
                      alpha=0.6, label='Before', color='skyblue')
        axes[0, 1].bar(range(len(quality_after_df)), quality_after_df['lexical_diversity'],
                      alpha=0.6, label='After', color='coral')
        axes[0, 1].set_title('Lexical Diversity Change', fontweight='bold')
        axes[0, 1].set_xlabel('Document Index')
        axes[0, 1].set_ylabel('Diversity Score')
        axes[0, 1].legend()
        
        # 3. Average word length
        axes[1, 0].boxplot([quality_before_df['avg_word_length'], 
                           quality_after_df['avg_word_length']],
                          labels=['Before', 'After'])
        axes[1, 0].set_title('Average Word Length Distribution', fontweight='bold')
        axes[1, 0].set_ylabel('Average Word Length')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Special character ratio
        axes[1, 1].scatter(quality_before_df['special_char_ratio'], 
                          quality_after_df['special_char_ratio'], 
                          s=100, alpha=0.6, color='green')
        axes[1, 1].plot([0, 0.2], [0, 0.2], 'r--', linewidth=2, label='No change')
        axes[1, 1].set_title('Special Character Ratio Change', fontweight='bold')
        axes[1, 1].set_xlabel('Before')
        axes[1, 1].set_ylabel('After')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # 5. Overall quality improvement
        quality_metrics = ['word_count', 'unique_words', 'lexical_diversity']
        before_avg = [quality_before_df[m].mean() for m in quality_metrics]
        after_avg = [quality_after_df[m].mean() for m in quality_metrics]
        
        x = np.arange(len(quality_metrics))
        width = 0.35
        axes[2, 0].bar(x - width/2, before_avg, width, label='Before', color='lightblue')
        axes[2, 0].bar(x + width/2, after_avg, width, label='After', color='lightcoral')
        axes[2, 0].set_title('Average Quality Metrics', fontweight='bold')
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels(quality_metrics, rotation=15)
        axes[2, 0].legend()
        axes[2, 0].grid(alpha=0.3, axis='y')
        
        # 6. Text length reduction
        char_reduction = [(b - a) / b * 100 
                         for b, a in zip(quality_before_df['char_count'], 
                                       quality_after_df['char_count'])]
        
        axes[2, 1].hist(char_reduction, bins=15, color='purple', 
                       edgecolor='black', alpha=0.7)
        axes[2, 1].axvline(np.mean(char_reduction), color='red', 
                          linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(char_reduction):.1f}%')
        axes[2, 1].set_title('Text Length Reduction Distribution', fontweight='bold')
        axes[2, 1].set_xlabel('Reduction %')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved to {save_path}")
        
        return fig
    
    def create_tfidf_features(self, texts):
        """Create TF-IDF features"""
        print("\nCreating TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names
        )
        
        print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        return tfidf_df
    
    def feature_importance_analysis(self, tfidf_df, save_path='feature_importance.png'):
        """EXTRA: Analyze feature importance in TF-IDF"""
        print("\n Analyzing feature importance...")
        
        # Calculate average TF-IDF scores
        avg_scores = tfidf_df.mean().sort_values(ascending=False)
        top_features = avg_scores.head(20)
        
        # Calculate variance (features with high variance are discriminative)
        variance_scores = tfidf_df.var().sort_values(ascending=False)
        top_variance = variance_scores.head(20)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TF-IDF Feature Importance Analysis', fontweight='bold', fontsize=16)
        
        # 1. Top features by average score
        axes[0, 0].barh(range(len(top_features)), top_features.values, color='skyblue')
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features.index, fontsize=9)
        axes[0, 0].set_title('Top 20 Features by Average TF-IDF', fontweight='bold')
        axes[0, 0].set_xlabel('Average TF-IDF Score')
        axes[0, 0].invert_yaxis()
        
        # 2. Top features by variance
        axes[0, 1].barh(range(len(top_variance)), top_variance.values, color='coral')
        axes[0, 1].set_yticks(range(len(top_variance)))
        axes[0, 1].set_yticklabels(top_variance.index, fontsize=9)
        axes[0, 1].set_title('Top 20 Features by Variance', fontweight='bold')
        axes[0, 1].set_xlabel('Variance')
        axes[0, 1].invert_yaxis()
        
        # 3. Feature correlation heatmap (top 15 features)
        top_15_features = avg_scores.head(15).index
        correlation_matrix = tfidf_df[top_15_features].corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
        axes[1, 0].set_title('Feature Correlation Matrix', fontweight='bold')
        
        # 4. TF-IDF distribution
        all_scores = tfidf_df.values.flatten()
        all_scores = all_scores[all_scores > 0]  # Remove zeros
        
        axes[1, 1].hist(all_scores, bins=50, color='green', 
                       edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('TF-IDF Score Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('TF-IDF Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_yscale('log')
        axes[1, 1].axvline(np.median(all_scores), color='red', 
                          linestyle='--', linewidth=2, 
                          label=f'Median: {np.median(all_scores):.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Feature importance analysis saved to {save_path}")
        
        return top_features, top_variance

def main():
    print("="*80)
    print("PRACTICAL 3: ADVANCED TEXT PREPROCESSING & ANALYSIS")
    print("="*80)
    
    # Sample dataset with various text characteristics
    texts = [
        "Machine Learning is AMAZING! Check out this article: http://example.com #AI #ML",
        "Natural Language Processing helps computers understand human language. Email: info@example.com",
        "Deep learning uses neural networks with MULTIPLE layers... It's revolutionary!!!",
        "Data Science combines statistics, programming & domain knowledge for insights.",
        "Python is the most popular language for AI development. Visit www.python.org",
        "The artificial intelligence revolution is transforming industries worldwide!",
        "Neural networks learn patterns from data through backpropagation algorithms.",
        "Big Data analytics requires distributed computing frameworks like Hadoop."
    ]
    
    # Labels for the texts
    labels = ['ML', 'NLP', 'DL', 'DS', 'Programming', 'AI', 'DL', 'BigData']
    
    preprocessor = AdvancedTextPreprocessor()
    
    print(f"\n Dataset: {len(texts)} documents")
    print(f"Categories: {len(set(labels))} unique labels")
    
    # Show original samples
    print("\n" + "="*80)
    print("ORIGINAL TEXTS (First 3):")
    print("="*80)
    for i, text in enumerate(texts[:3], 1):
        print(f"\nDoc {i}: {text}")
        quality = preprocessor.calculate_text_quality(text)
        print(f"  Quality: {quality['word_count']} words, "
              f"{quality['lexical_diversity']:.2f} diversity, "
              f"{quality['special_char_ratio']:.2%} special chars")
    
    # Preprocessing pipeline
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80)
    results = preprocessor.preprocess_pipeline(texts, augment=True)
    
    # Show processed samples
    print("\n" + "="*80)
    print("PROCESSED TEXTS (First 3):")
    print("="*80)
    for i in range(3):
        print(f"\nDoc {i+1}:")
        print(f"  Original:  {results['original'][i]}")
        print(f"  Cleaned:   {results['cleaned'][i]}")
        print(f"  Lemmatized: {results['lemmatized'][i]}")
        print(f"  Final:     {results['stopwords_removed'][i]}")
    
    # Visualize preprocessing impact
    preprocessor.visualize_preprocessing_impact(results)
    
    # Label encoding
    print("\n" + "="*80)
    print("LABEL ENCODING")
    print("="*80)
    encoded_labels = preprocessor.label_encoder.fit_transform(labels)
    
    label_mapping = dict(zip(preprocessor.label_encoder.classes_, 
                            range(len(preprocessor.label_encoder.classes_))))
    print("\nLabel Mapping:")
    for label, code in label_mapping.items():
        print(f"  {label:15s} -> {code}")
    
    # Save label encoder
    label_df = pd.DataFrame({
        'Original_Label': labels,
        'Encoded_Label': encoded_labels
    })
    label_df.to_csv('label_encoding.csv', index=False)
    
    # TF-IDF features
    print("\n" + "="*80)
    print("TF-IDF FEATURE EXTRACTION")
    print("="*80)
    tfidf_df = preprocessor.create_tfidf_features(results['stopwords_removed'])
    
    print("\nTF-IDF Matrix (First 3 docs, top 10 features):")
    print(tfidf_df.iloc[:3, :10].to_string())
    
    # Feature importance analysis
    top_features, top_variance = preprocessor.feature_importance_analysis(tfidf_df)
    
    print("\n\n📊 TOP FEATURES BY IMPORTANCE:")
    print(f"\nBy Average Score:")
    for feat, score in top_features.head(10).items():
        print(f"  {feat:20s}: {score:.4f}")
    
    print(f"\nBy Variance (Discriminative Power):")
    for feat, var in top_variance.head(10).items():
        print(f"  {feat:20s}: {var:.4f}")
    
    # Save all outputs
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save preprocessed texts
    output_df = pd.DataFrame({
        'original': results['original'],
        'cleaned': results['cleaned'],
        'lemmatized': results['lemmatized'],
        'final': results['stopwords_removed'],
        'label': labels,
        'encoded_label': encoded_labels
    })
    output_df.to_csv('preprocessed_texts.csv', index=False)
    print("  Saved: preprocessed_texts.csv")
    
    # Save augmented texts
    if 'augmented' in results:
        augmented_df = pd.DataFrame({'augmented_text': results['augmented']})
        augmented_df.to_csv('augmented_texts.csv', index=False)
        print("  Saved: augmented_texts.csv")
    
    # Save TF-IDF features
    tfidf_df.to_csv('tfidf_features.csv', index=False)
    print("  Saved: tfidf_features.csv")
    
    # Save quality metrics
    quality_df = pd.DataFrame({
        'doc_id': range(len(results['quality_before'])),
        **{f'before_{k}': [d[k] for d in results['quality_before']] 
           for k in results['quality_before'][0].keys()},
        **{f'after_{k}': [d[k] for d in results['quality_after']] 
           for k in results['quality_after'][0].keys()}
    })
    quality_df.to_csv('quality_metrics.csv', index=False)
    print("  Saved: quality_metrics.csv")
    
    print("\n" + "="*80)
    print("✓ PRACTICAL 3 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  preprocessed_texts.csv (Complete preprocessing pipeline)")
    print("  augmented_texts.csv (Data augmentation results)")
    print("  tfidf_features.csv (TF-IDF representation)")
    print("  label_encoding.csv (Encoded labels)")
    print("  quality_metrics.csv (Before/after quality metrics)")
    print("  preprocessing_impact.png (Comprehensive dashboard)")
    print("  feature_importance.png (Feature analysis)")

if __name__ == "__main__":
    main()
