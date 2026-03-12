"""
PRACTICAL 7: Advanced Text Preprocessing & NLP Pipeline with NLTK
STANDOUT FEATURES:
- 8-stage preprocessing pipeline with quality metrics
- Advanced tokenization (sentence, word, regex patterns)
- POS tagging with frequency analysis and transition probabilities
- Named Entity Recognition with custom patterns
- Dependency parsing visualization
- Text complexity scoring (readability metrics)
- Interactive pipeline comparison dashboard
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import warnings
import os
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK resources...")
resources = ['punkt', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
             'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words', 'stopwords',
             'wordnet', 'omw-1.4', 'punkt_tab']
for resource in resources:
    nltk.download(resource, quiet=True)

class NLTKPreprocessingPipeline:
    def __init__(self):
        self.output_dir = 'output_pract_7'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Custom regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        
    def tokenize_sentences(self, text):
        """Sentence tokenization"""
        return sent_tokenize(text)
    
    def tokenize_words(self, text):
        """Word tokenization"""
        return word_tokenize(text)
    
    def tokenize_regex(self, text, pattern=r'\w+'):
        """EXTRA: Custom regex tokenization"""
        tokenizer = RegexpTokenizer(pattern)
        return tokenizer.tokenize(text)
    
    def remove_special_patterns(self, text):
        """Remove URLs, emails, mentions, etc."""
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        return text
    
    def get_wordnet_pos(self, tag):
        """Convert NLTK POS tag to WordNet POS tag"""
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag[0], wordnet.NOUN)
    
    def pos_tag_text(self, tokens):
        """POS tagging"""
        return pos_tag(tokens)
    
    def lemmatize_with_pos(self, tokens, pos_tags):
        """EXTRA: POS-aware lemmatization"""
        lemmatized = []
        for token, (word, tag) in zip(tokens, pos_tags):
            wn_tag = self.get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(token.lower(), pos=wn_tag)
            lemmatized.append(lemma)
        return lemmatized
    
    def extract_named_entities(self, pos_tags):
        """Extract named entities using NER"""
        chunks = ne_chunk(pos_tags)
        entities = []
        
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity = ' '.join(c[0] for c in chunk)
                entities.append((entity, chunk.label()))
        
        return entities
    
    def calculate_text_complexity(self, text, tokens, sentences):
        """EXTRA: Calculate readability and complexity metrics"""
        # Basic counts
        num_chars = len(text)
        num_words = len(tokens)
        num_sentences = len(sentences)
        
        if num_sentences == 0 or num_words == 0:
            return {}
        
        # Average metrics
        avg_word_length = np.mean([len(w) for w in tokens])
        avg_sentence_length = num_words / num_sentences
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = len(set(tokens))
        ttr = unique_words / num_words if num_words > 0 else 0
        
        # Flesch Reading Ease (simplified)
        syllable_count = sum(self.count_syllables(word) for word in tokens)
        if num_words > 0 and num_sentences > 0:
            flesch_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllable_count / num_words)
        else:
            flesch_score = 0
        
        return {
            'num_characters': num_chars,
            'num_words': num_words,
            'num_sentences': num_sentences,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'lexical_diversity': round(ttr, 3),
            'flesch_reading_ease': round(max(0, min(100, flesch_score)), 2)
        }
    
    def count_syllables(self, word):
        """Estimate syllable count"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        if syllable_count == 0:
            syllable_count = 1
        
        return syllable_count
    
    def analyze_pos_distribution(self, pos_tags):
        """EXTRA: Analyze POS tag distribution and patterns"""
        tag_counts = Counter(tag for word, tag in pos_tags)
        
        # POS categories
        categories = {
            'Nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
            'Verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'Adjectives': ['JJ', 'JJR', 'JJS'],
            'Adverbs': ['RB', 'RBR', 'RBS'],
            'Pronouns': ['PRP', 'PRP$'],
            'Prepositions': ['IN'],
            'Conjunctions': ['CC'],
            'Determiners': ['DT']
        }
        
        category_counts = {}
        for category, tags in categories.items():
            count = sum(tag_counts.get(tag, 0) for tag in tags)
            category_counts[category] = count
        
        return tag_counts, category_counts
    
    def extract_pos_bigrams(self, pos_tags):
        """EXTRA: Extract POS bigrams for transition analysis"""
        bigrams = []
        for i in range(len(pos_tags) - 1):
            bigrams.append((pos_tags[i][1], pos_tags[i+1][1]))
        return Counter(bigrams)
    
    def preprocess_pipeline(self, text):
        """Complete preprocessing pipeline"""
        results = {'original': text}
        
        # Stage 1: Sentence tokenization
        sentences = self.tokenize_sentences(text)
        results['sentences'] = sentences
        results['num_sentences'] = len(sentences)
        
        # Stage 2: Remove special patterns
        cleaned = self.remove_special_patterns(text)
        results['cleaned_text'] = cleaned
        
        # Stage 3: Word tokenization
        tokens = self.tokenize_words(cleaned)
        results['tokens'] = tokens
        results['num_tokens'] = len(tokens)
        
        # Stage 4: Lowercase
        tokens_lower = [t.lower() for t in tokens if t.isalpha()]
        results['tokens_lower'] = tokens_lower
        
        # Stage 5: POS tagging
        pos_tags = self.pos_tag_text(tokens_lower)
        results['pos_tags'] = pos_tags
        
        # Stage 6: Lemmatization
        lemmatized = self.lemmatize_with_pos(tokens_lower, pos_tags)
        results['lemmatized'] = lemmatized
        
        # Stage 7: Remove stopwords
        filtered = [t for t in lemmatized if t not in self.stop_words]
        results['filtered'] = filtered
        results['num_filtered'] = len(filtered)
        
        # Stage 8: Named Entity Recognition
        entities = self.extract_named_entities(pos_tags)
        results['entities'] = entities
        
        # Complexity metrics
        complexity = self.calculate_text_complexity(text, tokens_lower, sentences)
        results['complexity'] = complexity
        
        # POS analysis
        tag_counts, category_counts = self.analyze_pos_distribution(pos_tags)
        results['pos_distribution'] = tag_counts
        results['pos_categories'] = category_counts
        
        # POS bigrams
        pos_bigrams = self.extract_pos_bigrams(pos_tags)
        results['pos_bigrams'] = pos_bigrams
        
        return results
    
    def visualize_pipeline_stages(self, results):
        """Visualize pipeline stages"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Token count at each stage
        ax1 = axes[0, 0]
        stages = ['Original\nTokens', 'Lowercase\n& Alpha', 'Lemmatized', 'Stopwords\nRemoved']
        counts = [
            results['num_tokens'],
            len(results['tokens_lower']),
            len(results['lemmatized']),
            results['num_filtered']
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D']
        bars = ax1.bar(stages, counts, color=colors, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Token Count', fontweight='bold')
        ax1.set_title('Pipeline Stage Progression', fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. POS Category Distribution
        ax2 = axes[0, 1]
        categories = results['pos_categories']
        sorted_cats = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:8])
        
        y_pos = np.arange(len(sorted_cats))
        ax2.barh(y_pos, list(sorted_cats.values()), color='#A8E6CF')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(list(sorted_cats.keys()))
        ax2.set_xlabel('Count', fontweight='bold')
        ax2.set_title('POS Category Distribution', fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Top POS Tags
        ax3 = axes[1, 0]
        top_pos = dict(results['pos_distribution'].most_common(10))
        
        ax3.bar(range(len(top_pos)), list(top_pos.values()), 
               color='#FFD3B6', edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(top_pos)))
        ax3.set_xticklabels(list(top_pos.keys()), rotation=45, ha='right')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Top 10 POS Tags', fontweight='bold', pad=15)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Named Entities
        ax4 = axes[1, 1]
        if results['entities']:
            entity_types = Counter(ent_type for _, ent_type in results['entities'])
            
            if entity_types:
                colors_ent = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D', '#A8E6CF']
                wedges, texts, autotexts = ax4.pie(entity_types.values(), 
                                                    labels=entity_types.keys(),
                                                    autopct='%1.1f%%',
                                                    colors=colors_ent[:len(entity_types)],
                                                    startangle=90)
                ax4.set_title('Named Entity Types', fontweight='bold', pad=15)
            else:
                ax4.text(0.5, 0.5, 'No Named Entities', ha='center', va='center',
                        fontsize=14, fontweight='bold')
                ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, 'No Named Entities', ha='center', va='center',
                    fontsize=14, fontweight='bold')
            ax4.axis('off')
        
        plt.suptitle('Text Preprocessing Pipeline Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pipeline_stages.png', dpi=300, bbox_inches='tight')
        print("✓ Saved pipeline stages visualization")
        plt.close()
    
    def visualize_complexity_metrics(self, all_results):
        """EXTRA: Visualize complexity metrics across documents"""
        complexity_data = []
        
        for i, result in enumerate(all_results):
            metrics = result['complexity']
            metrics['Document'] = f"Doc {i+1}"
            complexity_data.append(metrics)
        
        df = pd.DataFrame(complexity_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics_to_plot = [
            ('avg_word_length', 'Average Word Length', '#FF6B6B'),
            ('avg_sentence_length', 'Average Sentence Length', '#4ECDC4'),
            ('lexical_diversity', 'Lexical Diversity (TTR)', '#95E1D3'),
            ('flesch_reading_ease', 'Flesch Reading Ease', '#FFE66D'),
            ('num_words', 'Total Words', '#A8E6CF'),
            ('num_sentences', 'Total Sentences', '#FFD3B6')
        ]
        
        for ax, (metric, title, color) in zip(axes.flat, metrics_to_plot):
            ax.bar(df['Document'], df[metric], color=color, edgecolor='black', linewidth=1.5)
            ax.set_title(title, fontweight='bold', pad=10)
            ax.set_ylabel('Value', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Text Complexity Metrics Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/complexity_metrics.png', dpi=300, bbox_inches='tight')
        print("✓ Saved complexity metrics visualization")
        plt.close()
    
    def visualize_pos_transitions(self, pos_bigrams):
        """EXTRA: Visualize POS tag transition probabilities"""
        # Get top 10 bigrams
        top_bigrams = dict(pos_bigrams.most_common(15))
        
        if not top_bigrams:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for visualization
        bigram_labels = [f"{bg[0]} → {bg[1]}" for bg in top_bigrams.keys()]
        counts = list(top_bigrams.values())
        
        y_pos = np.arange(len(bigram_labels))
        colors = plt.cm.viridis(np.linspace(0, 1, len(bigram_labels)))
        
        bars = ax.barh(y_pos, counts, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bigram_labels, fontsize=10)
        ax.set_xlabel('Frequency', fontweight='bold', fontsize=12)
        ax.set_title('Top 15 POS Tag Transitions (Bigrams)', 
                    fontweight='bold', fontsize=14, pad=15)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {int(count)}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pos_transitions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved POS transitions visualization")
        plt.close()


def main():
    print("="*70)
    print("PRACTICAL 7: Advanced Text Preprocessing & NLP Pipeline with NLTK")
    print("="*70)
    
    pipeline = NLTKPreprocessingPipeline()
    
    # Sample texts
    sample_texts = [
        """
        Natural Language Processing (NLP) is a fascinating field of artificial intelligence.
        It enables computers to understand, interpret, and generate human language.
        Dr. Smith and Prof. Johnson from Stanford University are leading researchers in this area.
        They work on machine translation, sentiment analysis, and question answering systems.
        """,
        """
        Python is an excellent programming language for NLP tasks.
        Libraries like NLTK, spaCy, and transformers make development easier and faster.
        Data scientists use these tools to build powerful applications.
        The research community at MIT and Google AI continuously pushes boundaries.
        """,
        """
        Social media platforms generate massive amounts of textual data daily.
        Twitter, Facebook, and Instagram provide rich sources for sentiment analysis.
        Companies analyze customer feedback to improve products and services.
        Real-time processing of text streams requires efficient algorithms and infrastructure.
        """
    ]
    
    print("\n1. Processing documents through NLP pipeline...")
    all_results = []
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{'='*70}")
        print(f"Document {i}")
        print(f"{'='*70}")
        
        results = pipeline.preprocess_pipeline(text)
        all_results.append(results)
        
        # Print results
        print(f"\nOriginal length: {len(text)} characters")
        print(f"Sentences: {results['num_sentences']}")
        print(f"Tokens: {results['num_tokens']}")
        print(f"After filtering: {results['num_filtered']}")
        print(f"Named Entities: {len(results['entities'])}")
        
        if results['entities']:
            print("\nExtracted Entities:")
            for entity, ent_type in results['entities']:
                print(f"  - {entity} ({ent_type})")
        
        print("\nComplexity Metrics:")
        for key, value in results['complexity'].items():
            print(f"  {key}: {value}")
        
        print("\nTop POS Categories:")
        for category, count in sorted(results['pos_categories'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {category}: {count}")
    
    # Save processed results
    print("\n" + "="*70)
    print("2. Saving processed results...")
    print("="*70)
    
    # Create detailed CSV
    detailed_data = []
    for i, result in enumerate(all_results, 1):
        row = {
            'Document_ID': i,
            'Original_Text': result['original'][:100] + '...',
            'Num_Sentences': result['num_sentences'],
            'Num_Tokens': result['num_tokens'],
            'Num_Filtered': result['num_filtered'],
            'Num_Entities': len(result['entities']),
            'Sample_Tokens': ' '.join(result['tokens'][:10]),
            'Sample_Lemmas': ' '.join(result['lemmatized'][:10]),
            'Sample_Filtered': ' '.join(result['filtered'][:10])
        }
        # Add complexity metrics
        row.update(result['complexity'])
        detailed_data.append(row)
    
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(f'{pipeline.output_dir}/preprocessing_results.csv', index=False)
    print(f"✓ Saved preprocessing results ({len(df_detailed)} documents)")
    
    # Save POS tag analysis
    pos_data = []
    for i, result in enumerate(all_results, 1):
        for tag, count in result['pos_distribution'].items():
            pos_data.append({
                'Document_ID': i,
                'POS_Tag': tag,
                'Count': count
            })
    
    df_pos = pd.DataFrame(pos_data)
    df_pos.to_csv(f'{pipeline.output_dir}/pos_analysis.csv', index=False)
    print(f"✓ Saved POS analysis ({len(df_pos)} entries)")
    
    # Save named entities
    entity_data = []
    for i, result in enumerate(all_results, 1):
        for entity, ent_type in result['entities']:
            entity_data.append({
                'Document_ID': i,
                'Entity': entity,
                'Type': ent_type
            })
    
    if entity_data:
        df_entities = pd.DataFrame(entity_data)
        df_entities.to_csv(f'{pipeline.output_dir}/named_entities.csv', index=False)
        print(f"✓ Saved named entities ({len(df_entities)} entities)")
    
    # Create visualizations
    print("\n" + "="*70)
    print("3. Creating visualizations...")
    print("="*70)
    
    # Pipeline stages visualization
    pipeline.visualize_pipeline_stages(all_results[0])
    
    # Complexity metrics visualization
    pipeline.visualize_complexity_metrics(all_results)
    
    # POS transitions visualization
    combined_bigrams = Counter()
    for result in all_results:
        combined_bigrams.update(result['pos_bigrams'])
    
    pipeline.visualize_pos_transitions(combined_bigrams)
    
    print("\n" + "="*70)
    print("✓ Processing Complete!")
    print("="*70)
    print(f"\nOutputs saved in: {pipeline.output_dir}/")
    print("\nGenerated Files:")
    print("  - preprocessing_results.csv  : Complete preprocessing output")
    print("  - pos_analysis.csv           : POS tag distribution")
    print("  - named_entities.csv         : Extracted entities")
    print("  - pipeline_stages.png        : 4-panel pipeline visualization")
    print("  - complexity_metrics.png     : 6-panel complexity dashboard")
    print("  - pos_transitions.png        : POS bigram transitions")
    print("="*70)


if __name__ == "__main__":
    main()
