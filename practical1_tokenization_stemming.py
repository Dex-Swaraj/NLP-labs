"""
PRACTICAL 1: Advanced Tokenization & Stemming with Interactive Visual Analytics
STANDOUT FEATURES:
- Custom regex tokenizer for handling emojis and hashtags
- Comparative analysis dashboard of all tokenizers
- Interactive visualization of stemming effects
- Performance benchmarking
"""

import nltk
from nltk.tokenize import (
    word_tokenize, 
    wordpunct_tokenize, 
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)
from nltk.stem import PorterStemmer, SnowballStemmer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class AdvancedTokenizationSystem:
    def __init__(self):
        self.porter = PorterStemmer()
        self.snowball = SnowballStemmer('english')
        self.treebank = TreebankWordTokenizer()
        self.tweet_tokenizer = TweetTokenizer()
        self.mwe_tokenizer = MWETokenizer([
            ('artificial', 'intelligence'),
            ('machine', 'learning'),
            ('natural', 'language'),
            ('deep', 'learning'),
            ('data', 'science')
        ])
        
    def custom_emoji_tokenizer(self, text):
        """EXTRA: Custom tokenizer that preserves emojis and special characters"""
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        
        # Extract emojis
        emojis = emoji_pattern.findall(text)
        
        # Remove emojis temporarily
        text_no_emoji = emoji_pattern.sub(' EMOJI_PLACEHOLDER ', text)
        
        # Tokenize the rest
        tokens = re.findall(r'\b\w+\b|[#@]\w+|[^\w\s]', text_no_emoji)
        
        # Reinsert emojis
        result = []
        emoji_idx = 0
        for token in tokens:
            if token == 'EMOJI_PLACEHOLDER':
                if emoji_idx < len(emojis):
                    result.append(emojis[emoji_idx])
                    emoji_idx += 1
            else:
                result.append(token)
        
        return result
    
    def perform_all_tokenizations(self, text):
        """Perform all types of tokenization"""
        results = {}
        
        # 1. Whitespace tokenization
        start = time.time()
        results['Whitespace'] = text.split()
        results['Whitespace_time'] = time.time() - start
        
        # 2. Punctuation-based tokenization
        start = time.time()
        results['Punctuation'] = wordpunct_tokenize(text)
        results['Punctuation_time'] = time.time() - start
        
        # 3. Treebank tokenization
        start = time.time()
        results['Treebank'] = self.treebank.tokenize(text)
        results['Treebank_time'] = time.time() - start
        
        # 4. Tweet tokenization
        start = time.time()
        results['Tweet'] = self.tweet_tokenizer.tokenize(text)
        results['Tweet_time'] = time.time() - start
        
        # 5. MWE tokenization
        start = time.time()
        tokens = word_tokenize(text.lower())
        results['MWE'] = self.mwe_tokenizer.tokenize(tokens)
        results['MWE_time'] = time.time() - start
        
        # 6. EXTRA: Custom emoji tokenization
        start = time.time()
        results['Custom_Emoji'] = self.custom_emoji_tokenizer(text)
        results['Custom_Emoji_time'] = time.time() - start
        
        return results
    
    def compare_stemmers(self, words):
        """Compare Porter and Snowball stemmers with analysis"""
        comparison = []
        differences = []
        
        for word in words:
            porter_stem = self.porter.stem(word)
            snowball_stem = self.snowball.stem(word)
            
            comparison.append({
                'Original': word,
                'Porter': porter_stem,
                'Snowball': snowball_stem,
                'Same': porter_stem == snowball_stem
            })
            
            if porter_stem != snowball_stem:
                differences.append({
                    'Word': word,
                    'Porter': porter_stem,
                    'Snowball': snowball_stem
                })
        
        return pd.DataFrame(comparison), pd.DataFrame(differences)
    
    def visualize_tokenization_comparison(self, results, save_path='tokenization_comparison.png'):
        """EXTRA: Create comprehensive visualization of tokenization results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Tokenization Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Token count comparison
        tokenizer_names = ['Whitespace', 'Punctuation', 'Treebank', 'Tweet', 'MWE', 'Custom_Emoji']
        token_counts = [len(results[name]) for name in tokenizer_names]
        
        axes[0, 0].bar(tokenizer_names, token_counts, color=sns.color_palette('husl', 6))
        axes[0, 0].set_title('Token Count by Tokenizer', fontweight='bold')
        axes[0, 0].set_xlabel('Tokenizer Type')
        axes[0, 0].set_ylabel('Number of Tokens')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(token_counts):
            axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # 2. Performance comparison (time taken)
        times = [results[f'{name}_time'] * 1000 for name in tokenizer_names]  # Convert to ms
        axes[0, 1].barh(tokenizer_names, times, color=sns.color_palette('rocket', 6))
        axes[0, 1].set_title('Tokenization Performance (milliseconds)', fontweight='bold')
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Tokenizer Type')
        for i, v in enumerate(times):
            axes[0, 1].text(v + 0.01, i, f'{v:.3f}ms', va='center', fontweight='bold')
        
        # 3. Unique tokens comparison
        unique_counts = [len(set(results[name])) for name in tokenizer_names]
        axes[1, 0].plot(tokenizer_names, unique_counts, marker='o', linewidth=2, 
                       markersize=10, color='darkblue')
        axes[1, 0].fill_between(range(len(tokenizer_names)), unique_counts, alpha=0.3)
        axes[1, 0].set_title('Unique Tokens by Tokenizer', fontweight='bold')
        axes[1, 0].set_xlabel('Tokenizer Type')
        axes[1, 0].set_ylabel('Number of Unique Tokens')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Token length distribution for best tokenizer
        best_tokens = results['Tweet']  # Tweet tokenizer usually best for social media
        token_lengths = [len(token) for token in best_tokens]
        axes[1, 1].hist(token_lengths, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Token Length Distribution (Tweet Tokenizer)', fontweight='bold')
        axes[1, 1].set_xlabel('Token Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(sum(token_lengths)/len(token_lengths), color='red', 
                          linestyle='--', linewidth=2, label=f'Mean: {sum(token_lengths)/len(token_lengths):.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        return fig
    
    def analyze_stemming_patterns(self, words, save_path='stemming_analysis.png'):
        """EXTRA: Deep analysis of stemming patterns and differences"""
        porter_stems = [self.porter.stem(w) for w in words]
        snowball_stems = [self.snowball.stem(w) for w in words]
        
        # Calculate compression ratios
        porter_compression = [(len(w) - len(s)) / len(w) * 100 
                             for w, s in zip(words, porter_stems)]
        snowball_compression = [(len(w) - len(s)) / len(w) * 100 
                               for w, s in zip(words, snowball_stems)]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Advanced Stemming Analysis', fontsize=16, fontweight='bold')
        
        # 1. Compression comparison
        axes[0, 0].scatter(porter_compression, snowball_compression, alpha=0.6, s=100)
        axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Equal compression')
        axes[0, 0].set_title('Stemmer Compression Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Porter Compression %')
        axes[0, 0].set_ylabel('Snowball Compression %')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Word length vs stem length
        word_lengths = [len(w) for w in words]
        porter_stem_lengths = [len(s) for s in porter_stems]
        snowball_stem_lengths = [len(s) for s in snowball_stems]
        
        axes[0, 1].scatter(word_lengths, porter_stem_lengths, alpha=0.6, 
                          label='Porter', s=80, color='blue')
        axes[0, 1].scatter(word_lengths, snowball_stem_lengths, alpha=0.6, 
                          label='Snowball', s=80, color='orange')
        axes[0, 1].plot([0, max(word_lengths)], [0, max(word_lengths)], 
                       'r--', linewidth=2, label='No change')
        axes[0, 1].set_title('Word Length vs Stem Length', fontweight='bold')
        axes[0, 1].set_xlabel('Original Word Length')
        axes[0, 1].set_ylabel('Stem Length')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Difference frequency
        differences = sum([1 for p, s in zip(porter_stems, snowball_stems) if p != s])
        same = len(words) - differences
        
        axes[1, 0].pie([same, differences], labels=['Same Result', 'Different Result'],
                      autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'],
                      startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1, 0].set_title('Porter vs Snowball Agreement', fontweight='bold')
        
        # 4. Average compression by word length category
        length_categories = ['Short (≤4)', 'Medium (5-7)', 'Long (8-10)', 'Very Long (>10)']
        porter_avg = []
        snowball_avg = []
        
        for min_len, max_len in [(0, 4), (5, 7), (8, 10), (11, 100)]:
            p_comp = [c for w, c in zip(words, porter_compression) 
                     if min_len <= len(w) <= max_len]
            s_comp = [c for w, c in zip(words, snowball_compression) 
                     if min_len <= len(w) <= max_len]
            porter_avg.append(sum(p_comp)/len(p_comp) if p_comp else 0)
            snowball_avg.append(sum(s_comp)/len(s_comp) if s_comp else 0)
        
        x = range(len(length_categories))
        width = 0.35
        axes[1, 1].bar([i - width/2 for i in x], porter_avg, width, 
                      label='Porter', color='skyblue')
        axes[1, 1].bar([i + width/2 for i in x], snowball_avg, width, 
                      label='Snowball', color='salmon')
        axes[1, 1].set_title('Avg Compression by Word Length Category', fontweight='bold')
        axes[1, 1].set_xlabel('Word Length Category')
        axes[1, 1].set_ylabel('Compression %')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(length_categories, rotation=15)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Stemming analysis saved to {save_path}")
        return fig

def main():
    print("="*70)
    print("PRACTICAL 1: ADVANCED TOKENIZATION & STEMMING ANALYSIS")
    print("="*70)
    
    # Sample texts including social media content with emojis and hashtags
    texts = [
        "Machine learning and artificial intelligence are revolutionizing data science! 🚀 #AI #MachineLearning",
        "Natural language processing helps computers understand human languages. Check out @OpenAI's latest research!",
        "Running, runner, ran - these words share the same root. Stemming normalizes them effectively.",
        "The quick brown fox jumps over the lazy dog. It's a classic pangram sentence! 😊"
    ]
    
    system = AdvancedTokenizationSystem()
    
    # Process each text
    all_results = []
    for i, text in enumerate(texts, 1):
        print(f"\n{'='*70}")
        print(f"TEXT {i}: {text}")
        print(f"{'='*70}")
        
        # Perform all tokenizations
        results = system.perform_all_tokenizations(text)
        
        # Display results
        print("\nTOKENIZATION RESULTS:")
        print("-" * 70)
        for tokenizer in ['Whitespace', 'Punctuation', 'Treebank', 'Tweet', 'MWE', 'Custom_Emoji']:
            tokens = results[tokenizer]
            time_taken = results[f'{tokenizer}_time'] * 1000
            print(f"\n{tokenizer:20s}: {len(tokens):3d} tokens ({time_taken:.3f}ms)")
            print(f"  Sample: {tokens[:10]}")
        
        all_results.append(results)
    
    # Create comprehensive visualization for first text
    print("\n\nCREATING ADVANCED VISUALIZATIONS...")
    system.visualize_tokenization_comparison(all_results[0])
    
    # Stemming analysis
    print("\n\nSTEMMING ANALYSIS:")
    print("="*70)
    
    test_words = [
        'running', 'runner', 'ran', 'runs',
        'computing', 'computer', 'computed', 'computation',
        'studying', 'student', 'studies', 'studious',
        'connecting', 'connection', 'connected', 'connectivity',
        'organizing', 'organization', 'organized', 'organizer'
    ]
    
    comparison_df, differences_df = system.compare_stemmers(test_words)
    
    print("\nSTEMMER COMPARISON TABLE:")
    print(comparison_df.to_string(index=False))
    
    print("\n\nWORDS WITH DIFFERENT STEMS:")
    if len(differences_df) > 0:
        print(differences_df.to_string(index=False))
    else:
        print("All words produced identical stems!")
    
    # Save results
    comparison_df.to_csv('stemmer_comparison.csv', index=False)
    differences_df.to_csv('stemmer_differences.csv', index=False)
    print("\nResults saved to CSV files")
    
    # Advanced stemming analysis
    system.analyze_stemming_patterns(test_words)
    
    # Summary statistics
    print("\n\nSUMMARY STATISTICS:")
    print("="*70)
    print(f"Total test words: {len(test_words)}")
    print(f"Words with identical stems: {sum(comparison_df['Same'])}")
    print(f"Words with different stems: {len(differences_df)}")
    print(f"Agreement rate: {sum(comparison_df['Same'])/len(comparison_df)*100:.1f}%")
    
    print("\n" + "="*70)
    print("PRACTICAL 1 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  tokenization_comparison.png - Interactive dashboard")
    print("  stemming_analysis.png - Deep stemming insights")
    print("  stemmer_comparison.csv - Detailed comparison")
    print("  stemmer_differences.csv - Difference analysis")

if __name__ == "__main__":
    main()
