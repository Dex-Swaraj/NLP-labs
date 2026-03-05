"""
PRACTICAL 5: WordNet-Based Semantic Relationship Analysis with Network Visualization
STANDOUT FEATURES:
- Interactive semantic network graph (synonyms, antonyms, hypernyms, hyponyms)
- Semantic similarity scoring (path, Wu-Palmer, Leacock-Chodorow)
- Automated relationship discovery for word sets
- Hierarchical taxonomy visualization
- Word relationship matrix with heatmaps
"""

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

class WordNetSemanticAnalyzer:
    def __init__(self):
        self.relationships = defaultdict(list)
        self.output_dir = 'output_pract_5'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_all_relationships(self, word, pos=None):
        """Get comprehensive semantic relationships for a word"""
        synsets = wn.synsets(word, pos=pos)
        
        if not synsets:
            return None
        
        results = {
            'word': word,
            'synsets': [],
            'synonyms': set(),
            'antonyms': set(),
            'hypernyms': set(),
            'hyponyms': set(),
            'meronyms': set(),
            'holonyms': set()
        }
        
        for synset in synsets:
            results['synsets'].append({
                'name': synset.name(),
                'definition': synset.definition(),
                'examples': synset.examples(),
                'pos': synset.pos()
            })
            
            # Synonyms (lemmas from same synset)
            for lemma in synset.lemmas():
                if lemma.name().lower() != word.lower():
                    results['synonyms'].add(lemma.name().replace('_', ' '))
                
                # Antonyms
                for antonym in lemma.antonyms():
                    results['antonyms'].add(antonym.name().replace('_', ' '))
            
            # Hypernyms (more general terms)
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    results['hypernyms'].add(lemma.name().replace('_', ' '))
            
            # Hyponyms (more specific terms)
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    results['hyponyms'].add(lemma.name().replace('_', ' '))
            
            # Part meronyms (has parts)
            for meronym in synset.part_meronyms():
                for lemma in meronym.lemmas():
                    results['meronyms'].add(lemma.name().replace('_', ' '))
            
            # Part holonyms (part of)
            for holonym in synset.part_holonyms():
                for lemma in holonym.lemmas():
                    results['holonyms'].add(lemma.name().replace('_', ' '))
        
        # Convert sets to lists
        for key in ['synonyms', 'antonyms', 'hypernyms', 'hyponyms', 'meronyms', 'holonyms']:
            results[key] = list(results[key])[:10]  # Limit to 10 for readability
        
        return results
    
    def calculate_semantic_similarity(self, word1, word2):
        """EXTRA: Calculate multiple types of semantic similarity"""
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        
        if not synsets1 or not synsets2:
            return None
        
        similarities = {
            'path_similarity': [],
            'wup_similarity': [],
            'lch_similarity': []
        }
        
        for s1 in synsets1:
            for s2 in synsets2:
                # Path similarity
                path_sim = s1.path_similarity(s2)
                if path_sim:
                    similarities['path_similarity'].append(path_sim)
                
                # Wu-Palmer similarity
                wup_sim = s1.wup_similarity(s2)
                if wup_sim:
                    similarities['wup_similarity'].append(wup_sim)
                
                # Leacock-Chodorow similarity (only for same POS)
                if s1.pos() == s2.pos():
                    try:
                        lch_sim = s1.lch_similarity(s2)
                        if lch_sim:
                            similarities['lch_similarity'].append(lch_sim)
                    except:
                        pass
        
        # Return maximum similarities
        result = {}
        for metric, values in similarities.items():
            result[metric] = max(values) if values else 0.0
        
        return result
    
    def build_semantic_network(self, words):
        """EXTRA: Build a network graph of semantic relationships"""
        G = nx.Graph()
        
        for word in words:
            relationships = self.get_all_relationships(word)
            if not relationships:
                continue
            
            G.add_node(word, node_type='primary')
            
            # Add synonym edges
            for syn in relationships['synonyms'][:5]:
                G.add_node(syn, node_type='synonym')
                G.add_edge(word, syn, relationship='synonym')
            
            # Add antonym edges
            for ant in relationships['antonyms'][:3]:
                G.add_node(ant, node_type='antonym')
                G.add_edge(word, ant, relationship='antonym')
            
            # Add hypernym edges
            for hyper in relationships['hypernyms'][:3]:
                G.add_node(hyper, node_type='hypernym')
                G.add_edge(word, hyper, relationship='hypernym')
        
        return G
    
    def visualize_semantic_network(self, G, title="Semantic Relationship Network"):
        """Visualize the semantic network with color-coded relationships"""
        plt.figure(figsize=(16, 12))
        
        # Define colors for node types
        color_map = {
            'primary': '#FF6B6B',
            'synonym': '#4ECDC4',
            'antonym': '#FFE66D',
            'hypernym': '#95E1D3'
        }
        
        node_colors = [color_map.get(G.nodes[node].get('node_type', 'primary'), '#CCCCCC') 
                      for node in G.nodes()]
        
        # Define edge colors
        edge_colors = []
        edge_styles = []
        for u, v in G.edges():
            rel = G[u][v].get('relationship', 'synonym')
            if rel == 'synonym':
                edge_colors.append('#4ECDC4')
                edge_styles.append('solid')
            elif rel == 'antonym':
                edge_colors.append('#FFE66D')
                edge_styles.append('dashed')
            else:  # hypernym
                edge_colors.append('#95E1D3')
                edge_styles.append('dotted')
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=800, alpha=0.9)
        
        # Draw edges
        for (u, v), color, style in zip(G.edges(), edge_colors, edge_styles):
            nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=[color], 
                                  style=style, width=2, alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                      markersize=10, label='Primary Words'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
                      markersize=10, label='Synonyms'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFE66D', 
                      markersize=10, label='Antonyms'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95E1D3', 
                      markersize=10, label='Hypernyms')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/semantic_network.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved semantic network visualization")
        plt.close()
    
    def create_similarity_matrix(self, words):
        """EXTRA: Create similarity matrices using different metrics"""
        n = len(words)
        matrices = {
            'path': np.zeros((n, n)),
            'wup': np.zeros((n, n)),
            'lch': np.zeros((n, n))
        }
        
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i == j:
                    matrices['path'][i, j] = 1.0
                    matrices['wup'][i, j] = 1.0
                    matrices['lch'][i, j] = 1.0
                elif i < j:
                    sim = self.calculate_semantic_similarity(word1, word2)
                    if sim:
                        matrices['path'][i, j] = matrices['path'][j, i] = sim['path_similarity']
                        matrices['wup'][i, j] = matrices['wup'][j, i] = sim['wup_similarity']
                        matrices['lch'][i, j] = matrices['lch'][j, i] = sim['lch_similarity']
        
        return matrices
    
    def visualize_similarity_matrices(self, words, matrices):
        """Visualize similarity matrices as heatmaps"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        titles = ['Path Similarity', 'Wu-Palmer Similarity', 'Leacock-Chodorow Similarity']
        matrix_keys = ['path', 'wup', 'lch']
        
        for ax, title, key in zip(axes, titles, matrix_keys):
            sns.heatmap(matrices[key], annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=words, yticklabels=words, ax=ax,
                       cbar_kws={'label': 'Similarity Score'}, vmin=0, vmax=1)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Words', fontsize=10)
            ax.set_ylabel('Words', fontsize=10)
        
        plt.suptitle('Semantic Similarity Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/similarity_matrices.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved similarity matrices visualization")
        plt.close()
    
    def create_hierarchy_tree(self, word, max_depth=3):
        """EXTRA: Create hierarchical taxonomy visualization"""
        synsets = wn.synsets(word)
        if not synsets:
            return None
        
        synset = synsets[0]  # Use first synset
        
        G = nx.DiGraph()
        
        def add_hypernyms(syn, depth=0, parent=None):
            if depth > max_depth:
                return
            
            node_label = syn.lemmas()[0].name().replace('_', ' ')
            G.add_node(node_label, depth=depth, definition=syn.definition()[:50])
            
            if parent:
                G.add_edge(parent, node_label)
            
            for hypernym in syn.hypernyms():
                add_hypernyms(hypernym, depth + 1, node_label)
        
        add_hypernyms(synset)
        
        return G
    
    def visualize_hierarchy(self, word, G):
        """Visualize hierarchical taxonomy"""
        if G is None or len(G.nodes()) == 0:
            return
        
        plt.figure(figsize=(14, 10))
        
        # Hierarchical layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Color by depth
        depths = [G.nodes[node]['depth'] for node in G.nodes()]
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=depths, cmap='viridis',
                              node_size=1500, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                              arrowsize=20, width=2, alpha=0.6,
                              connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        plt.title(f'Hypernym Hierarchy for "{word}"', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                    label='Hierarchy Depth', ax=plt.gca())
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hierarchy_tree.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved hierarchy tree visualization")
        plt.close()
    
    def analyze_corpus(self, text):
        """Analyze semantic relationships in a text corpus"""
        from nltk.tokenize import word_tokenize
        nltk.download('punkt', quiet=True)
        
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Filter meaningful words
        meaningful_words = [w for w in tokens if w.isalpha() and 
                          w not in stop_words and len(w) > 3]
        
        # Get unique words (limit for performance)
        unique_words = list(set(meaningful_words))[:20]
        
        return unique_words
    
    def generate_report(self, words_data):
        """Generate comprehensive report"""
        rows = []
        
        for word_data in words_data:
            if not word_data:
                continue
            
            rows.append({
                'Word': word_data['word'],
                'Num_Synsets': len(word_data['synsets']),
                'Num_Synonyms': len(word_data['synonyms']),
                'Num_Antonyms': len(word_data['antonyms']),
                'Num_Hypernyms': len(word_data['hypernyms']),
                'Num_Hyponyms': len(word_data['hyponyms']),
                'Synonyms': ', '.join(word_data['synonyms'][:5]),
                'Antonyms': ', '.join(word_data['antonyms'][:5]),
                'Hypernyms': ', '.join(word_data['hypernyms'][:5])
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(f'{self.output_dir}/semantic_relationships.csv', index=False)
        print(f"✓ Saved semantic relationships report")
        
        return df
    
    def create_dashboard(self, df):
        """Create comprehensive visualization dashboard"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Relationship type distribution
        ax1 = fig.add_subplot(gs[0, 0])
        rel_counts = df[['Num_Synonyms', 'Num_Antonyms', 'Num_Hypernyms', 'Num_Hyponyms']].sum()
        colors = ['#4ECDC4', '#FFE66D', '#95E1D3', '#FF6B6B']
        rel_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Relationship Type Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(['Synonyms', 'Antonyms', 'Hypernyms', 'Hyponyms'], rotation=45)
        
        # 2. Words by number of synsets
        ax2 = fig.add_subplot(gs[0, 1])
        df.nlargest(10, 'Num_Synsets')[['Word', 'Num_Synsets']].plot(
            x='Word', y='Num_Synsets', kind='barh', ax=ax2, color='#A8E6CF', legend=False)
        ax2.set_title('Top 10 Words by Synset Count', fontweight='bold')
        ax2.set_xlabel('Number of Synsets')
        
        # 3. Relationship richness (total relationships per word)
        ax3 = fig.add_subplot(gs[0, 2])
        df['Total_Relations'] = (df['Num_Synonyms'] + df['Num_Antonyms'] + 
                                 df['Num_Hypernyms'] + df['Num_Hyponyms'])
        df.nlargest(10, 'Total_Relations')[['Word', 'Total_Relations']].plot(
            x='Word', y='Total_Relations', kind='barh', ax=ax3, color='#FFD3B6', legend=False)
        ax3.set_title('Most Semantically Rich Words', fontweight='bold')
        ax3.set_xlabel('Total Relationships')
        
        # 4. Relationship composition by word
        ax4 = fig.add_subplot(gs[1, :])
        top_words = df.nlargest(8, 'Total_Relations')
        x = np.arange(len(top_words))
        width = 0.2
        
        ax4.bar(x - 1.5*width, top_words['Num_Synonyms'], width, label='Synonyms', color='#4ECDC4')
        ax4.bar(x - 0.5*width, top_words['Num_Antonyms'], width, label='Antonyms', color='#FFE66D')
        ax4.bar(x + 0.5*width, top_words['Num_Hypernyms'], width, label='Hypernyms', color='#95E1D3')
        ax4.bar(x + 1.5*width, top_words['Num_Hyponyms'], width, label='Hyponyms', color='#FF6B6B')
        
        ax4.set_xlabel('Words', fontweight='bold')
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.set_title('Relationship Composition for Top Words', fontweight='bold', pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(top_words['Word'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Synset distribution
        ax5 = fig.add_subplot(gs[2, 0])
        df['Num_Synsets'].hist(bins=10, ax=ax5, color='#FFAAA5', edgecolor='black')
        ax5.set_title('Distribution of Synset Counts', fontweight='bold')
        ax5.set_xlabel('Number of Synsets')
        ax5.set_ylabel('Frequency')
        
        # 6. Scatter: Synonyms vs Antonyms
        ax6 = fig.add_subplot(gs[2, 1])
        scatter = ax6.scatter(df['Num_Synonyms'], df['Num_Antonyms'], 
                            c=df['Num_Synsets'], cmap='viridis', 
                            s=100, alpha=0.6, edgecolors='black')
        ax6.set_xlabel('Number of Synonyms', fontweight='bold')
        ax6.set_ylabel('Number of Antonyms', fontweight='bold')
        ax6.set_title('Synonyms vs Antonyms', fontweight='bold')
        plt.colorbar(scatter, ax=ax6, label='Synsets')
        
        # 7. Statistics table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        stats_data = [
            ['Total Words Analyzed', len(df)],
            ['Avg Synsets per Word', f"{df['Num_Synsets'].mean():.2f}"],
            ['Avg Synonyms per Word', f"{df['Num_Synonyms'].mean():.2f}"],
            ['Words with Antonyms', f"{(df['Num_Antonyms'] > 0).sum()}"],
            ['Most Synsets', f"{df['Num_Synsets'].max()}"],
            ['Most Relationships', f"{df['Total_Relations'].max()}"]
        ]
        table = ax7.table(cellText=stats_data, cellLoc='left',
                         colWidths=[0.6, 0.4], loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(stats_data)):
            table[(i, 0)].set_facecolor('#E8F4F8')
            table[(i, 1)].set_facecolor('#FFFFFF')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax7.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.suptitle('WordNet Semantic Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/semantic_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved comprehensive dashboard")
        plt.close()


def main():
    print("="*70)
    print("PRACTICAL 5: WordNet Semantic Relationship Analysis")
    print("="*70)
    
    analyzer = WordNetSemanticAnalyzer()
    
    # Sample text for analysis
    sample_text = """
    Artificial intelligence and machine learning are revolutionizing technology.
    Computers learn from data to make predictions and decisions.
    Happy developers create amazing software, while sad bugs cause frustration.
    The large algorithm processes information efficiently, unlike slow manual methods.
    Good code is beautiful, but bad code is ugly and problematic.
    """
    
    # Test words showcasing different relationships
    test_words = ['good', 'happy', 'computer', 'large', 'fast', 'learn', 
                  'create', 'beautiful', 'intelligent', 'efficient']
    
    print("\n1. Analyzing semantic relationships...")
    words_data = []
    for word in test_words:
        print(f"   Processing: {word}")
        data = analyzer.get_all_relationships(word)
        if data:
            words_data.append(data)
            print(f"      Found: {len(data['synonyms'])} synonyms, "
                  f"{len(data['antonyms'])} antonyms, "
                  f"{len(data['hypernyms'])} hypernyms")
    
    # Generate report
    print("\n2. Generating comprehensive report...")
    df = analyzer.generate_report(words_data)
    print(f"   Analyzed {len(df)} words successfully")
    
    # Create similarity matrices
    print("\n3. Calculating semantic similarities...")
    matrices = analyzer.create_similarity_matrix(test_words[:8])  # Limit for performance
    analyzer.visualize_similarity_matrices(test_words[:8], matrices)
    
    # Build and visualize semantic network
    print("\n4. Building semantic network...")
    G = analyzer.build_semantic_network(test_words[:6])
    analyzer.visualize_semantic_network(G)
    print(f"   Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create hierarchy tree
    print("\n5. Creating hypernym hierarchy...")
    hierarchy = analyzer.create_hierarchy_tree('computer')
    analyzer.visualize_hierarchy('computer', hierarchy)
    
    # Create dashboard
    print("\n6. Creating visualization dashboard...")
    analyzer.create_dashboard(df)
    
    # Save detailed relationships
    print("\n7. Saving detailed relationship data...")
    detailed_data = []
    for word_data in words_data:
        for synset_info in word_data['synsets']:
            detailed_data.append({
                'Word': word_data['word'],
                'Synset': synset_info['name'],
                'POS': synset_info['pos'],
                'Definition': synset_info['definition'],
                'Examples': '; '.join(synset_info['examples']) if synset_info['examples'] else 'N/A'
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(f'{analyzer.output_dir}/synset_definitions.csv', index=False)
    print(f"   Saved {len(detailed_df)} synset definitions")
    
    # Calculate and save similarity scores
    print("\n8. Saving similarity scores...")
    similarity_data = []
    for i, word1 in enumerate(test_words[:8]):
        for j, word2 in enumerate(test_words[:8]):
            if i < j:
                sim = analyzer.calculate_semantic_similarity(word1, word2)
                if sim:
                    similarity_data.append({
                        'Word1': word1,
                        'Word2': word2,
                        'Path_Similarity': f"{sim['path_similarity']:.3f}",
                        'WuP_Similarity': f"{sim['wup_similarity']:.3f}",
                        'LCH_Similarity': f"{sim['lch_similarity']:.3f}"
                    })
    
    sim_df = pd.DataFrame(similarity_data)
    sim_df.to_csv(f'{analyzer.output_dir}/similarity_scores.csv', index=False)
    print(f"   Saved {len(sim_df)} pairwise similarity scores")
    
    print("\n" + "="*70)
    print("✓ Analysis Complete!")
    print("="*70)
    print(f"Outputs saved in: {analyzer.output_dir}/")
    print("\nGenerated Files:")
    print("  - semantic_relationships.csv      : Main relationship data")
    print("  - synset_definitions.csv          : Detailed synset information")
    print("  - similarity_scores.csv           : Pairwise similarity metrics")
    print("  - semantic_dashboard.png          : 7-panel analysis dashboard")
    print("  - semantic_network.png            : Interactive network graph")
    print("  - similarity_matrices.png         : 3 similarity heatmaps")
    print("  - hierarchy_tree.png              : Hypernym taxonomy tree")
    print("="*70)


if __name__ == "__main__":
    main()
