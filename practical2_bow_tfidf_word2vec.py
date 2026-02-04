"""
PRACTICAL 2: Advanced BOW, TF-IDF & Word2Vec with Clustering & Visualization
STANDOUT FEATURES:
- 3D Word2Vec visualization with interactive plots
- Document similarity heatmap
- Semantic clustering using Word2Vec
- Comparative analysis of BOW vs TF-IDF vs Word2Vec
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Download required resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class AdvancedTextRepresentation:
    def __init__(self):
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.documents = None
        
    def bag_of_words(self, documents):
        """BOW with count and normalized count"""
        print("\n PROCESSING BAG-OF-WORDS...")
        
        # Count occurrence (absolute counts)
        self.bow_vectorizer = CountVectorizer()
        bow_matrix = self.bow_vectorizer.fit_transform(documents)
        
        # Normalized count occurrence (relative frequencies)
        bow_normalized = bow_matrix.toarray()
        row_sums = bow_normalized.sum(axis=1, keepdims=True)
        bow_normalized = bow_normalized / row_sums
        
        # Create DataFrames
        feature_names = self.bow_vectorizer.get_feature_names_out()
        bow_df = pd.DataFrame(bow_matrix.toarray(), 
                             columns=feature_names,
                             index=[f'Doc{i+1}' for i in range(len(documents))])
        
        bow_norm_df = pd.DataFrame(bow_normalized,
                                   columns=feature_names,
                                   index=[f'Doc{i+1}' for i in range(len(documents))])
        
        print(f"  Vocabulary size: {len(feature_names)}")
        print(f"  Matrix shape: {bow_matrix.shape}")
        
        return bow_df, bow_norm_df
    
    def tfidf_representation(self, documents):
        """TF-IDF with detailed statistics"""
        print("\nPROCESSING TF-IDF...")
        
        self.tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                               columns=feature_names,
                               index=[f'Doc{i+1}' for i in range(len(documents))])
        
        # Calculate IDF values
        idf_values = dict(zip(feature_names, self.tfidf_vectorizer.idf_))
    
        print(f"  Vocabulary size: {len(feature_names)}")
        print(f"  Matrix shape: {tfidf_matrix.shape}")
        
        return tfidf_df, idf_values
    
    def word2vec_embeddings(self, documents, vector_size=100, window=5, min_count=1):
        """EXTRA: Advanced Word2Vec with multiple training configurations"""
        print("\nTRAINING WORD2VEC MODEL...")
        
        # Tokenize documents
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        
        # Train Word2Vec with CBOW
        self.word2vec_model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=0  # CBOW
        )
        
        # Also train Skip-gram for comparison
        skipgram_model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=1  # Skip-gram
        )
        
        print(f"  Vocabulary size: {len(self.word2vec_model.wv)}")
        print(f"  Vector dimensions: {vector_size}")
        
        # Create document vectors (average of word vectors)
        doc_vectors = []
        for tokens in tokenized_docs:
            vectors = [self.word2vec_model.wv[word] for word in tokens 
                      if word in self.word2vec_model.wv]
            if vectors:
                doc_vectors.append(np.mean(vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(vector_size))
        
        return np.array(doc_vectors), self.word2vec_model, skipgram_model
    
    def visualize_word2vec_3d(self, model, save_path='word2vec_3d.png'):
        """EXTRA: 3D visualization of word embeddings"""
        print("\nCreating 3D Word2Vec visualization...")
        
        # Get all words and their vectors
        words = list(model.wv.index_to_key)
        word_vectors = np.array([model.wv[word] for word in words])
        
        # Reduce to 3D using PCA
        pca = PCA(n_components=3)
        vectors_3d = pca.fit_transform(word_vectors)
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by word length
        colors = [len(word) for word in words]
        scatter = ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2],
                           c=colors, cmap='viridis', s=100, alpha=0.6)
        
        # Label some important words
        for i, word in enumerate(words[:20]):  # Label first 20 words
            ax.text(vectors_3d[i, 0], vectors_3d[i, 1], vectors_3d[i, 2], 
                   word, fontsize=9, fontweight='bold')
        
        ax.set_xlabel('PC1', fontweight='bold', fontsize=12)
        ax.set_ylabel('PC2', fontweight='bold', fontsize=12)
        ax.set_zlabel('PC3', fontweight='bold', fontsize=12)
        ax.set_title('3D Word2Vec Embedding Space\n(Color = Word Length)', 
                    fontweight='bold', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Word Length', fontweight='bold')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   3D visualization saved to {save_path}")
        
        # Print variance explained
        print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
        
        return fig
    
    def semantic_clustering(self, model, n_clusters=3, save_path='word_clusters.png'):
        """EXTRA: Cluster words by semantic similarity"""
        print("\nPERFORMING SEMANTIC CLUSTERING...")
        
        words = list(model.wv.index_to_key)
        word_vectors = np.array([model.wv[word] for word in words])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(word_vectors)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(word_vectors)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot 1: Clusters
        scatter = axes[0].scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                                 c=clusters, cmap='tab10', s=150, alpha=0.6)
        
        # Label representative words from each cluster
        for cluster_id in range(n_clusters):
            cluster_words = [words[i] for i in range(len(words)) if clusters[i] == cluster_id]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_center_2d = pca.transform([cluster_center])[0]
            
            # Add cluster label
            axes[0].annotate(f'Cluster {cluster_id}\n{cluster_words[:3]}',
                           xy=cluster_center_2d,
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[0].set_xlabel('PC1', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('PC2', fontweight='bold', fontsize=12)
        axes[0].set_title('Semantic Word Clusters', fontweight='bold', fontsize=14)
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Word distribution by cluster
        cluster_sizes = [sum(clusters == i) for i in range(n_clusters)]
        axes[1].bar(range(n_clusters), cluster_sizes, color=plt.cm.tab10(range(n_clusters)))
        axes[1].set_xlabel('Cluster ID', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Number of Words', fontweight='bold', fontsize=12)
        axes[1].set_title('Words per Cluster', fontweight='bold', fontsize=14)
        axes[1].set_xticks(range(n_clusters))
        
        for i, v in enumerate(cluster_sizes):
            axes[1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Clustering saved to {save_path}")
        
        # Print cluster details
        print("\n  📋 CLUSTER DETAILS:")
        for cluster_id in range(n_clusters):
            cluster_words = [words[i] for i in range(len(words)) if clusters[i] == cluster_id]
            print(f"    Cluster {cluster_id}: {cluster_words[:10]}")
        
        return clusters
    
    def document_similarity_heatmap(self, vectors, method_name, save_path):
        """EXTRA: Create similarity heatmap between documents"""
        similarity_matrix = cosine_similarity(vectors)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   xticklabels=[f'Doc{i+1}' for i in range(len(vectors))],
                   yticklabels=[f'Doc{i+1}' for i in range(len(vectors))],
                   cbar_kws={'label': 'Cosine Similarity'})
        
        plt.title(f'Document Similarity Heatmap ({method_name})', 
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Similarity heatmap saved to {save_path}")
        
        return similarity_matrix
    
    def comparative_analysis(self, bow_df, tfidf_df, w2v_vectors, save_path='comparative_analysis.png'):
        """EXTRA: Compare all three methods comprehensively"""
        print("\nCREATING COMPARATIVE ANALYSIS...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Comparison: BOW vs TF-IDF vs Word2Vec', 
                    fontweight='bold', fontsize=16)
        
        # 1. Sparsity comparison
        bow_sparsity = (bow_df.values == 0).sum() / bow_df.size * 100
        tfidf_sparsity = (tfidf_df.values == 0).sum() / tfidf_df.size * 100
        w2v_sparsity = 0  # Dense representation
        
        methods = ['BOW', 'TF-IDF', 'Word2Vec']
        sparsities = [bow_sparsity, tfidf_sparsity, w2v_sparsity]
        
        axes[0, 0].bar(methods, sparsities, color=['skyblue', 'lightgreen', 'coral'])
        axes[0, 0].set_title('Sparsity Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Sparsity %')
        for i, v in enumerate(sparsities):
            axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # 2. Dimensionality comparison
        dimensions = [bow_df.shape[1], tfidf_df.shape[1], w2v_vectors.shape[1]]
        axes[0, 1].bar(methods, dimensions, color=['skyblue', 'lightgreen', 'coral'])
        axes[0, 1].set_title('Dimensionality Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_yscale('log')
        for i, v in enumerate(dimensions):
            axes[0, 1].text(i, v * 1.1, str(v), ha='center', fontweight='bold')
        
        # 3. Value distribution (BOW)
        axes[0, 2].hist(bow_df.values.flatten(), bins=30, color='skyblue', 
                       edgecolor='black', alpha=0.7)
        axes[0, 2].set_title('BOW Value Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Count Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_yscale('log')
        
        # 4. Value distribution (TF-IDF)
        axes[1, 0].hist(tfidf_df.values.flatten(), bins=30, color='lightgreen',
                       edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('TF-IDF Value Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('TF-IDF Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        
        # 5. Value distribution (Word2Vec)
        axes[1, 1].hist(w2v_vectors.flatten(), bins=30, color='coral',
                       edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Word2Vec Value Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Embedding Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Top terms by method
        top_n = 10
        bow_top = bow_df.sum(axis=0).nlargest(top_n)
        tfidf_top = tfidf_df.sum(axis=0).nlargest(top_n)
        
        y_pos = np.arange(len(bow_top))
        axes[1, 2].barh(y_pos, bow_top.values, alpha=0.6, label='BOW', color='skyblue')
        axes[1, 2].barh(y_pos, tfidf_top.values, alpha=0.6, label='TF-IDF', color='lightgreen')
        axes[1, 2].set_yticks(y_pos)
        axes[1, 2].set_yticklabels(bow_top.index, fontsize=8)
        axes[1, 2].set_title(f'Top {top_n} Terms (BOW vs TF-IDF)', fontweight='bold')
        axes[1, 2].set_xlabel('Score')
        axes[1, 2].legend()
        axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Comparative analysis saved to {save_path}")
        
        return fig

def main():
    print("="*80)
    print("PRACTICAL 2: ADVANCED BOW, TF-IDF & WORD2VEC ANALYSIS")
    print("="*80)
    
    # Sample corpus with diverse documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data",
        "Deep learning uses neural networks with multiple layers to process complex patterns",
        "Natural language processing helps computers understand and generate human language",
        "Data science combines statistics, programming, and domain knowledge to extract insights",
        "Artificial intelligence and machine learning are transforming various industries worldwide",
        "Python is the most popular programming language for machine learning and data science",
        "Neural networks are inspired by the structure of the human brain",
        "Big data analytics requires distributed computing frameworks like Hadoop and Spark"
    ]
    
    system = AdvancedTextRepresentation()
    system.documents = documents
    
    print(f"\nCorpus: {len(documents)} documents")
    
    # 1. Bag of Words
    print("\n" + "="*80)
    print("1. BAG-OF-WORDS ANALYSIS")
    print("="*80)
    bow_df, bow_norm_df = system.bag_of_words(documents)
    
    print("\nCount Occurrence (First 3 docs, top 10 words):")
    print(bow_df.iloc[:3, :10].to_string())
    
    print("\n\nNormalized Count Occurrence (First 3 docs, top 10 words):")
    print(bow_norm_df.iloc[:3, :10].to_string())
    
    bow_df.to_csv('bow_counts.csv')
    bow_norm_df.to_csv('bow_normalized.csv')
    
    # 2. TF-IDF
    print("\n" + "="*80)
    print("2. TF-IDF ANALYSIS")
    print("="*80)
    tfidf_df, idf_values = system.tfidf_representation(documents)
    
    print("\nTF-IDF Scores (First 3 docs, top 10 words):")
    print(tfidf_df.iloc[:3, :10].to_string())
    
    print("\n\nTop 10 Terms by IDF (Rare = Important):")
    sorted_idf = sorted(idf_values.items(), key=lambda x: x[1], reverse=True)[:10]
    for term, idf in sorted_idf:
        print(f"  {term:20s}: {idf:.4f}")
    
    tfidf_df.to_csv('tfidf_scores.csv')
    pd.DataFrame(list(idf_values.items()), 
                columns=['Term', 'IDF']).to_csv('idf_values.csv', index=False)
    
    # 3. Word2Vec
    print("\n" + "="*80)
    print("3. WORD2VEC EMBEDDINGS")
    print("="*80)
    w2v_vectors, cbow_model, skipgram_model = system.word2vec_embeddings(documents, 
                                                                          vector_size=100)
    
    print("\nDocument Vector Shape:", w2v_vectors.shape)
    print("\nSample vector (Doc1, first 10 dims):")
    print(w2v_vectors[0, :10])
    
    # Save Word2Vec model
    cbow_model.save('word2vec_cbow.model')
    skipgram_model.save('word2vec_skipgram.model')
    np.save('document_vectors.npy', w2v_vectors)
    
    # Find similar words
    print("\n\nWORD SIMILARITY EXAMPLES:")
    test_words = ['machine', 'learning', 'data']
    for word in test_words:
        if word in cbow_model.wv:
            similar = cbow_model.wv.most_similar(word, topn=3)
            print(f"\n  Words similar to '{word}':")
            for sim_word, score in similar:
                print(f"    {sim_word:15s}: {score:.4f}")
    
    # 4. EXTRA VISUALIZATIONS
    print("\n" + "="*80)
    print("4. ADVANCED VISUALIZATIONS & ANALYSIS")
    print("="*80)
    
    # 3D Word2Vec visualization
    system.visualize_word2vec_3d(cbow_model)
    
    # Semantic clustering
    system.semantic_clustering(cbow_model, n_clusters=3)
    
    # Document similarity heatmaps
    system.document_similarity_heatmap(bow_df.values, 'BOW', 
                                      'similarity_bow.png')
    system.document_similarity_heatmap(tfidf_df.values, 'TF-IDF', 
                                      'similarity_tfidf.png')
    system.document_similarity_heatmap(w2v_vectors, 'Word2Vec', 
                                      'similarity_word2vec.png')
    
    # Comparative analysis
    system.comparative_analysis(bow_df, tfidf_df, w2v_vectors)
    
    print("\n" + "="*80)
    print("✓ PRACTICAL 2 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n Generated files:")
    print("   bow_counts.csv, bow_normalized.csv")
    print("   tfidf_scores.csv, idf_values.csv")
    print("   word2vec_cbow.model, word2vec_skipgram.model")
    print("   document_vectors.npy")
    print("   word2vec_3d.png (3D embedding visualization)")
    print("   word_clusters.png (Semantic clustering)")
    print("   similarity_*.png (3 heatmaps)")
    print("   comparative_analysis.png (Comprehensive comparison)")

if __name__ == "__main__":
    main()
