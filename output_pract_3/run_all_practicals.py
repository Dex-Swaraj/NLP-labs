"""
MASTER SCRIPT - Run All 4 Advanced NLP Practicals
This script executes all practicals in sequence with progress tracking
"""

import subprocess
import sys
import time
from datetime import datetime

def print_banner(text):
    """Print a styled banner"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_practical(script_name, practical_num, description):
    """Run a single practical script"""
    print_banner(f"PRACTICAL {practical_num}: {description}")
    
    print(f"⏱️  Starting: {datetime.now().strftime('%H:%M:%S')}")
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Completed in {elapsed:.2f} seconds")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}")
        print(f"Error: {e}")
        return False, 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False, 0

def main():
    print("\n" + "🚀"*40)
    print("   ADVANCED NLP PRACTICALS - MASTER EXECUTION SUITE")
    print("   Running all 4 practicals sequentially...")
    print("🚀"*40)
    
    # Define practicals
    practicals = [
        {
            'script': 'practical1_tokenization_stemming.py',
            'num': 1,
            'desc': 'Advanced Tokenization & Stemming Analysis'
        },
        {
            'script': 'practical2_bow_tfidf_word2vec.py',
            'num': 2,
            'desc': 'BOW, TF-IDF & Word2Vec with Clustering'
        },
        {
            'script': 'practical3_preprocessing_tfidf.py',
            'num': 3,
            'desc': 'Text Preprocessing & Feature Engineering'
        },
        {
            'script': 'practical4_ner_advanced.py',
            'num': 4,
            'desc': 'Named Entity Recognition System'
        }
    ]
    
    # Track results
    results = []
    total_start = time.time()
    
    # Run each practical
    for practical in practicals:
        success, elapsed = run_practical(
            practical['script'],
            practical['num'],
            practical['desc']
        )
        results.append({
            'num': practical['num'],
            'name': practical['desc'],
            'success': success,
            'time': elapsed
        })
        
        if not success:
            print(f"\n⚠️  Practical {practical['num']} failed. Continuing...")
        
        # Small delay between practicals
        time.sleep(1)
    
    # Final summary
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print("  📊 EXECUTION SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\n📋 Individual Results:")
    print("-" * 80)
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        print(f"  Practical {r['num']}: {status:10s} - {r['time']:6.2f}s - {r['name']}")
    
    print("\n" + "="*80)
    if successful == len(results):
        print("  🎉 ALL PRACTICALS COMPLETED SUCCESSFULLY!")
    else:
        print(f"  ⚠️  {failed} practical(s) had errors. Check logs above.")
    print("="*80)
    
    # List output files
    print("\n📁 Generated Output Files:")
    print("-" * 80)
    print("""
Practical 1:
  • tokenization_comparison.png
  • stemming_analysis.png
  • stemmer_comparison.csv
  • stemmer_differences.csv

Practical 2:
  • bow_counts.csv, bow_normalized.csv
  • tfidf_scores.csv, idf_values.csv
  • word2vec_cbow.model, word2vec_skipgram.model
  • document_vectors.npy
  • word2vec_3d.png
  • word_clusters.png
  • similarity_bow.png, similarity_tfidf.png, similarity_word2vec.png
  • comparative_analysis.png

Practical 3:
  • preprocessed_texts.csv
  • augmented_texts.csv
  • tfidf_features.csv
  • label_encoding.csv
  • quality_metrics.csv
  • preprocessing_impact.png
  • feature_importance.png

Practical 4:
  • extracted_entities.csv
  • ner_predictions.csv
  • ner_metrics.csv
  • classification_report.csv
  • entity_distribution.png
  • confusion_matrix.png
  • performance_metrics.png

Total: 30+ output files!
    """)
    
    print("="*80)
    print("  🌟 Check the README.md for detailed documentation")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
