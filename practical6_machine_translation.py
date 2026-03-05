"""
PRACTICAL 6: Machine Translation with Indian Language Support & BLEU Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from transformers import MarianMTModel, MarianTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')

warnings.filterwarnings('ignore')


class MachineTranslationSystem:

    def __init__(self):

        self.output_dir = "output_pract_6"
        os.makedirs(self.output_dir, exist_ok=True)

        self.indian_languages = {
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi'
        }

        self.models = {}
        self.smoothing = SmoothingFunction()

    def load_model(self, src="en", tgt="hi"):

        key = f"{src}-{tgt}"

        if key in self.models:
            return self.models[key]

        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"

        print(f"Loading model {model_name}")

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        self.models[key] = (tokenizer, model)

        return tokenizer, model

    def translate(self, text, src="en", tgt="hi"):

        tokenizer, model = self.load_model(src, tgt)

        tokens = tokenizer(text, return_tensors="pt", padding=True)

        translated = model.generate(**tokens)

        output = tokenizer.decode(translated[0], skip_special_tokens=True)

        return output

    def back_translate(self, text, tgt="hi"):

        forward = self.translate(text, "en", tgt)
        backward = self.translate(forward, tgt, "en")

        return forward, backward

    def calculate_bleu(self, reference, hypothesis):

        ref = [word_tokenize(reference.lower())]
        hyp = word_tokenize(hypothesis.lower())

        scores = {
            "BLEU-1": sentence_bleu(ref, hyp, weights=(1,0,0,0),
                                   smoothing_function=self.smoothing.method1),

            "BLEU-2": sentence_bleu(ref, hyp, weights=(0.5,0.5,0,0),
                                   smoothing_function=self.smoothing.method1),

            "BLEU-3": sentence_bleu(ref, hyp, weights=(0.33,0.33,0.33,0),
                                   smoothing_function=self.smoothing.method1),

            "BLEU-4": sentence_bleu(ref, hyp, weights=(0.25,0.25,0.25,0.25),
                                   smoothing_function=self.smoothing.method1)
        }

        return scores

    def similarity(self, s1, s2):

        w1 = set(word_tokenize(s1.lower()))
        w2 = set(word_tokenize(s2.lower()))

        if not w1 or not w2:
            return 0

        return len(w1 & w2) / len(w1 | w2)

    def translate_dataset(self, sentences, tgt="hi"):

        results = []

        for s in sentences:

            print("\nOriginal:", s)

            forward, backward = self.back_translate(s, tgt)

            bleu = self.calculate_bleu(s, backward)
            sim = self.similarity(s, backward)

            print("Translated:", forward)
            print("Back:", backward)
            print("BLEU-4:", round(bleu["BLEU-4"],3))

            results.append({
                "Original": s,
                "Translation": forward,
                "BackTranslation": backward,
                "BLEU-1": bleu["BLEU-1"],
                "BLEU-2": bleu["BLEU-2"],
                "BLEU-3": bleu["BLEU-3"],
                "BLEU-4": bleu["BLEU-4"],
                "Similarity": sim
            })

        return pd.DataFrame(results)

    def dashboard(self, df):

        fig, ax = plt.subplots(2,2, figsize=(12,8))

        df[["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]].boxplot(ax=ax[0,0])
        ax[0,0].set_title("BLEU Score Distribution")

        df["Similarity"].hist(ax=ax[0,1])
        ax[0,1].set_title("Similarity Distribution")

        ax[1,0].scatter(df["BLEU-4"], df["Similarity"])
        ax[1,0].set_xlabel("BLEU-4")
        ax[1,0].set_ylabel("Similarity")

        df[["BLEU-1","BLEU-2","BLEU-3","BLEU-4"]].mean().plot(kind="bar", ax=ax[1,1])
        ax[1,1].set_title("Average BLEU Scores")

        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/translation_dashboard.png")

        print("Dashboard saved.")


def main():

    print("\nMachine Translation Practical\n")

    mt = MachineTranslationSystem()

    sentences = [

        "Hello how are you today",

        "India is a beautiful country",

        "Artificial intelligence is changing the world",

        "Education leads to success",

        "Technology connects people"
    ]

    df = mt.translate_dataset(sentences, "hi")

    df.to_csv("output_pract_6/translations.csv", index=False)

    mt.dashboard(df)

    print("\nAverage BLEU:", round(df["BLEU-4"].mean(),3))


if __name__ == "__main__":
    main()