#!/usr/bin/env python3
"""Create stratified differential word clouds by continent and gender.
Shows distinctive words for correct vs incorrect within each category."""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
import os
from pathlib import Path
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
_lemmatizer = WordNetLemmatizer()

import argparse

parser = argparse.ArgumentParser(description='Create stratified differential word clouds')
parser.add_argument('--input', type=str, required=True, help='Input CSV file')
parser.add_argument('--output_dir', type=str, default='wordclouds_stratified_differential', help='Output directory')
parser.add_argument('--color', choices=['highlight', 'full'], default='highlight',
                    help='Color scheme: "highlight" = grayscale with orange on "emotional"/"theme" (default); '
                         '"full" = standard colormap')
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

def load_predictions(path: str) -> pd.DataFrame:
    """Load predictions from CSV or JSON/JSONL."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=suffix in {".jsonl", ".ndjson"})
    raise ValueError(f"Unsupported input format: {suffix}")

# Load the predictions
df = load_predictions(args.input)

print(f"Total samples: {len(df)}")

df['gender_normalized'] = df['original_gender'].str.lower().str.strip()
df['predicted_gender_normalized'] = df['predicted_gender'].str.lower().str.strip()
df['continent_normalized'] = df['original_continent'].str.lower().str.strip()
df['predicted_continent_normalized'] = df['predicted_continent'].str.lower().str.strip()
df['gender_correct'] = df['gender_normalized'] == df['predicted_gender_normalized']
df['continent_correct'] = df['continent_normalized'] == df['predicted_continent_normalized']

custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    'gender', 'continent', 'male', 'female', 'africa', 'asia', 'europe',
    'north', 'south', 'america', 'oceania', 'lyrics', 'song', 'artist',
    'indicates', 'suggests', 'likely', 'appears', 'seems', 'could', 'would',
    'may', 'might', 'can', 'will', 'shall', 'based', 'use', 'using', 'used',
    'mention', 'mentions', 'mentioned', 'reference', 'references', 'referenced',
    'word', 'words', 'language', 'phrase', 'phrases', 'text', 'context',
    'prediction', 'predicted', 'predict', 'reasoning', 'reason', 'reasons',
    'keyword', 'keywords', 'narrator', 'speaker', 'singer', 'one', 'two', 'three',
    'african', 'asian', 'european', 'american', 'australian', 'latin',
    'culture', 'cultural', 'cultures', 'music',
    'implies', 'supports', 'elements', 'suggest', 'aligns', 'associated'
])

HIGHLIGHT_WORDS = {'emotional', 'theme'}
HIGHLIGHT_COLOR = 'rgb(220, 80, 0)'


def highlight_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """Color only 'emotional' and 'theme'; everything else is grayscale."""
    if word.lower() in HIGHLIGHT_WORDS:
        return HIGHLIGHT_COLOR
    gray = random_state.randint(60, 150) if random_state is not None else 100
    return f'rgb({gray}, {gray}, {gray})'


color_func = highlight_color_func if args.color == 'highlight' else None


def clean_text(text):
    """Clean and prepare text for word cloud."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_word_frequencies(text_series):
    """Get word frequency counter from a series of text."""
    all_words = []
    for text in text_series.apply(clean_text).dropna():
        words = text.split()
        words = [_lemmatizer.lemmatize(w) for w in words if w not in custom_stopwords and len(w) > 2]
        # Re-filter after lemmatization (e.g. "cultures" → "culture" which is now a stopword)
        words = [w for w in words if w not in custom_stopwords]
        all_words.extend(words)
    return Counter(all_words)

def get_differential_frequencies(freq1, freq2, min_ratio=1.1):
    """Get differential word frequencies. Only keep words more prominent in freq1."""
    differential = {}
    total1 = sum(freq1.values())
    total2 = sum(freq2.values())

    if total1 == 0 or total2 == 0:
        return freq1

    for word, count1 in freq1.items():
        count2 = freq2.get(word, 0)
        norm1 = count1 / total1
        norm2 = count2 / total2

        if norm2 == 0:
            differential[word] = count1
        elif norm1 / norm2 >= min_ratio:
            differential[word] = count1

    return differential

def create_differential_wordcloud(freq_dict, title, filename, min_words=5):
    """Create and save a word cloud from a frequency dictionary."""
    if not freq_dict or len(freq_dict) < min_words:
        print(f"Skipping {filename}: insufficient words ({len(freq_dict)} < {min_words})")
        return

    wordcloud = WordCloud(
        width=1200,
        height=1200,
        background_color='white',
        max_words=80,
        color_func=color_func,
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(freq_dict)

    plt.figure(figsize=(12, 12))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout(pad=0)

    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {full_path} ({len(freq_dict)} unique words)")
    plt.close()

# Get unique continents and genders
continents = df['original_continent'].unique()
genders = df['original_gender'].unique()

# ===== CONTINENT STRATIFIED DIFFERENTIAL WORD CLOUDS =====
print("\n" + "="*80)
print("CREATING CONTINENT STRATIFIED DIFFERENTIAL WORD CLOUDS")
print("="*80)

for continent in continents:
    print(f"\nProcessing continent: {continent}")
    continent_data = df[df['original_continent'] == continent]
    continent_label = str(continent).strip()
    if continent_label.lower() in {"south america", "latin america"}:
        continent_label = "Latin America"

    continent_incorrect = continent_data[~continent_data['continent_correct']]
    continent_correct = continent_data[continent_data['continent_correct']]

    print(f"  Incorrect: {len(continent_incorrect)}, Correct: {len(continent_correct)}")

    if len(continent_incorrect) < 5 or len(continent_correct) < 5:
        print(f"  Skipping - insufficient samples")
        continue

    # Get word frequencies
    freq_incorrect = get_word_frequencies(continent_incorrect['continent_reasoning'])
    freq_correct = get_word_frequencies(continent_correct['continent_reasoning'])

    # Get differential frequencies
    diff_incorrect = get_differential_frequencies(freq_incorrect, freq_correct)
    diff_correct = get_differential_frequencies(freq_correct, freq_incorrect)

    # Create word clouds
    continent_safe = continent.lower().replace(" ", "_")

    create_differential_wordcloud(
        diff_incorrect,
        f'{continent_label.upper()} - Incorrect',
        f'continent_diff_incorrect_{continent_safe}.png'
    )

    create_differential_wordcloud(
        diff_correct,
        f'{continent_label.upper()} - Correct',
        f'continent_diff_correct_{continent_safe}.png'
    )

    # Print top words
    print(f"\n  Top 10 distinctive words for INCORRECT {continent} predictions:")
    for word, count in sorted(diff_incorrect.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {word:20s}: {count:4d}")

    print(f"\n  Top 10 distinctive words for CORRECT {continent} predictions:")
    for word, count in sorted(diff_correct.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {word:20s}: {count:4d}")

# ===== GENDER STRATIFIED DIFFERENTIAL WORD CLOUDS =====
print("\n" + "="*80)
print("CREATING GENDER STRATIFIED DIFFERENTIAL WORD CLOUDS")
print("="*80)

for gender in genders:
    print(f"\nProcessing gender: {gender}")
    gender_data = df[df['original_gender'] == gender]

    gender_incorrect = gender_data[~gender_data['gender_correct']]
    gender_correct = gender_data[gender_data['gender_correct']]

    print(f"  Incorrect: {len(gender_incorrect)}, Correct: {len(gender_correct)}")

    if len(gender_incorrect) < 5 or len(gender_correct) < 5:
        print(f"  Skipping - insufficient samples")
        continue

    # Get word frequencies
    freq_incorrect = get_word_frequencies(gender_incorrect['gender_reasoning'])
    freq_correct = get_word_frequencies(gender_correct['gender_reasoning'])

    # Get differential frequencies
    diff_incorrect = get_differential_frequencies(freq_incorrect, freq_correct)
    diff_correct = get_differential_frequencies(freq_correct, freq_incorrect)

    # Create word clouds
    gender_safe = gender.lower()

    create_differential_wordcloud(
        diff_incorrect,
        f'{gender.upper()} - Incorrect',
        f'gender_diff_incorrect_{gender_safe}.png'
    )

    create_differential_wordcloud(
        diff_correct,
        f'{gender.upper()} - Correct',
        f'gender_diff_correct_{gender_safe}.png'
    )

print("\n" + "="*80)
print("ALL STRATIFIED DIFFERENTIAL WORD CLOUDS CREATED!")
print("="*80)
