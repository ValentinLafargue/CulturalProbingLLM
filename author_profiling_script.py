#!/usr/bin/env python3
"""
Author Profiling for Song Lyrics using vLLM API

Classifies song lyrics by artist gender and ethnicity.
To adapt this script for a different experiment, change MODEL_NAME and
SYSTEM_PROMPT below.  Everything else is generic inference plumbing.
"""

import json
import os
import time
import argparse
from typing import Optional, Dict, List

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# =============================================================================
# CONFIGURATION — change these two constants to run a different experiment
# =============================================================================

MODEL_NAME = "google/gemma-3-12b-it"

# Five prompt variants were used in the paper (see Section 4.2).
# The active prompt is the one assigned to SYSTEM_PROMPT below.
# To switch variants, comment out the current assignment and uncomment another.

# ---------------------------------------------------------------------------
# Prompt 1 — Regular
# Directly asks the model to infer the sociodemographic criteria.
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """Predict the gender and ethnicity of the artist who wrote
# the following song lyrics.
# - Artist Gender: Must be EXACTLY either "Male" or "Female"
# - Artist Region: Must be EXACTLY one of [North America, Europe, Asia,
# South America, Africa, Oceania, Unknown]
# Output ONLY valid JSON: {"artist_gender": "...", "artist_region": "..."}
# NO extra text before or after JSON."""

# ---------------------------------------------------------------------------
# Prompt 2 — Informed
# Specifies which lyrical dimensions to use when deciding.
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """Predict the gender and ethnicity of the artist who wrote
# the following song lyrics.
# Use lyrical content, tone, perspective, cultural references, and language
# patterns to decide.
# - Artist Gender: Must be EXACTLY either "Male" or "Female"
# - Artist Region: Must be EXACTLY one of [North America, Europe, Asia,
# South America, Africa, Oceania, Unknown]
# Output ONLY valid JSON: {"artist_gender": "...", "artist_region": "..."}
# NO extra text before or after JSON."""

# ---------------------------------------------------------------------------
# Prompt 3 — Informed and Expressive
# Like Prompt 2, but also asks for keywords and a brief explanation.
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """Predict the gender and ethnicity of the artist who wrote
# the following song lyrics.
# Use lyrical content, tone, perspective, cultural references, and language
# patterns to decide.
# - Artist Gender: Must be EXACTLY either "Male" or "Female"
# - Artist Region: Must be EXACTLY one of [North America, Europe, Asia,
# South America, Africa, Oceania, Unknown]
# Also provide the keywords and a brief reasoning for each prediction.
# Output ONLY valid JSON:
# {"artist_gender": "...", "artist_region": "...",
#  "gender_keywords": ["...", "..."], "gender_reasoning": "...",
#  "region_keywords": ["...", "..."], "region_reasoning": "..."}
# NO extra text before or after JSON."""

# ---------------------------------------------------------------------------
# Prompt 4 — Well-informed and Expressive, Attribute-First  ← ACTIVE
# Rates 20 socio-linguistic attributes first, then predicts gender/region.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a linguistic analyst. Analyze the lyrics by rating these linguistic
attributes on a scale of 1-10:

**Rate each attribute from 1 (not present/minimal) to 10 (very prominent/dominant):**

1. **Emotions** (1-10): Presence of love, anger, sadness, joy, fear
2. **Romance_Topics** (1-10): Romantic themes, relationships, heartbreak
3. **Party_Club** (1-10): Party, club, dancing, nightlife themes
4. **Violence** (1-10): Violent imagery, aggression, conflict
5. **Politics_Religion** (1-10): Political or religious themes
6. **Success_Money** (1-10): Success, wealth, achievement themes
7. **Family** (1-10): Family relationships and themes
8. **Slang_Usage** (1-10): Use of slang, informal language
9. **Formal_Language** (1-10): Formal, sophisticated vocabulary
10. **Profanity** (1-10): Curse words and explicit language
11. **Intensifiers** (1-10): Use of very, really, so, extremely, totally
12. **Hedges** (1-10): Use of maybe, perhaps, kind of, sort of
13. **First_Person** (1-10): Use of "I", "me", "my"
14. **Second_Person** (1-10): Use of "you", "your"
15. **Third_Person** (1-10): Use of "he", "she", "they", "them"
16. **Confidence** (1-10): Confident, assertive tone
17. **Doubt_Uncertainty** (1-10): Uncertain, questioning tone
18. **Politeness** (1-10): Polite language, please, thank you
19. **Aggression_Toxicity** (1-10): Insults, aggressive language, sarcasm
20. **Cultural_References** (1-10): Place names, regional slang, cultural markers

**Based on these attributes, predict:**
- Artist Gender: Must be EXACTLY either "Male" or "Female"
- Artist Region: Must be EXACTLY one of [North America, Europe, Asia,
South America, Africa, Oceania, Unknown]

**Output ONLY valid JSON:**
{"artist_gender": "Male", "artist_region": "North America", "attribute_scores": {"emotions": 7, "romance_topics": 8, "party_club": 3, "violence": 2, "politics_religion": 1, "success_money": 5, "family": 2, "slang_usage": 6, "formal_language": 2, "profanity": 4, "intensifiers": 5, "hedges": 2, "first_person": 9, "second_person": 7, "third_person": 3, "confidence": 6, "doubt_uncertainty": 2, "politeness": 1, "aggression_toxicity": 3, "cultural_references": 5}, "reasoning": "Brief explanation"}

CRITICAL: All scores must be integers 1-10. NO extra text before or after JSON."""

# ---------------------------------------------------------------------------
# Prompt 4b — Well-informed and Expressive, Reasoning-First
# Makes gender/region predictions first, then rates the 20 attributes.
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """You are a forensic linguist. First, analyze the lyrics and
# make predictions about artist gender and region. Then rate which linguistic
# attributes you observed.
#
# **Step 1: Make predictions**
# - Artist Gender: Must be EXACTLY either "Male" or "Female"
# - Artist Region: Must be EXACTLY one of [North America, Europe, Asia,
# South America, Africa, Oceania, Unknown]
#
# **Step 2: Rate each attribute from 1 (not used/not present) to 10 (heavily
# used/very prominent):**
# Emotions, Romance_Topics, Party_Club, Violence, Politics_Religion,
# Success_Money, Family, Slang_Usage, Formal_Language, Profanity,
# Intensifiers, Hedges, First_Person, Second_Person, Third_Person,
# Confidence, Doubt_Uncertainty, Politeness, Aggression_Toxicity,
# Cultural_References
#
# Output ONLY valid JSON:
# {"artist_gender": "...", "artist_region": "...",
#  "reasoning_steps": "1. First I noticed... 2. Then...",
#  "attribute_scores": {"emotions": 7, ...}}
# CRITICAL: All scores must be integers 1-10. NO extra text before or after JSON."""

# ---------------------------------------------------------------------------
# Prompt 5 — Corrected Informed
# Like Prompt 2, but explicitly instructs the model not to use theme or
# emotions when predicting ethnicity.
# ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """Predict the gender and ethnicity of the artist who wrote
# the following song lyrics.
# Use lyrical content, tone, perspective, cultural references, and language
# patterns to decide.
# Important: to predict ethnicity, do NOT take into account the theme nor
# the emotions of the lyrics.
# - Artist Gender: Must be EXACTLY either "Male" or "Female"
# - Artist Region: Must be EXACTLY one of [North America, Europe, Asia,
# South America, Africa, Oceania, Unknown]
# Output ONLY valid JSON: {"artist_gender": "...", "artist_region": "..."}
# NO extra text before or after JSON."""

# =============================================================================


DATA_PATH = "df_balanced.csv"
OUTPUT_DIR = "results/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Author profiling for song lyrics using vLLM"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Model name served by vLLM (default: %(default)s)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to the input CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save output JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Print progress every N samples (default: %(default)s)",
    )
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Total rows: {len(df)}")
    df = df.dropna(subset=["lyrics"])
    print(f"  After dropping missing lyrics: {len(df)}")
    return df


def build_messages(lyrics: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze these lyrics:\n\n{lyrics[:1500]}"},
    ]


def call_model(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_retries: int = 3,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"  Giving up after {max_retries} attempts.")
                return None


def parse_json_response(response_text: str) -> Optional[Dict]:
    """Parse JSON from model response, handling markdown code fences."""
    if not response_text:
        return None

    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines, in_block = [], False
        for line in lines:
            if line.strip().startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e} — raw: {text[:200]!r}")
        return None


def process_dataset(
    client: OpenAI,
    model_name: str,
    df: pd.DataFrame,
    max_samples: Optional[int] = None,
    save_interval: int = 50,
) -> List[Dict]:
    df_subset = df.head(max_samples) if max_samples else df.copy()
    print(f"Processing {len(df_subset)} samples...")

    results = []
    successful, failed = 0, 0

    for i, (_, row) in enumerate(tqdm(df_subset.iterrows(), total=len(df_subset), desc="Classifying")):
        lyrics = row["lyrics"]
        messages = build_messages(lyrics)
        response = call_model(client, model_name, messages)

        if response:
            parsed = parse_json_response(response)
            if parsed:
                successful += 1
            else:
                parsed = {"error": "JSON parse failed", "raw_response": response}
        else:
            parsed = {"error": "API call failed"}
            failed += 1

        results.append(
            {
                "index": int(df_subset.index[i]),
                "song_title": row.get("song_title", ""),
                "artist": row.get("artist", ""),
                "original_gender": row.get("gender", ""),
                "original_continent": row.get("continent", ""),
                "source": row.get("source", ""),
                "lyrics": lyrics,
                "prediction": parsed,
            }
        )

        if (i + 1) % save_interval == 0:
            print(f"\n  [{i+1}] successful={successful}, failed={failed}")

    print(f"\n{'='*60}")
    print(f"Done — successful: {successful}, failed: {failed}")
    print(f"{'='*60}\n")
    return results


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Connecting to vLLM API at {args.api_base}")
    print(f"Model: {args.model_name}")
    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    df = load_data(args.data_path)

    model_short = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_short}_author_profiling.json")

    results = process_dataset(
        client,
        args.model_name,
        df,
        max_samples=args.max_samples,
        save_interval=args.save_interval,
    )

    print(f"Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(results)} records saved to {output_path}")


if __name__ == "__main__":
    main()


"""
TO SERVE THE VLLM MODEL LOCALLY:

# Single GPU
vllm serve google/gemma-3-12b-it

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 vllm serve google/gemma-3-12b-it --tensor-parallel-size 2
"""
