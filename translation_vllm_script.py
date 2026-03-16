#!/usr/bin/env python3
"""
Song Lyrics Translation Script using vLLM API

Translates non-English lyrics to English using a local LLM via vLLM.

File format and translation-flag column are detected automatically from
the input file.
"""

import argparse
import os
import re
import time
from typing import Optional

import pandas as pd
from openai import OpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE  = "merged-ds_unstranslated.csv"
OUTPUT_FILE = "final_merged-ds.csv"

MODEL_NAME  = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
VLLM_API    = "http://localhost:8000/v1"

# =============================================================================


class LyricsTranslator:
    """Translates song lyrics to English using a vLLM-served model."""

    def __init__(self, api_base: str, model_name: str, input_file: str, output_file: str):
        self.client = OpenAI(base_url=api_base, api_key="EMPTY")
        self.model_name = model_name
        self.input_file = input_file
        self.output_file = output_file
        self.df = None

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load input file (CSV or Excel) into a DataFrame."""
        try:
            print(f"Loading: {self.input_file}")
            if self.input_file.endswith(".xlsx"):
                self.df = pd.read_excel(self.input_file)
            else:
                self.df = pd.read_csv(self.input_file)
            print(f"  {len(self.df)} rows | columns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print(f"Error: '{self.input_file}' not found.")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    def save(self, backup: bool = False):
        """Save DataFrame to the output file (CSV or Excel)."""
        try:
            if backup and os.path.exists(self.output_file):
                ext = os.path.splitext(self.output_file)[1]
                backup_path = self.output_file.replace(ext, f"_backup{ext}")
                if not os.path.exists(backup_path):
                    self._write(backup_path)
                    print(f"Backup created: {backup_path}")
            self._write(self.output_file)
            print(f"Saved: {self.output_file}")
        except Exception as e:
            print(f"Error saving: {e}")

    def _write(self, path: str):
        if path.endswith(".xlsx"):
            self.df.to_excel(path, index=False)
        else:
            self.df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Lyrics cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def clean_contributor_prefix(lyrics: str) -> str:
        """Remove Genius-style contributor headers: 'N Contributor(s) TITLE Lyrics'."""
        if pd.isna(lyrics) or not lyrics:
            return lyrics
        pattern = r'^\d+\s+Contributors?\s+.*?\s+Lyrics\s*'
        return re.sub(pattern, '', lyrics, flags=re.IGNORECASE).strip()

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _build_prompt(self, lyrics: str) -> str:
        return (
            "Your task is to translate any non-English portions of the following song lyrics into English, "
            "while keeping any parts that are already in English unchanged.\n\n"
            "Instructions:\n"
            "- Check if the lyrics are entirely in a language other than English. If so, translate the entire lyrics to English.\n"
            "- If the lyrics contain a mix of English and non-English parts, translate only the non-English parts to English.\n"
            "- Maintain the original structure, line breaks, and formatting of the lyrics.\n"
            "- Translate ONLY the non-English parts to English\n"
            "- Keep the original English parts as they are\n"
            "- Maintain the structure, line breaks, and formatting\n"
            "- If the entire lyrics are already in English, return them unchanged\n"
            "- Provide ONLY the translated lyrics in your response, without any additional commentary\n\n"
            f"Lyrics to translate:\n{lyrics}\n\nTranslated lyrics:"
        )

    def translate(self, lyrics: str, max_retries: int = 3) -> Optional[str]:
        if pd.isna(lyrics) or not lyrics:
            return None
        prompt = self._build_prompt(lyrics)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional translator specializing in song lyrics translation."},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        print(f"  Failed after {max_retries} attempts.")
        return None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process(self, save_interval: int = 5):
        if self.df is None:
            print("No data loaded.")
            return

        if 'lyrics' not in self.df.columns:
            print(f"Error: 'lyrics' column not found. Available: {list(self.df.columns)}")
            return

        # --- Step 1: clean contributor prefixes (harmless if not present) ---
        print("\n" + "=" * 60)
        print("STEP 1: Cleaning contributor prefixes")
        print("=" * 60)
        cleaned_count = 0
        for idx in self.df.index:
            original = self.df.loc[idx, 'lyrics']
            cleaned  = self.clean_contributor_prefix(original)
            if cleaned != original and not pd.isna(original):
                cleaned_count += 1
            self.df.loc[idx, 'lyrics'] = cleaned
        print(f"  {cleaned_count} entries cleaned.")

        # --- Step 2: determine which rows need translation ---
        print("\n" + "=" * 60)
        print("STEP 2: Translating non-English lyrics")
        print("=" * 60)

        if 'language_detect' in self.df.columns:
            # Wasabi-style: translate everything that isn't detected as English
            needs_translation = ~self.df['language_detect'].apply(
                lambda v: str(v).lower() == 'english' if not pd.isna(v) else False
            )
            print("  Using 'language_detect' column to select rows.")
        elif 'needs_translation' in self.df.columns:
            # Spotify-style: explicit boolean flag
            needs_translation = self.df['needs_translation'] == True
            print("  Using 'needs_translation' column to select rows.")
        else:
            print("Error: neither 'language_detect' nor 'needs_translation' column found.")
            print(f"  Available columns: {list(self.df.columns)}")
            return

        rows_to_translate = self.df[needs_translation].index.tolist()
        print(f"  {len(rows_to_translate)} rows to translate | {(~needs_translation).sum()} English (will be copied).")

        if 'lyrics_translated' not in self.df.columns:
            self.df['lyrics_translated'] = None

        translated, failed, skipped = 0, 0, 0

        for i, idx in enumerate(rows_to_translate, 1):
            lyrics   = self.df.loc[idx, 'lyrics']
            language = self.df.loc[idx, 'language_detect'] if 'language_detect' in self.df.columns else '?'

            if pd.isna(lyrics) or not str(lyrics).strip():
                print(f"[{i}/{len(rows_to_translate)}] Row {idx} (lang: {language}): skipped (empty)")
                skipped += 1
                continue

            print(f"[{i}/{len(rows_to_translate)}] Row {idx} (lang: {language})...", end=" ")
            result = self.translate(lyrics)

            if result:
                self.df.loc[idx, 'lyrics_translated'] = result
                translated += 1
                print("✓")
            else:
                failed += 1
                print("✗")

            if i % save_interval == 0:
                self.save(backup=True)
                print(f"  Progress: {translated} translated, {failed} failed, {skipped} skipped")

        # Copy English lyrics as-is
        english_mask = ~needs_translation
        self.df.loc[english_mask, 'lyrics_translated'] = self.df.loc[english_mask, 'lyrics']

        print(f"\n{'=' * 60}")
        print("Translation complete!")
        print(f"  Translated:      {translated}")
        print(f"  Failed:          {failed}")
        print(f"  Skipped (empty): {skipped}")
        print(f"  English (copied):{english_mask.sum()}")
        print(f"{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate song lyrics to English via vLLM")
    parser.add_argument("--input_file",  default=INPUT_FILE,  help="Input CSV or Excel file (default: %(default)s)")
    parser.add_argument("--output_file", default=OUTPUT_FILE, help="Output CSV or Excel file (default: %(default)s)")
    parser.add_argument("--api_base",    default=VLLM_API,    help="vLLM API endpoint (default: %(default)s)")
    parser.add_argument("--model_name",  default=MODEL_NAME,  help="Model name served by vLLM (default: %(default)s)")
    parser.add_argument("--save_interval", type=int, default=5, help="Save progress every N rows (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Lyrics Translation Tool (vLLM)")
    print(f"  Input:  {args.input_file}")
    print(f"  Output: {args.output_file}")
    print(f"  Model:  {args.model_name}")
    print("=" * 60)

    translator = LyricsTranslator(
        api_base=args.api_base,
        model_name=args.model_name,
        input_file=args.input_file,
        output_file=args.output_file,
    )

    if not translator.load():
        return

    translator.process(save_interval=args.save_interval)
    translator.save(backup=True)
    print("Done!")


if __name__ == "__main__":
    main()


"""
TO SERVE THE VLLM MODEL LOCALLY:

vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --tokenizer_mode mistral --config_format mistral \
  --load_format mistral --tool-call-parser mistral \
  --enable-auto-tool-choice --limit-mm-per-prompt '{"image":10}' \
  --tensor-parallel-size 2

"""
