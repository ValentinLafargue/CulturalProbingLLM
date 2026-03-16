import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Update these paths to where your raw files are located
WASABI_PATH = Path("Data/deezer_songs_with_translation_information.csv")
SPOTIFY_PATH = Path("Data/spotify_songs_with_translation_information.csv")

df_wasabi = pd.read_csv(WASABI_PATH)
df_spotify = pd.read_csv(SPOTIFY_PATH)

print("Wasabi rows:", len(df_wasabi))
print("Spotify rows:", len(df_spotify))

spotify_final = pd.DataFrame({
    'song_title': df_spotify['song_title'],
    'artist': df_spotify['artist'],
    'ethnicity': df_spotify['ethnicity'],
    'gender': df_spotify['genre'],
    'lyrics': df_spotify['lyrics'],
    'source': 'spotify'
})

wasabi_final = pd.DataFrame({
    'song_title': df_wasabi['title'],
    'artist': df_wasabi['artist'],
    'ethnicity': df_wasabi['ethnicity'],
    'gender': df_wasabi['gender'],
    'lyrics': df_wasabi['lyrics'],
    'source': 'wasabi'
})

df_final = pd.concat([spotify_final, wasabi_final], ignore_index=True)

cont = df_final['ethnicity'].astype(str).str.strip().str.lower()
ethnicity_map = {
    'north-american': 'North America',
    'north america': 'North America',
    'european': 'Europe',
    'europe': 'Europe',
    'latino': 'South America',
    'south america': 'South America',
    'africa': 'Africa',
    'african': 'Africa',
    'asia': 'Asia',
    'asian': 'Asia',
    'oceania': 'Oceania'
}
df_final['ethnicity'] = cont.map(ethnicity_map)

g = df_final['gender'].astype(str).str.strip().str.lower()
gender_map = {
    'male': 'Male',
    'female': 'Female',
    'group': 'Group',
    'band': 'Group',
    'duo': 'Group',
    'collective': 'Group',
    'non-binary': 'Other',
    'person': 'Other',
    'other': 'Other'
}
df_final['gender'] = g.map(gender_map).fillna('Other')

df_final = df_final.dropna(subset=['ethnicity'])
df_final = df_final[df_final['gender'].isin(['Male', 'Female'])].reset_index(drop=True)

print(df_final[['ethnicity', 'gender']].head())
print('Rows after normalization:', len(df_final))

titles = df_final['song_title'].fillna('').astype(str).tolist()
artists = df_final['artist'].fillna('').astype(str).tolist()

X = TfidfVectorizer().fit_transform(titles)
sim_matrix = cosine_similarity(X)

threshold = 0.85
edges = {}
graph = defaultdict(set)

n = len(titles)
for i in range(n):
    for j in range(i + 1, n):
        sim = sim_matrix[i, j]
        if sim >= threshold and artists[i].lower() == artists[j].lower():
            key = frozenset((i, j))
            edges[key] = sim
            graph[i].add(j)
            graph[j].add(i)

visited = set()
components = []
for node in graph:
    if node in visited:
        continue
    stack = [node]
    comp = set()
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        comp.add(v)
        for nei in graph[v]:
            if nei not in visited:
                stack.append(nei)
    if len(comp) > 1:
        components.append(comp)

rows = []
for comp in components:
    anchor = min(comp)
    others = sorted(i for i in comp if i != anchor)
    for j in others:
        key = frozenset((anchor, j))
        if key in edges:
            rows.append({
                'index_A': anchor,
                'index_B': j,
                'song_A': titles[anchor],
                'song_B': titles[j],
                'artist': artists[anchor],
                'similarity': edges[key],
            })

duplicates_df = pd.DataFrame(rows)

to_drop = duplicates_df['index_B'].unique() if len(duplicates_df) else []
df_clean = df_final.drop(index=to_drop).reset_index(drop=True)

print('Removed duplicate rows:', len(to_drop))
print('Final shape:', df_clean.shape)

df_clean.to_csv('merged-ds_unstranslated.csv', index=False)
# Optional: save duplicate pairs for audit
duplicates_df.to_csv('duplicates.csv', index=False)
print('Saved final-merged-ds.csv and duplicates.csv')
