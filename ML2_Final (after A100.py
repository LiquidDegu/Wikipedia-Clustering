#!/usr/bin/env python
# coding: utf-8

# # Samantic Clustering of Wikipedia Articles with unsupervised learning
# 
# ## Problem
# I thought I wanted to cluster some Semantic data, but how would I do It? I don't want to label thousands of orticles, also mabe the labels I set are not good clusters. I want this dome automaticly with unsupervised learning

# ## Getting the Data, performing EDA, and Data cleaning
# I got the Data from Wikidumps,
# to get the Data passable I needed to edit the Wikidump and export it to json. I did this with WikiExtractor wich is an open source opten for that (even though it was hart to get to function because of its age)
# 
# Bercause it Split the entire dump (20GB) into small 1MB chunks, first I needed to combine the again
# 
# 
# I have to preclean the Data because I want to transform it with semantic embedding (extract the meaning out of the text) for that I have to deate the stopwords and optimaly reduce the number of words / deleate the pages who have no words, and so on. My dataset features are Title ID and Text.
# Title and Text are strings ID is an integer. In the futer I will also have Embedding as a numpy arry and more. (the more will mostly be arrays that I saved to work on them later)

# In[ ]:


import os
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


data_root = Path("PATH") 
docs = []
for file in tqdm(list(data_root.rglob("*")), desc="Reading files"):
    if not file.is_file():
        continue
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                docs.append({
                    "id":    obj.get("id"),
                    "title": obj.get("title", ""),
                    "text":  obj.get("text", "")
                })
            except json.JSONDecodeError:
                continue



df = pd.DataFrame(docs)
print(f"Loaded {len(df)} documents")



# ### Basic EDA
# First I counted the number of words and charecters per Artice,
# and displayed them in a graph
# I also looked at the distribution of the number of words/ chars from the articles

# In[ ]:


df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print(df[['char_count', 'word_count']].describe())

df['word_count'].hist(bins=50)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")


df_words = df[['word_count']].copy()

fig, ax = plt.subplots(figsize=(10, 6))
bins = np.arange(0, 10_001, 500)          
ax.hist(df_words['word_count'], bins=bins, edgecolor="black")
ax.set(
    title="Distribution of article lengths (words)",
    xlabel="Words per article",
    ylabel="Number of articles"
)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_words.loc[df_words['word_count'] <= 2000, 'word_count'],
        bins=40, edgecolor="black")
ax.set(
    title="Distribution of shorter articles (≤ 2 000 words)",
    xlabel="Words per article",
    ylabel="Number of articles"
)
plt.show()


fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(df_words['word_count'], vert=False, showfliers=False)
ax.set(
    title="Boxplot of article word counts",
    xlabel="Words per article"
)
plt.show()


df_len = df[['word_count', 'char_count']].copy()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_len['word_count'], df_len['char_count'],
           alpha=0.05, s=5)              
ax.set(
    title="Characters vs. Words per article",
    xlabel="Words per article",
    ylabel="Characters per article"
)
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
counts, edges = np.histogram(df_words['word_count'], bins=100)
cdf = np.cumsum(counts) / counts.sum()
ax.plot(edges[1:], cdf)
ax.set(
    title="CDF – proportion of articles up to length X",
    xlabel="Words per article",
    ylabel="Cumulative proportion"
)
plt.show()


top20 = df.nlargest(20, 'word_count')[['title', 'word_count']].reset_index(drop=True)
display(top20)


def simple_tokens(text: str):
    """Lower‑cases & grabs alphanum tokens."""
    return re.findall(r'\b\w+\b', text.lower())

sample_size = 50_000                              
sample = df.sample(min(sample_size, len(df)), random_state=42)

tok_counter = Counter()
for doc_text in sample['text']:
    tok_counter.update(simple_tokens(doc_text))

top_words = pd.DataFrame(tok_counter.most_common(30),
                         columns=['token', 'freq'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_words['token'][::-1], top_words['freq'][::-1])
ax.set(
    title="Top 30 tokens in a 50 k‑article sample",
    xlabel="Frequency"
)
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_0_500   = df.loc[df['word_count']       <=  500].copy()
df_500_1k  = df.loc[(df['word_count'] > 500) &
                    (df['word_count'] <= 1000)].copy()
df_0_1k    = df.loc[df['word_count']       <= 1000].copy()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

axes[0].hist(df_0_500['word_count'],
             bins=np.arange(0, 501, 25), edgecolor="black")
axes[0].set(title="Articles 0 – 500 words",
            xlabel="Words per article", ylabel="Number of articles")

# 500‑1 000 words
axes[1].hist(df_500_1k['word_count'],
             bins=np.arange(500, 1001, 25), edgecolor="black")
axes[1].set(title="Articles 500 – 1 000 words",
            xlabel="Words per article")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(
    [df_0_500['word_count'],
     df_500_1k['word_count'],
     df_0_1k['word_count']],
    labels=["0‑500", "500‑1 000", "0‑1 000"],
    vert=False,
    showfliers=False
)
ax.set(title="Word‑count distribution by range",
       xlabel="Words per article")
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
counts, edges = np.histogram(df_0_1k['word_count'], bins=100)
ax.plot(edges[1:], np.cumsum(counts) / counts.sum())
ax.set(title="CDF for articles ≤ 1 000 words",
       xlabel="Words per article",
       ylabel="Cumulative proportion")
plt.show()


bands = pd.Series({
    "0‑500"      : len(df_0_500),
    "500‑1 000"  : len(df_500_1k),
    "1 000‑∞"    : len(df) - len(df_0_1k)
})
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(bands.index, bands.values)
ax.set(title="Article‑count by length band",
       ylabel="Number of articles")
for i, v in enumerate(bands.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.show()


# ## EDA First results
# If you look at the graphs you will see that there is a high increase in the number of articles the shorter they are, this seems logical. There also are some major outlires when it comes to article lenghth. We will take a menal note of that and process that later
# 
# The distiribution indicates that there are a number of articles where the number of wordas is 0 I cant group them semanticly when I havce no information.

# In[ ]:


num_zero_word = (df['word_count'] == 0).sum()
total         = len(df)

print(f"{num_zero_word:,} of {total:,} pages have 0 words "
      f"({num_zero_word/total:.2%} of the corpus).")


# ### Fist Data Cleaning
# as you can see 1.9 Million articles have no words. We can drop them as they are not computable for my clustering

# In[ ]:


before = len(df)
df_clean = df.loc[df['word_count'] > 0].reset_index(drop=True)
after = len(df_clean)

print(f"Dropped {before - after:,} zero‑word pages "
      f"({(before - after) / before:.2%} of the corpus).")
print(f"Remaining pages: {after:,}")
df = df_clean


# ### EDA after loosing 1.9 Million articles
# because the number of articles that were deleated was so large I wanted to perform the first EDA step again but now without the articles where the number of words is 0

# In[ ]:


df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print(df[['char_count', 'word_count']].describe())

df['word_count'].hist(bins=50)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")

df_words = df[['word_count']].copy()

fig, ax = plt.subplots(figsize=(10, 6))
bins = np.arange(0, 10_001, 500)          
ax.hist(df_words['word_count'], bins=bins, edgecolor="black")
ax.set(
    title="Distribution of article lengths (words)",
    xlabel="Words per article",
    ylabel="Number of articles"
)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_words.loc[df_words['word_count'] <= 2000, 'word_count'],
        bins=40, edgecolor="black")
ax.set(
    title="Distribution of shorter articles (≤ 2 000 words)",
    xlabel="Words per article",
    ylabel="Number of articles"
)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(df_words['word_count'], vert=False, showfliers=False)
ax.set(
    title="Boxplot of article word counts",
    xlabel="Words per article"
)
plt.show()

df_len = df[['word_count', 'char_count']].copy()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_len['word_count'], df_len['char_count'],
           alpha=0.05, s=5)               
ax.set(
    title="Characters vs. Words per article",
    xlabel="Words per article",
    ylabel="Characters per article"
)
plt.show()



fig, ax = plt.subplots(figsize=(8, 6))
counts, edges = np.histogram(df_words['word_count'], bins=100)
cdf = np.cumsum(counts) / counts.sum()
ax.plot(edges[1:], cdf)
ax.set(
    title="CDF – proportion of articles up to length X",
    xlabel="Words per article",
    ylabel="Cumulative proportion"
)
plt.show()


top20 = df.nlargest(20, 'word_count')[['title', 'word_count']].reset_index(drop=True)
display(top20)


def simple_tokens(text: str):
    """Lower‑cases & grabs alphanum tokens."""
    return re.findall(r'\b\w+\b', text.lower())

sample_size = 50_000                              # down‑sample for speed
sample = df.sample(min(sample_size, len(df)), random_state=42)

tok_counter = Counter()
for doc_text in sample['text']:
    tok_counter.update(simple_tokens(doc_text))

top_words = pd.DataFrame(tok_counter.most_common(30),
                         columns=['token', 'freq'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_words['token'][::-1], top_words['freq'][::-1])
ax.set(
    title="Top 30 tokens in a 50 k‑article sample",
    xlabel="Frequency"
)
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ❶ Build throw‑away subsets
df_0_500   = df.loc[df['word_count']       <=  500].copy()
df_500_1k  = df.loc[(df['word_count'] > 500) &
                    (df['word_count'] <= 1000)].copy()
df_0_1k    = df.loc[df['word_count']       <= 1000].copy()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# 0‑500 words
axes[0].hist(df_0_500['word_count'],
             bins=np.arange(0, 501, 25), edgecolor="black")
axes[0].set(title="Articles 0 – 500 words",
            xlabel="Words per article", ylabel="Number of articles")

# 500‑1 000 words
axes[1].hist(df_500_1k['word_count'],
             bins=np.arange(500, 1001, 25), edgecolor="black")
axes[1].set(title="Articles 500 – 1 000 words",
            xlabel="Words per article")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(
    [df_0_500['word_count'],
     df_500_1k['word_count'],
     df_0_1k['word_count']],
    labels=["0‑500", "500‑1 000", "0‑1 000"],
    vert=False,
    showfliers=False
)
ax.set(title="Word‑count distribution by range",
       xlabel="Words per article")
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
counts, edges = np.histogram(df_0_1k['word_count'], bins=100)
ax.plot(edges[1:], np.cumsum(counts) / counts.sum())
ax.set(title="CDF for articles ≤ 1 000 words",
       xlabel="Words per article",
       ylabel="Cumulative proportion")
plt.show()


bands = pd.Series({
    "0‑500"      : len(df_0_500),
    "500‑1 000"  : len(df_500_1k),
    "1 000‑∞"    : len(df) - len(df_0_1k)
})
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(bands.index, bands.values)
ax.set(title="Article‑count by length band",
       ylabel="Number of articles")
for i, v in enumerate(bands.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.show()


# In[ ]:


import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


num_zero = int((df['word_count'] == 0).sum())
print(f"{num_zero} articles have 0 words in the current DataFrame.")


df_short = df[df['word_count'] <= 10].copy()
print(f"{len(df_short)} articles have 0–10 words.")


fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_short['word_count'], bins=np.arange(-0.5, 11.5, 1), edgecolor="black")
ax.set(
    title="Distribution of very short articles (0–10 words)",
    xlabel="Words per article",
    ylabel="Number of articles"
)
plt.show()


def simple_tokens(text):
    """Lower‑case alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())

token_counter = Counter()
for text in df_short['text']:
    token_counter.update(simple_tokens(text))

top_tokens = token_counter.most_common(20)
tokens = [t[0] for t in top_tokens][::-1]  
freqs = [t[1] for t in top_tokens][::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(tokens, freqs)
ax.set(
    title="Top 20 tokens in articles with ≤ 10 words",
    xlabel="Frequency"
)
plt.show()


# ## EDA middel results
# The distribution is still majored at the lower end of the word specturm. The outliars at the top still persist. Because I don't know where the semantic cutoff is I don't want to deleate the number of words lower than 10 also because you can see the highest number of articles with under 10 words has 6. wich after some diging arround in the dataset was offen corrolated with family names

# ### Data cleaning middle
# in ther EDA you coulöd see until now there where german stopwords in the dataset we can deleate them because they provide no valuble inform,ation to our ebedding
# I used this topword list:
# + https://github.com/solariz/german_stopwords

# In[ ]:


from pathlib import Path
import re
from tqdm import tqdm

stopwords_path = Path("PATH")

with open(stopwords_path, 'r', encoding='utf-8') as f:
    custom_german_stopwords = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(custom_german_stopwords)} German stopwords.")

token_re = re.compile(r'\b\w+\b', re.UNICODE)

def clean_text(text: str) -> str:
    tokens = (tok.lower() for tok in token_re.findall(text))
    kept = [tok for tok in tokens if tok not in custom_german_stopwords]
    return " ".join(kept)

tqdm.pandas(desc="Cleaning text with custom stopwords")

df_clean["text"] = df_clean["text"].progress_apply(clean_text)

print(df_clean["text"].head(3))

df = df_clean


# ## EDA Nr. 3
# After deleating the number of stopwords I updated the number words and chars and performed the EDA one again

# In[ ]:


df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print(df[['char_count', 'word_count']].describe())

df['word_count'].hist(bins=50)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Handy Jupyter settings
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")

df_words = df[['word_count']].copy()

fig, ax = plt.subplots(figsize=(10, 6))
bins = np.arange(0, 10_001, 500)          # 0‑10 k words, 500‑word bins
ax.hist(df_words['word_count'], bins=bins, edgecolor="black")
ax.set(
    title="Distribution of article lengths (words)",
    xlabel="Words per article",
    ylabel="Number of articles"
)
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_words.loc[df_words['word_count'] <= 2000, 'word_count'],
        bins=40, edgecolor="black")
ax.set(
    title="Distribution of shorter articles (≤ 2 000 words)",
    xlabel="Words per article",
    ylabel="Number of articles"
)
plt.show()


fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(df_words['word_count'], vert=False, showfliers=False)
ax.set(
    title="Boxplot of article word counts",
    xlabel="Words per article"
)
plt.show()


df_len = df[['word_count', 'char_count']].copy()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_len['word_count'], df_len['char_count'],
           alpha=0.05, s=5)               # light dots for big data
ax.set(
    title="Characters vs. Words per article",
    xlabel="Words per article",
    ylabel="Characters per article"
)
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
counts, edges = np.histogram(df_words['word_count'], bins=100)
cdf = np.cumsum(counts) / counts.sum()
ax.plot(edges[1:], cdf)
ax.set(
    title="CDF – proportion of articles up to length X",
    xlabel="Words per article",
    ylabel="Cumulative proportion"
)
plt.show()

top20 = df.nlargest(20, 'word_count')[['title', 'word_count']].reset_index(drop=True)
display(top20)


def simple_tokens(text: str):
    """Lower‑cases & grabs alphanum tokens."""
    return re.findall(r'\b\w+\b', text.lower())

sample_size = 50_000                              # down‑sample for speed
sample = df.sample(min(sample_size, len(df)), random_state=42)

tok_counter = Counter()
for doc_text in sample['text']:
    tok_counter.update(simple_tokens(doc_text))

top_words = pd.DataFrame(tok_counter.most_common(30),
                         columns=['token', 'freq'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_words['token'][::-1], top_words['freq'][::-1])
ax.set(
    title="Top 30 tokens in a 50 k‑article sample",
    xlabel="Frequency"
)
plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ❶ Build throw‑away subsets
df_0_500   = df.loc[df['word_count']       <=  500].copy()
df_500_1k  = df.loc[(df['word_count'] > 500) &
                    (df['word_count'] <= 1000)].copy()
df_0_1k    = df.loc[df['word_count']       <= 1000].copy()


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# 0‑500 words
axes[0].hist(df_0_500['word_count'],
             bins=np.arange(0, 501, 25), edgecolor="black")
axes[0].set(title="Articles 0 – 500 words",
            xlabel="Words per article", ylabel="Number of articles")

# 500‑1 000 words
axes[1].hist(df_500_1k['word_count'],
             bins=np.arange(500, 1001, 25), edgecolor="black")
axes[1].set(title="Articles 500 – 1 000 words",
            xlabel="Words per article")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(
    [df_0_500['word_count'],
     df_500_1k['word_count'],
     df_0_1k['word_count']],
    labels=["0‑500", "500‑1 000", "0‑1 000"],
    vert=False,
    showfliers=False
)
ax.set(title="Word‑count distribution by range",
       xlabel="Words per article")
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
counts, edges = np.histogram(df_0_1k['word_count'], bins=100)
ax.plot(edges[1:], np.cumsum(counts) / counts.sum())
ax.set(title="CDF for articles ≤ 1 000 words",
       xlabel="Words per article",
       ylabel="Cumulative proportion")
plt.show()


bands = pd.Series({
    "0‑500"      : len(df_0_500),
    "500‑1 000"  : len(df_500_1k),
    "1 000‑∞"    : len(df) - len(df_0_1k)
})
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(bands.index, bands.values)
ax.set(title="Article‑count by length band",
       ylabel="Number of articles")
for i, v in enumerate(bands.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.show()


# In[ ]:


import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


num_zero = int((df['word_count'] == 0).sum())
print(f"{num_zero} articles have 0 words in the current DataFrame.")


df_short = df[df['word_count'] <= 10].copy()
print(f"{len(df_short)} articles have 0–10 words.")


fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_short['word_count'], bins=np.arange(-0.5, 11.5, 1), edgecolor="black")
ax.set(
    title="Distribution of very short articles (0–10 words)",
    xlabel="Words per article",
    ylabel="Number of articles"
)
plt.show()


def simple_tokens(text):
    """Lower‑case alphanumeric tokens."""
    return re.findall(r"\b\w+\b", text.lower())

token_counter = Counter()
for text in df_short['text']:
    token_counter.update(simple_tokens(text))

top_tokens = token_counter.most_common(20)
tokens = [t[0] for t in top_tokens][::-1]  
freqs = [t[1] for t in top_tokens][::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(tokens, freqs)
ax.set(
    title="Top 20 tokens in articles with ≤ 10 words",
    xlabel="Frequency"
)
plt.show()


# In[ ]:


word_to_count = "der"
total_der = df["text"].str.count(rf"\b{word_to_count}\b").sum()

print(f'The word "{word_to_count}" appears {total_der:,} times in the corpus.')


# In[ ]:


df.to_parquet("cleaned_wiki_de.parquet", index=False)


# In[ ]:


df.head()


# # TDIF Vectorisation TEST
# #### Hypothesis: This will not work because TDIF is't rally a semantic algorithm it just looks at the relevance of each words but easyly gets lost one you write truck instead of heavy transport vehicle
# #### I ended up not using it because it wasn't "smart" enough and needent a better sentiment embedding

# In[ ]:


import pandas as pd


df = pd.read_parquet("cleaned_wiki_de.parquet")

print(f"Loaded {len(df):,} articles")
print(df.columns)
print(df.head(2))


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from tqdm import tqdm


batch_size = 1000  
docs = df['text'].tolist()


vectorizer = TfidfVectorizer(
    max_features=200_000,   
    min_df=5,
    max_df=0.5
)

print("Fitting TF-IDF vocabulary...")
vectorizer.fit(docs)  

print("Transforming documents in batches with progress...")
X_batches = []
for i in tqdm(range(0, len(docs), batch_size)):
    batch_docs = docs[i:i+batch_size]
    X_batch = vectorizer.transform(batch_docs)
    X_batches.append(X_batch)

X_tfidf = vstack(X_batches)

print("TF-IDF shape:", X_tfidf.shape)


# #### Fitting TF-IDF vocabulary...
# took 5 Minutes

# In[ ]:


from scipy.sparse import save_npz

save_npz("tfidf_matrix.npz", X_tfidf)


# In[ ]:


import joblib

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# In[ ]:


from scipy.sparse import load_npz

X_tfidf = load_npz("tfidf_matrix.npz")


# ### Fitting TDIF to the clustering model

# In[ ]:


import numpy as np
from scipy.sparse import load_npz
import joblib
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt

X_tfidf = load_npz("tfidf_matrix.npz")                # your (3M × 8.3M) sparse matrix
vectorizer = joblib.load("tfidf_vectorizer.pkl")      # fitted TfidfVectorizer





# In[ ]:


svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)  # → (3M × 100)


# In[ ]:


def find_best_k(X, k_min=2, k_max=20, sample_size=10000):
    idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
    X_samp = X[idx]

    best_k, best_score = k_min, -1
    scores = {}
    for k in tqdm(range(k_min, k_max+1), desc="Silhouette scan"):
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X_samp)
        score = silhouette_score(X_samp, labels)
        scores[k] = score
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score, scores

best_k, best_score, scores = find_best_k(X_reduced, k_min=2, k_max=30, sample_size=20000)
print(f"▶ Best k = {best_k} (silhouette = {best_score:.4f})")



# In[ ]:


kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X_reduced)



# In[ ]:


terms = vectorizer.get_feature_names_out()
centroids_orig = kmeans.cluster_centers_ @ svd.components_
top_terms = []
for i in range(best_k):
    top_idx = np.argsort(centroids_orig[i])[::-1][:10]
    top_terms.append([terms[j] for j in top_idx])

pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_reduced)

plot_size = 15000
idx_plot = np.random.choice(X_2d.shape[0], size=plot_size, replace=False)
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(
    X_2d[idx_plot,0],
    X_2d[idx_plot,1],
    c=labels[idx_plot],
    alpha=0.5,
    s=5
)

centers_2d = pca2.transform(kmeans.cluster_centers_)
for i, (x,y) in enumerate(centers_2d):
    ax.text(x, y, f"#{i}\n" + ", ".join(top_terms[i][:5]),
            fontsize=9, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.3))

ax.set_title(f"KMeans clustering (k={best_k}) of Wikipedia articles")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
plt.tight_layout()
plt.show()


# ###TDIF results
# The results with TDIF where underwhelming you yould see a major clustering at the  0,0 point and then only linerar ofshhot groups

# # Now with e5 embedding
# the e5 large model is a embedding model wich scores high on the huggingface embedding leaderboard. Espeachily in clustering. The good thing is because it is only a 512b parameter model it works on a moderalty speced hardware
# 
# Because the model input is limited I neede to perform further data cleaning and EDA

# #### EDA for e5

# In[ ]:


count_above_2200 = len(df[df['word_count'] > 2200])
print(f"Number of articles with word count > 2200: {count_above_2200}")


# In[ ]:


median_above_2200 = df[df['word_count'] > 2200]['word_count'].median()
print(f"Median word count of articles with > 2200 words: {median_above_2200}")


# In[ ]:


mean_above_2200 = df[df['word_count'] > 2200]['word_count'].mean()
print(f"Mean word count of articles with > 2200 words: {mean_above_2200:.2f}")


# #### Datacleaning for e5
# Because the number of tokens is limited I wanted to remove the number wich I think don't hold further value. Namely the numbers 0-99.
# They where found in the remaining number of words suprisingly often, and shoudn't have much of a meaning because tehy are used in lsits etc.

# In[ ]:


from collections import Counter
import re

# Combine all text into one big string
all_text = " ".join(df['text'].astype(str))

# Tokenize the text: lowercase, remove punctuation, split by whitespace
words = re.findall(r'\b\w+\b', all_text.lower())

# Count word frequencies
word_counts = Counter(words)

# Get the 100 most common words
common_words = word_counts.most_common(100)

# Display as a DataFrame (optional, for readability)
common_df = pd.DataFrame(common_words, columns=['word', 'count'])
print(common_df)


# In[ ]:


import re

removed_word_count = 0

def clean_text_and_count(text):
    global removed_word_count
    text = text.lower()
    to_remove = re.findall(r'\b([1-9]?[0-9])\b', text)
    removed_word_count += len(to_remove)
    text = re.sub(r'\b([1-9]?[0-9])\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_cleaned'] = df['text'].astype(str).apply(clean_text_and_count)

print(f"🔢 Number of numeric words (0–99) removed: {removed_word_count:,}")


# #### Results after datacleaning
# Another 20 Million chars removed
# BUt this is still not enough so I loaded the bigger stopwords list and cleaned the Text with it

# In[ ]:


df['word_count_cleaned'] = df['text_cleaned'].apply(lambda x: len(x.split()))

total_words = df['word_count_cleaned'].sum()
print(f"📝 Total words after cleaning: {total_words:,}")
print(df[['word_count', 'word_count_cleaned']].describe())


# In[ ]:


count_above_2200 = len(df[df['word_count_cleaned'] > 2200])
print(f"Number of articles with word count > 2200: {count_above_2200}")


# In[ ]:


median_above_2200 = df[df['word_count_cleaned'] > 2200]['word_count_cleaned'].median()
print(f"Median word count of articles with > 2200 words: {median_above_2200}")


# In[ ]:


mean_above_2200 = df[df['word_count_cleaned'] > 2200]['word_count_cleaned'].mean()
print(f"Mean word count of articles with > 2200 words: {mean_above_2200:.2f}")


# In[ ]:


from collections import Counter
import re
from itertools import chain

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

token_lists = df['text_cleaned'].apply(tokenize)
all_words = list(chain.from_iterable(token_lists))

word_counts = Counter(all_words)
common_words = word_counts.most_common(20)

common_df = pd.DataFrame(common_words, columns=['word', 'count'])
print(common_df)


# In[ ]:


from collections import Counter
import re

all_text = " ".join(df['text'].astype(str))

words = re.findall(r'\b\w+\b', all_text.lower())

word_counts = Counter(words)

common_words = word_counts.most_common(100)

common_df = pd.DataFrame(common_words, columns=['word', 'count'])
print(common_df)


# In[ ]:


from pathlib import Path
import re
from tqdm import tqdm

stopwords_path = Path("german_stopwords_full.txt")

with open(stopwords_path, 'r', encoding='utf-8') as f:
    custom_german_stopwords = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(custom_german_stopwords)} German stopwords.")

token_re = re.compile(r'\b\w+\b', re.UNICODE)

def clean_text(text: str) -> str:
    tokens = (tok.lower() for tok in token_re.findall(text))
    kept = [tok for tok in tokens if tok not in custom_german_stopwords]
    return " ".join(kept)

tqdm.pandas(desc="Cleaning text with custom stopwords")

df["text_cleaned"] = df["text_cleaned"].progress_apply(clean_text)

print(df["text_cleaned"].head(3))



# ##### Saving the output to not have to do the long process again and again

# In[ ]:


print(df.head())


# In[ ]:


df = df.drop(columns=["text", "word_count", "char_count"])


# In[ ]:


df.to_parquet("wikipedia_stream.parquet", index=False)


# ### Dataclening to 500 tokens+
# because the model can only take 514 tokens I needed to prune each text to fit into the token limit (For this I had Chatgpt help me)
# 
# Because I had to tokenise the entire text to do that (I think) I first limited the number of words/ characters the Text could have and cut of after the first 450. The intorduction with the most valuble information is tipicly saved tehre so I should still have semantics intact.
# 
# #### EDA for Tokens
# At first I thought jsut limiting the number of Chars to 350 would solve the issue the problem with that is the long German Words . you will see some graphs below wich display the number of tokens to the number of words.
# 
# So the only feasible way to combat this was to cutt of at the embedding level

# In[ ]:


import pyarrow.parquet as pq

pq_file = pq.ParquetFile("wikipedia_stream.parquet")
print("Num Row Groups:", pq_file.num_row_groups)
print("Num Rows:", pq_file.metadata.num_rows)
print("Columns:", pq_file.schema)


# In[ ]:


import pandas as pd

df = pd.read_parquet("wikipedia_stream.parquet")
total_limited_char_count = df["text_cleaned"].str.len().clip(upper=1000).sum()
print(total_limited_char_count)


# In[ ]:


import pandas as pd

df = pd.read_parquet("wikipedia_stream.parquet")
def cap_at_350_words(text):
    return ' '.join(text.split()[:450])

df['text_350_words'] = df['text_cleaned'].apply(cap_at_350_words)


# In[ ]:


import matplotlib.pyplot as plt

over_limit_count = (df['token_count_350'] > 514).sum()

print(f"Number of articles over 514 tokens: {over_limit_count}")

plt.figure(figsize=(10, 6))
plt.hist(df['token_count_350'], bins=50, edgecolor='black')
plt.axvline(x=514, color='red', linestyle='--', label='Token limit (514)')
plt.title('Distribution of Token Counts (350-word intros)')
plt.xlabel('Token count')
plt.ylabel('Number of articles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

df['char_count_350'] = df['text_350_words'].str.len()

token_char_stats = df.groupby('token_count_350')['char_count_350'].agg(['mean', 'count']).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(token_char_stats['token_count_350'], token_char_stats['mean'], marker='o')
plt.title('Average Character Count vs. Token Count (350-word samples)')
plt.xlabel('Token Count')
plt.ylabel('Average Character Count')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


range_df = df[(df['token_count_350'] >= 490) & (df['token_count_350'] <= 514)]

average_chars = range_df['char_count_350'].mean()

print(f"📏 Average character count for token length 490–514: {average_chars:.2f}")


# In[ ]:


def cap_at_2015_chars_preserve_words(text):
    if len(text) <= 2015:
        return text
    trimmed = text[:2015]
    last_space = trimmed.rfind(' ')
    return trimmed[:last_space] if last_space != -1 else trimmed

df['text_2015char_clean'] = df['text_350_words'].apply(cap_at_2015_chars_preserve_words)


# In[ ]:


from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")

texts = df['text_2015char_clean'].tolist()

batch_size = 512
token_counts = []

for i in tqdm(range(0, len(texts), batch_size), desc="Batch tokenizing"):
    batch = texts[i:i+batch_size]
    encoded = tokenizer(batch, truncation=False, padding=False, add_special_tokens=True)
    lengths = [len(ids) for ids in encoded['input_ids']]
    token_counts.extend(lengths)

df['token_count_350_char'] = token_counts


# In[ ]:


import matplotlib.pyplot as plt

over_limit_count = (df['token_count_350_char'] > 514).sum()

print(f"Number of articles over 514 tokens: {over_limit_count}")

plt.figure(figsize=(10, 6))
plt.hist(df['token_count_350_char'], bins=50, edgecolor='black')
plt.axvline(x=514, color='red', linestyle='--', label='Token limit (514)')
plt.title('Distribution of Token Counts (350-word intros)')
plt.xlabel('Token count')
plt.ylabel('Number of articles')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")

def truncate_batch(texts, max_tokens=500):
    encodings = tokenizer(
        texts,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
        add_special_tokens=True
    )

    truncated_texts = []
    for text, offsets, input_ids in zip(texts, encodings['offset_mapping'], encodings['input_ids']):
        if len(input_ids) <= max_tokens:
            truncated_texts.append(text)
        else:
            last_char_pos = offsets[max_tokens - 1][1]
            truncated_texts.append(text[:last_char_pos].rstrip())
    return truncated_texts

BATCH_SIZE = 1024

def batched_truncate(df, column="text_350_words"):
    results = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Batch Tokenizing"):
        batch = df[column].iloc[i:i + BATCH_SIZE].tolist()
        truncated = truncate_batch(batch)
        results.extend(truncated)
    return results

df['text_exact_500'] = batched_truncate(df)

print(df.head)


# In[ ]:


import pandas as pd

df.to_parquet("wikidump_514.parquet", index=False)


# In[ ]:


from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("wikidump_514.parquet", columns=['text_exact_514'])

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")

texts = df['text_exact_514'].tolist()
batch_size = 512
token_counts = []

for i in tqdm(range(0, len(texts), batch_size), desc="Counting tokens in batches"):
    batch = texts[i:i+batch_size]
    encoded = tokenizer(batch, truncation=False, padding=False, add_special_tokens=True)
    lengths = [len(ids) for ids in encoded['input_ids']]
    token_counts.extend(lengths)

df['token_count_exact_514'] = token_counts

plt.figure(figsize=(10, 6))
plt.hist(df['token_count_exact_514'], bins=50, edgecolor='black')
plt.title('Token Count Distribution for text_exact_514')
plt.xlabel('Token Count')
plt.ylabel('Number of Articles')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Resusts Now I have a Workable Text set now onto the embedding
# Because of the size of the dataset every step took some time but the time for the embedding had me floored. It should take 200h so I decided I would only process the first half of my dataset from here on foreward.
# With a more Powerful GPU this would have been possible with the entire dataset, unfortionatly I am not lucky enough to have a A100

# In[ ]:


df = pd.read_parquet("wikidump_514.parquet")
print(df.head)


# In[ ]:


print(df.columns)


# In[ ]:


import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

df = pd.read_parquet("wikidump_514.parquet")

df = df.iloc[:len(df) // 2]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {device}")

model = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=device)

texts = ["passage: " + t for t in df['text_exact_500'].tolist()]

batch_size = 64
embeddings = []

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batch"):
    batch = texts[i:i + batch_size]
    batch_embeddings = model.encode(
        batch,
        batch_size=batch_size,
        show_progress_bar=False,
        device=device
    )
    embeddings.extend(batch_embeddings)

df["embedding"] = embeddings

os.makedirs("chunks_50_embedding", exist_ok=True)
df.to_parquet("chunks_50_embedding/wikidump_part_01_with_embedding.parquet", index=False)

print("✅ Embedding complete and saved.")


# ### Time
# This Embedding for half of df took on a 4090 7 hours

# # Unsupervised Clustering
# 

# ## Kmeans

# In[2]:


import pandas as pd
df = pd.read_parquet("wikidump_part_01_with_embedding.parquet")

print(df.columns)
print(df.head(2))


# In[3]:


df = df.drop(columns=["text_cleaned", "id", "text_350_words", "text_exact_500"])


# In[4]:


df.to_parquet("wikidump_half_title_embedding.parquet", index=False)


# In[2]:


import pandas as pd
df = pd.read_parquet("wikidump_half_title_embedding.parquet")

print(df.columns)
print(df.head(2))


# In[ ]:


import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

embeddings = np.vstack(df['embedding'].values).astype('float32')




# In[ ]:


faiss.normalize_L2(embeddings)


# In[5]:


import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
# Set number of clusters
n_clusters = 15000
kmeans = faiss.Kmeans(d=embeddings.shape[1], k=n_clusters, niter=20, verbose=True)
kmeans.train(embeddings)

# Search nearest cluster center for each vector (with tqdm progress)
batch_size = 100_000
cluster_ids = []

index = kmeans.index
for i in tqdm(range(0, embeddings.shape[0], batch_size), desc="Assigning clusters"):
    end = min(i + batch_size, embeddings.shape[0])
    _, I = index.search(embeddings[i:end], 1)
    cluster_ids.extend(I.flatten())

df['cluster'] = cluster_ids


# In[ ]:


top_articles = (
    df.groupby('cluster')['title']
    .apply(lambda x: x.sample(n=min(5, len(x)), random_state=42).tolist())
    .reset_index(name='example_titles')
)

print(top_articles.head(50))


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm

embeddings = np.vstack(df['embedding'].values).astype('float32')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px



# Reduce to 2D using PCA
pca = PCA(n_components=2)
embedding_2d = pca.fit_transform(embeddings)

# Add to DataFrame
df["x"] = embedding_2d[:, 0]
df["y"] = embedding_2d[:, 1]



# In[10]:


print(df.head(2))


# In[11]:


df.to_parquet("wikidump_half_title_embedding_cluster_x_y.parquet", index=False)


# In[2]:


import pandas as pd
df = pd.read_parquet("wikidump_half_title_embedding_cluster_x_y.parquet")


# In[ ]:


import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()


# In[3]:


import plotly.express as px

fig = px.scatter(
    df.sample(100_000),
    x="x", y="y",
    color="cluster",
    hover_data=["title"],
    title="Wikipedia Clusters in 2D",
    width=1000, height=800
)
fig.show()



# In[ ]:


import plotly.express as px
import numpy as np

random_clusters = np.random.choice(df['cluster'].unique(), size=2, replace=False)

filtered_df = df[df['cluster'].isin(random_clusters)]

filtered_df = filtered_df.sample(min(10000, len(filtered_df)))

fig = px.scatter(
    filtered_df,
    x="x", y="y",
    color="cluster",
    hover_data=["title"],
    title=f"Comparison of 2 Random Wikipedia Clusters: {random_clusters[0]} vs {random_clusters[1]}",
    width=1000, height=800
)
fig.show()


# In[7]:


df = pd.read_parquet("wikidump_half_title_embedding_cluster_x_y.parquet")


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

selected_clusters = np.random.choice(df['cluster'].unique(), size=10, replace=False)

samples_per_cluster = 1000
filtered_df = pd.concat([
    df[df['cluster'] == cluster].sample(n=min(samples_per_cluster, len(df[df['cluster'] == cluster])), random_state=42)
    for cluster in selected_clusters
])

pca = PCA(n_components=3)
embedding_3d = pca.fit_transform(np.vstack(filtered_df['embedding'].values).astype('float32'))

filtered_df["x3d"] = embedding_3d[:, 0]
filtered_df["y3d"] = embedding_3d[:, 1]
filtered_df["z3d"] = embedding_3d[:, 2]

fig = px.scatter_3d(
    filtered_df,
    x="x3d", y="y3d", z="z3d",
    color="cluster",
    hover_data=["title"],
    title="3D Visualization of 10 Random Wikipedia Clusters",
    width=1000, height=800
)
fig.show()


# ### Results of K-Means
# AS you could see the K-means Clustering provided workable clusters.
# With a bit of manual Validation many of the clusters are correct or mostly correct.
# The biggest Problem of the M-Means algorithm is that you have to define the number of clusters.
# I chose 15.000 Clusters after some experementing because that provided good results with less wrong articles in a cluster.
# But I also did't want to have way to many clusters.
# 
# #### Mabe there is something better?
# Because of havong to set the number of Clusters the performance was not the best clustering possible.
# Next I wanted to use HDBSCAN to find out the number of clusters Dynamicly

# ## HDBSCAN
# Because performing hdbscan on my hardware with 1000 dimensions was not possible I will bring down the number of dimensions down to 30, first with fast PCA and than more granualr with UMAP.
# After that I displayed a few exambles of the clusters in graphs. More on that in the results section.

# In[ ]:


import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

print("📥 Loading data...")
df = pd.read_parquet("wikidump_half_title_embedding.parquet")

print("🔄 Converting embeddings to NumPy array...")
start = time.time()
embeddings = np.vstack(df['embedding'].values).astype('float32')
print(f"✅ Done in {time.time() - start:.2f} seconds.")






# In[ ]:


import time
from sklearn.decomposition import PCA
from umap import UMAP
import hdbscan

# Reduce input dims from 1000 to 100
print("🔍 Running PCA...")
pca = PCA(n_components=100, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

# Step 2: UMAP 10D
print("🧭 Running UMAP...")
umap_model = UMAP(
    n_components=10,
    n_neighbors=30,
    metric='cosine',
    random_state=42,
    verbose=True,
    n_jobs=-1  # Use all cores
)
X_umap = umap_model.fit_transform(embeddings_pca)

for i in range(X_umap.shape[1]):
    df[f'umap_{i}'] = X_umap[:, i]

print("🔗 Running HDBSCAN...")
start = time.time()
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, metric='euclidean')
labels = clusterer.fit_predict(X_umap)
print(f"✅ HDBSCAN done in {time.time() - start:.2f} seconds.")

df['hdbscan_cluster'] = labels

output_path = "wikidump_half_title_embedding_cluster_umap.parquet"
df.to_parquet(output_path, index=False)
print(f"📦 Saved clustered DataFrame with UMAP to: {output_path}")


# In[ ]:


import time
import umap.umap_ as umap
from sklearn.decomposition import PCA
from umap import UMAP

# Reduce input dims from 1000 to 100
pca = PCA(n_components=100, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

umap_model = UMAP(n_components=10, n_neighbors=30, metric='cosine', random_state=42, verbose=True)
X_umap = umap_model.fit_transform(embeddings_pca)


# In[ ]:


print("🔗 Running HDBSCAN...")
start = time.time()
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, metric='euclidean')
labels = clusterer.fit_predict(X_umap)
print(f"✅ HDBSCAN done in {time.time() - start:.2f} seconds.")

# Assign labels
df['hdbscan_cluster'] = labels



# In[ ]:


import time
import cupy as cp
import cudf
from cuml.decomposition import PCA
from cuml.manifold import UMAP

# Load data
print("📥 Loading data...")
import pandas as pd
import numpy as np
df = pd.read_parquet("wikidump_half_title_embedding.parquet")

# Convert embeddings to CuPy / cuDF
print("🔄 Converting embeddings to GPU format...")
start = time.time()
embeddings_np = np.vstack(df['embedding'].values).astype('float32')
embeddings_gpu = cp.asarray(embeddings_np)  # OR cudf.DataFrame(embeddings_np)
print(f"✅ Done in {time.time() - start:.2f} seconds.")

# CA on GPU
print("📊 Running PCA on GPU...")
start = time.time()
pca = PCA(n_components=100, random_state=42)
embeddings_pca_gpu = pca.fit_transform(embeddings_gpu)
print(f"✅ PCA done in {time.time() - start:.2f} seconds.")

# UMAP on GPU
print("🗺️ Running UMAP on GPU...")
start = time.time()
umap_model = UMAP(n_components=10, n_neighbors=30, metric='cosine', random_state=42, verbose=True)
X_umap_gpu = umap_model.fit_transform(embeddings_pca_gpu)
print(f"✅ UMAP done in {time.time() - start:.2f} seconds.")


# In[2]:


df.to_parquet("wikidump_half_title_embedding_cluster_umap.parquet", index=False)


# In[3]:


import pandas as pd

df = pd.read_parquet("wikidump_half_title_embedding_cluster_umap.parquet")


# In[3]:


from cuml.cluster import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=30)
labels = clusterer.fit_predict(X_umap_gpu)


# In[4]:


labels_np = cp.asnumpy(labels)

df['cluster'] = labels_np


# In[4]:


print(df.head(3))


# In[5]:


df.to_parquet("wikidump_clustered.parquet", index=False)


# In[ ]:


import matplotlib.pyplot as plt
import cupy as cp

# Get 2D coordinates
X_plot = cp.asnumpy(X_umap_gpu[:, :2])
labels_cpu = cp.asnumpy(labels)

plt.figure(figsize=(12, 8))
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels_cpu, cmap='tab20', s=2)
plt.colorbar(label="Cluster ID")
plt.title("Wikipedia Embedding Clusters (UMAP + Clustering)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()


# In[ ]:


import pandas as pd
import cupy as cp

df['cluster'] = cp.asnumpy(labels)
df['x'] = cp.asnumpy(X_umap_gpu[:, 0])
df['y'] = cp.asnumpy(X_umap_gpu[:, 1])


# In[ ]:


import plotly.express as px
import numpy as np

# 2 
random_clusters = np.random.choice(df['cluster'].unique(), size=2, replace=False)

filtered_df = df[df['cluster'].isin(random_clusters)]

# downsample because otherwise I just wont load/ use way to much ram
filtered_df = filtered_df.sample(min(10000, len(filtered_df)))

fig = px.scatter(
    filtered_df,
    x="x", y="y",
    color="cluster",
    hover_data=["title"],
    title=f"Comparison of 2 Random Wikipedia Clusters: {random_clusters[0]} vs {random_clusters[1]}",
    width=1000, height=800
)
fig.show()


# In[ ]:


import plotly.express as px
import numpy as np

# 6
random_clusters = np.random.choice(df['cluster'].unique(), size=6, replace=False)
filtered_df = df[df['cluster'].isin(random_clusters)]

# downsample because otherwise I just wont load/ use way to much ram
filtered_df = filtered_df.sample(min(10000, len(filtered_df)))

# Plot
fig = px.scatter(
    filtered_df,
    x="x", y="y",
    color="cluster",
    hover_data=["title"],
    title=f"Comparison of 6 Random Wikipedia Clusters: {', '.join(map(str, random_clusters))}",
    width=1000, height=800
)
fig.show()


# In[ ]:


import plotly.express as px

# No noise (-1)
filtered_df = df[df['cluster'] != -1]

# downsample because otherwise I just wont load/ use way to much ram
filtered_df = filtered_df.sample(min(10000, len(filtered_df)))

fig = px.scatter(
    filtered_df,
    x="x", y="y",
    color="cluster",
    hover_data=["title"],
    title="All Wikipedia Clusters (Noise Removed, Sampled)",
    width=1000, height=800
)
fig.show()


# In[ ]:


import numpy as np
import plotly.express as px

# 100 random clusters
random_clusters = np.random.choice(df['cluster'].unique(), size=100, replace=False)

filtered_df_100 = df[df['cluster'].isin(random_clusters)]

# downsample because otherwise I just wont load/ use way to much ram
filtered_df_100 = filtered_df_100.sample(min(10000, len(filtered_df_100)))

fig = px.scatter(
    filtered_df_100,
    x="x", y="y",
    color="cluster",
    hover_data=["title"],
    title="Comparison of 100 Random Wikipedia Clusters (sampled)",
    width=1000, height=800
)
fig.show()


# In[ ]:


import cupy as cp
from cuml.cluster import HDBSCAN
import matplotlib.pyplot as plt
from collections import Counter

labels_cpu = cp.asnumpy(labels)

num_clusters = len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)
print(f"🔢 Number of clusters found (excluding noise): {num_clusters}")

cluster_counts = Counter(labels_cpu)
sorted_clusters, sorted_sizes = zip(*sorted(cluster_counts.items(), key=lambda x: x[0]))

plt.figure(figsize=(12, 6))
plt.bar([str(c) for c in sorted_clusters], sorted_sizes)
plt.title("Cluster Size Distribution")
plt.xlabel("Cluster ID (-1 = noise)")
plt.ylabel("Number of Items")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


#Convert labels from CuPy to NumPy
labels_cpu = cp.asnumpy(labels)

#(excluding noise)
num_clusters = len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)
print(f"🔢 Number of clusters found (excluding noise): {num_clusters}")

#items (excluding noise)
cluster_counts = Counter(label for label in labels_cpu if label != -1)
sorted_clusters, sorted_sizes = zip(*sorted(cluster_counts.items(), key=lambda x: x[0]))

plt.figure(figsize=(12, 6))
plt.bar([str(c) for c in sorted_clusters], sorted_sizes)
plt.title("Cluster Size Distribution (Noise Removed)")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Items")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[6]:


df[df['cluster'] != -1][['cluster', 'x', 'y']].to_csv("wikipedia_clusters_xy_no_noise.csv", index=False)


# In[7]:


df[['cluster', 'x', 'y']].to_csv("wikipedia_clusters_xy_with_noise.csv", index=False)


# ### Results HDBSCAN
# As you might be able to see the clusters worked really well. It decided to make arround 17000 clusters. The clusters in of themselfs were semanticly good. 
# The number of articles inside the cliusters are not evenly distributed in Wikipedia but that is not suprising. HDB had problems with noise (articles wehere it coudn't find 30 to cluster them together)
# The Graphs you see are projected down to 2 dminesions from 10 so the many of the information about the distance of clusters can not be displayed in that way.
# 
# Overall the HDBScan worked better, mainly because I didn't have to set the number of clusters myself

# # Supervised approach
# Because the Supervidsed approach only works if I group some Wikipedia articles beforehand, like in the last Kaggle Project a supervised approach could then sort the rest into these groups.  I implimented some dummy code to simualte that. 
# Dummy because I don't want to label the wikipedia articles

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

np.random.seed(42)
embeddings_np = np.random.rand(1000, 100).astype('float32')  
labels = np.random.choice(['Science', 'History', 'Music', 'Technology'], size=1000)  

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings_np, labels, test_size=0.2, random_state=42)

# Step 3: Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


# # Overall Discussion / Results
# 
# The Semantic clustering worked really well, both algorithms where able to deliver passable clusters of the topics of the wikipedia articles. 
# Both of them had some downsides for example K-means I had to input the number of clusters mysef, whereas because of the min. articles HDBscan had for a cluster it had more problem with noise.
# 
# ## Hyperparamitisation
# I changed the parameters by hand to find good and working parameters. Because of hardware limitations I was not able to Programmaticly find "the best" Hyperparameters
# 
# ## Usecase
# Projects like these could be used with further refinement for axample to cluster and combine Company documentation. 
# Many Times I have come across scattered Compay documentation without a red line to follow. Mit Semantic clustering I think some it it could be lessened.
# The alsorithm could precluster them and find a new headline to group them under to better find stuff when you search for it
# 
# ## Improvements for the future
# ### What went wrong?
# One of the main problems was GPU power and scope. The entire German wikipedia on a Laptop / household PC was way to large of a scope.
# Because of that I had to use only half of the german wikipedia halfway through because of comute times.
# 
# Also there could be better Hyperparemeterisation / reducing noise.
# 
# In the end I think I was to careful with the datacleaning, and I should have "eliminated" every article with under 4 Words further reducing noise.
# 
# ### Gerneral Improvements
# Run this on a more powerful Hardware to compute the entirety.
# Use more complex embedding algorithms to get more semantic information
# 

# ##### Extra
# Later I had acess to a A100 and I Clustered it without first reducing it to 30 dimensions insead working with 300. The parquet is available on Kaggle. But because of time constrains I don't have the time to rerun / rework my entire project

# In[2]:


# 🚀 Full GPU-based Wikipedia Embedding Clustering Pipeline

import time
import pandas as pd
import numpy as np
import cupy as cp
import cudf
from cuml import UMAP
from cuml.cluster import HDBSCAN

# Step 1: Load embeddings
print("📥 Loading data...")
df = pd.read_parquet("wikidump_half_title_embedding.parquet")

# Step 2: Convert embeddings to cuDF for GPU processing
print("🔄 Converting embeddings to GPU format...")
start = time.time()
embeddings_np = np.vstack(df['embedding'].values).astype('float32')
embeddings_gpu_df = cudf.DataFrame(embeddings_np)
print(f"✅ Done in {time.time() - start:.2f} seconds.")

# Step 3: UMAP on GPU
print("🗺️ Running UMAP on GPU...")
start = time.time()
umap_model = UMAP(
    n_components=200,
    n_neighbors=30,
    metric='cosine',
    random_state=42,
    verbose=True
)
X_umap_gpu = umap_model.fit_transform(embeddings_gpu_df)
print(f"✅ UMAP done in {time.time() - start:.2f} seconds.")

# Step 4: HDBSCAN on GPU
print("🔎 Running HDBSCAN on GPU...")
start = time.time()
clusterer = HDBSCAN(
    min_cluster_size=30,
    metric='euclidean',  # UMAP changes the space, so Euclidean is fine here
    cluster_selection_method='eom'
)
labels_gpu = clusterer.fit_predict(X_umap_gpu)  # Output: cudf.Series
df['cluster'] = labels_gpu.to_pandas()
print(f"✅ HDBSCAN done in {time.time() - start:.2f} seconds.")

# Step 5: Show summary
print("📦 Example cluster counts:")
print(df['cluster'].value_counts())

# Optional: Save result
# df.to_parquet("clustered_output_gpu.parquet")


# In[3]:


# 🧩 Step 1: Convert UMAP dimensions to a DataFrame
umap_columns = [f'umap_{i}' for i in range(X_umap_gpu.shape[1])]
umap_df = X_umap_gpu.to_pandas()
umap_df.columns = umap_columns

# 🧩 Step 2: Convert cluster labels to pandas
labels = labels_gpu.to_pandas()

# 🧩 Step 3: Merge everything
df_umap = pd.concat([df.reset_index(drop=True), umap_df], axis=1)
df_umap['cluster'] = labels

# 💾 Step 4: Export
df_umap.to_parquet("clustered_embeddings_with_umap.parquet", index=False)
# Or if you prefer CSV:
# df_umap.to_csv("clustered_embeddings_with_umap.csv", index=False)

print("✅ UMAP dimensions and clusters saved!")


# In[4]:


print(df_umap.head())


# In[ ]:





# ### Tools I Used
# 
# + I used ChatGPT to impliment timings into my code, sometimes it that changed a few parts of the code when I only wanted to implement the abilyty to estimate the runtime. (Minutes or Hours or days)
# + I also used Huggingface Leaderbord for semantic embeddings, the Wikidums, and Chatgpt to help be find the right embedding algorithms / clustering types.
# + Chagpt to impliment visual Print statements (and with that sometimes comments becuase GPT cant help itself)

# 
