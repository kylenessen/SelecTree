# %%
from openai import OpenAI
import pandas as pd
from tqdm.notebook import tqdm
from scipy.spatial.distance import cosine

client = OpenAI()

# %%
df = pd.read_csv("common_names_20240206.csv")
# script breaks if there are empty names
df = df.dropna(subset=['common_name'])

# %%


def get_embedding(common_name):
    try:
        response = client.embeddings.create(
            input=common_name,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Failed to get embedding for {common_name}: {e}")
        # Handle the error appropriately: log it, retry, etc.


# %%
# Apply get_embedding function only to rows with missing 'embeddings'
df.loc[:, 'embeddings'] = df['common_name'].apply(get_embedding)


# %%
df = df[~df['embeddings'].isna()]


def cosine_similarity(vector1, vector2):
    assert isinstance(vector1, list), f"Vector1 is not a list: {vector1}"
    assert isinstance(vector2, list), f"Vector2 is not a list: {vector2}"
    assert len(vector1) == len(
        vector2), f"Vectors are of different lengths: {len(vector1)} vs {len(vector2)}"
    return 1 - cosine(vector1, vector2)


def calculate_similarities(group):
    similarities = []
    for i, row1 in group.iterrows():
        for j, row2 in group.iterrows():
            if i < j:  # Avoid duplicate pairs and self-comparison
                emb1 = row1['embeddings']
                emb2 = row2['embeddings']

                # Debugging printouts
                print(f"Comparing indices {i} and {j}")
                print(f"Vector 1 length: {len(emb1)}")
                print(f"Vector 2 length: {len(emb2)}")

                # Calculate similarity
                sim_score = cosine_similarity(emb1, emb2)
                similarities.append(
                    (i, row1['common_name'], j, row2['common_name'], sim_score))
    return similarities


# %%
# Assuming 'df' is your DataFrame with an 'embeddings' column
# Create an empty list to store all similarities
all_similarities = []

# Apply the function to each group
for name, group in df.groupby('scientific_name'):
    group_similarities = calculate_similarities(group)
    all_similarities.extend(group_similarities)

# Create a DataFrame from the similarities list
similarities_df = pd.DataFrame(all_similarities, columns=[
                               'Index', 'CommonName', 'SimilarToIndex', 'SimilarToCommonName', 'SimilarityScore'])

# Now, you can merge this back into your original DataFrame if needed
df = df.merge(similarities_df, left_index=True, right_on='Index')

# Optionally, you can sort by 'SimilarityScore' to see the most similar pairs at the top
df.sort_values(by='SimilarityScore', ascending=False, inplace=True)

# %%

# %%

# Makes the file huge. Basically free to check so dropping it.
df = df.drop('embeddings', axis=1)

# %%

# %%
df['CommonName'] = df['CommonName'].str.upper()
df['SimilarToCommonName'] = df['SimilarToCommonName'].str.upper()
df = df.drop("common_name", axis=1)

df.to_csv("common_name_embeddings_20240206.csv", index=False)
# %%
