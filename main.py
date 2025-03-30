import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationSystem:
    def __init__(self, data_dir, model='sentence-transformers/all-mpnet-base-v2'):
        self.data_dir = Path(data_dir)
        self.model = SentenceTransformer(model)
        
    def load_data(self, csv_file):
        """Load item data."""
        return pd.read_csv(self.data_dir / csv_file)
    
    def clean_names(self, df, column, company_names):
        """Remove company names from a given column."""
        copied_df = df.copy()
        for name in company_names:
            copied_df[column] = copied_df[column].str.replace(name, '', regex=False).str.strip()
        copied_df[f'{column}_CLEANED'] = copied_df[column]
        return copied_df
    
    def get_low_super_cat_items(self, df):
        """Identify items in super categories with fewer than four items."""
        copied_df = df.copy()
        super_cats = copied_df.groupby("SUPER_CAT").filter(lambda x: len(x) < 4)["SUPER_CAT"].unique()
        return copied_df[copied_df["SUPER_CAT"].isin(super_cats)]
    
    def get_high_super_cat_items(self, df):
        """Identify items in super categories with four or more items."""
        copied_df = df.copy()
        super_cats = copied_df.groupby("SUPER_CAT").filter(lambda x: len(x) < 4)["SUPER_CAT"].unique()
        return copied_df[~copied_df["SUPER_CAT"].isin(super_cats)]
    
    def generate_recommendations(self, df):
        """Generate item recommendations based on hierarchy."""
        copied_df = df.copy()
        recommendations = {}
        for item in df["ITEM_NAME"].unique():
            recommended_items = []
            for level in ["SEGMENT", "SUB_CAT", "CAT", "SUPER_CAT"]:
                category_value = copied_df[copied_df["ITEM_NAME"] == item].sort_values('TRANSACTIONS', ascending=False)[level].values[0]
                category_items = copied_df[copied_df[level] == category_value]["ITEM_NAME"].values
                category_items = [i for i in category_items if i != item]
                recommended_items.extend(category_items)
                if len(set(recommended_items)) >= 4:
                    break
            recommendations[item] = list(dict.fromkeys(recommended_items))[:4]
        return recommendations
    
    def compute_embeddings(self, df):
        """Compute embeddings for item names."""
        copied_df = df.copy()
        self.embeddings = self.model.encode(copied_df['ITEM_NAME_CLEANED'].tolist())
        return self.embeddings
    
    def get_similar_item(self, query, df, top_n=1):
        """Find the most similar item using cosine similarity."""
        copied_df = df.copy()
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)
        top_indices = similarities[0].argsort()[-top_n:][::-1]
        copied_df['COSINE_SIMILARITY'] = similarities[0]
        return copied_df.iloc[top_indices]
    
    def save_recommendations(self, recommendations, filename="recommendations.csv"):
        """Save recommendations to a CSV file."""
        recommendations_df = pd.DataFrame.from_dict(recommendations, orient='index', columns=[f"RECOMMENDATION {i+1}" for i in range(4)])
        recommendations_df = recommendations_df.reset_index().rename(columns={'index': 'ITEM_NAME'})
        recommendations_df.to_csv(self.data_dir / filename, index=False)
        print(f"\nRecommendations saved to {self.data_dir / filename}", "Number of items: ", len(recommendations_df))

if __name__ == "__main__":

    data_dir = Path.cwd() / "data"
    print("Loading data from: ", data_dir)

    items_file = "items.csv"
    model = 'sentence-transformers/all-mpnet-base-v2'



    RecommendationSystem = RecommendationSystem(data_dir, model)
    items_df = RecommendationSystem.load_data(items_file)
    print("Data loaded successfully")

    company_names = ['Stamford Street', 'JS ', 'SSTC', 'Stamford St', 'Sainsburys']

    items_cleaned_name_df = RecommendationSystem.clean_names(items_df, "ITEM_NAME", company_names)

    print("Cleaning item names successfully")

    items_with_low_super_cat = RecommendationSystem.get_low_super_cat_items(items_cleaned_name_df)
    items_with_high_super_cat = RecommendationSystem.get_high_super_cat_items(items_cleaned_name_df)

    print("Items found with low super categories: ", len(items_with_low_super_cat))

    item_with_four_recommendations_dict = RecommendationSystem.generate_recommendations(items_with_high_super_cat)

    print("Recommendations generated for items with high super categories")

    print("Using sentence-transformers model to compute embeddings")

    RecommendationSystem.compute_embeddings(items_with_high_super_cat)

    print("Embeddings computed successfully")
    print("Finding similar items for items with low super categories using AI")

    for item_for_swap in items_with_low_super_cat['ITEM_NAME_CLEANED'].values:
        query = item_for_swap
        recommendations = RecommendationSystem.get_similar_item(query, items_with_high_super_cat, top_n=1)
        top_cosine_item = recommendations['ITEM_NAME'].values[0]

        copied_recomendations = item_with_four_recommendations_dict[top_cosine_item]

        item_for_dict = items_with_low_super_cat[items_with_low_super_cat['ITEM_NAME_CLEANED'] == item_for_swap]['ITEM_NAME'].values[0]
        list_for_dict = copied_recomendations
        
        item_with_four_recommendations_dict[item_for_dict] = list_for_dict

        print("\nItem for swap: ", item_for_dict, 
            "\nTaking Recommendations From: ", top_cosine_item, 
            "\nRecommended Items: ", list_for_dict)
    
    print("Recommendations generated for items with low super categories")
        
    RecommendationSystem.save_recommendations(item_with_four_recommendations_dict, filename="recommendations_main.csv")
    

    





