#!/usr/bin/env python3
"""
Create a sample news dataset for demonstration purposes
This simulates the real Kaggle News Category Dataset structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_news_dataset(num_articles=1000):
    """Create a sample news dataset with realistic content"""

    # Define categories and their characteristics
    categories = {
        'POLITICS': {
            'keywords': ['president', 'government', 'election', 'policy', 'congress', 'senate', 'vote', 'campaign', 'democrat', 'republican', 'bill', 'law', 'political', 'administration'],
        'templates': [
            "{} announces new policy",
            "Government passes {} legislation",
            "{} election results show change",
            "Congress debates {} bill",
            "Administration announces new regulations"
        ]
        },
        'SPORTS': {
            'keywords': ['team', 'game', 'player', 'championship', 'score', 'win', 'lose', 'season', 'league', 'tournament', 'athlete', 'coach', 'stadium', 'match'],
        'templates': [
            "{} defeats opponent in championship game",
            "{} wins major championship",
            "{} player scores record points",
            "{} team advances to finals",
            "Season ends with victory"
        ]
        },
        'TECH': {
            'keywords': ['technology', 'software', 'app', 'digital', 'computer', 'internet', 'data', 'artificial intelligence', 'machine learning', 'startup', 'innovation', 'cyber', 'platform'],
        'templates': [
            "New {} technology launched",
            "{} company launches mobile app",
            "{} software adds new features",
            "{} startup raises millions",
            "Platform gains new users"
        ]
        },
        'BUSINESS': {
            'keywords': ['business', 'company', 'market', 'stock', 'economy', 'financial', 'revenue', 'profit', 'investment', 'banking', 'trade', 'industry', 'corporate', 'merger'],
        'templates': [
            "{} company reports strong earnings",
            "Stock market rises on positive news",
            "{} industry shows growth",
            "Major merger creates new company",
            "Business announces expansion"
        ]
        },
        'ENTERTAINMENT': {
            'keywords': ['movie', 'film', 'actor', 'celebrity', 'music', 'album', 'concert', 'show', 'television', 'streaming', 'award', 'premiere', 'box office', 'entertainment'],
        'templates': [
            "{} movie breaks box office records",
            "{} actor wins major award",
            "{} music releases new album",
            "{} show announces premiere",
            "Entertainment industry shows success"
        ]
        },
        'SCIENCE': {
            'keywords': ['research', 'study', 'scientist', 'discovery', 'experiment', 'laboratory', 'university', 'medical', 'health', 'disease', 'treatment', 'cure', 'breakthrough', 'findings'],
        'templates': [
            "Scientists make new discovery",
            "Research shows breakthrough in medicine",
            "{} study reveals findings",
            "Medical research develops new treatment",
            "Research shows positive results"
        ]
        },
        'WORLD NEWS': {
            'keywords': ['international', 'global', 'world', 'country', 'nation', 'summit', 'treaty', 'agreement', 'crisis', 'conflict', 'peace', 'diplomacy', 'foreign', 'ambassador'],
        'templates': [
            "International summit concludes",
            "{} country signs agreement",
            "Global crisis requires response",
            "World leaders meet for talks",
            "Nation announces new response"
        ]
        },
        'CRIME': {
            'keywords': ['police', 'arrest', 'crime', 'suspect', 'investigation', 'court', 'trial', 'jury', 'verdict', 'sentence', 'prison', 'jail', 'criminal', 'justice'],
        'templates': [
            "Police arrest suspect in case",
            "Investigation leads to arrest",
            "Court announces verdict in trial",
            "Crime investigation continues",
            "Justice served for victims"
        ]
        },
        'FOOD & DRINK': {
            'keywords': ['restaurant', 'food', 'chef', 'cuisine', 'recipe', 'cooking', 'kitchen', 'dining', 'meal', 'taste', 'flavor', 'ingredient', 'cookbook', 'culinary'],
        'templates': [
            "{} restaurant launches new menu",
            "Chef wins award for cuisine",
            "{} cuisine gains popularity",
            "New recipe becomes popular",
            "Food trend emerges in market"
        ]
        },
        'STYLE & BEAUTY': {
            'keywords': ['fashion', 'style', 'beauty', 'designer', 'clothing', 'makeup', 'cosmetics', 'trend', 'model', 'runway', 'collection', 'brand', 'outfit', 'accessories'],
        'templates': [
            "{} fashion sets new trend",
            "Designer launches collection",
            "{} beauty releases product",
            "Style influences new generation",
            "Brand announces new launch"
        ]
        }
    }

    # Additional words for variety
    additional_words = [
        'major', 'new', 'latest', 'recent', 'significant', 'important', 'breaking', 'exclusive',
        'report', 'announces', 'reveals', 'confirms', 'denies', 'responds', 'reacts', 'comments',
        'today', 'yesterday', 'this week', 'this month', 'this year', 'recently', 'soon', 'expected'
    ]

    # Generate articles
    articles = []

    for i in range(num_articles):
        # Select random category
        category = random.choice(list(categories.keys()))
        cat_info = categories[category]

        # Generate description
        template = random.choice(cat_info['templates'])
        keyword1 = random.choice(cat_info['keywords'])
        keyword2 = random.choice(cat_info['keywords'])
        additional = random.choice(additional_words)

        # Create description
        description = template.format(keyword1, keyword2)
        if random.random() > 0.5:
            description = f"{additional.title()} {description.lower()}"

        # Add some variety
        if random.random() > 0.7:
            description += f" {random.choice(cat_info['keywords'])}"

        # Create article
        article = {
            'short_description': description,
            'category': category,
            'headline': f"{description.title()} - Breaking News",
            'authors': f"Staff Writer {random.randint(1, 50)}",
            'link': f"https://news.example.com/article/{i+1}",
            'date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
            'text': f"{description}. This is a detailed news article about {keyword1} and {keyword2}. The story continues with additional information and context about the recent developments in this area."
        }

        articles.append(article)

    # Create DataFrame
    df = pd.DataFrame(articles)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df

def main():
    print("🚀 CREATING SAMPLE NEWS DATASET")
    print("=" * 50)

    # Create dataset
    print("Generating sample news articles...")
    df = create_sample_news_dataset(num_articles=1000)

    # Save to CSV
    filename = "News_Category_Dataset_v3.csv"
    df.to_csv(filename, index=False)

    print(f"✅ Dataset created: {filename}")
    print(f"   Articles: {len(df)}")
    print(f"   Categories: {len(df['category'].unique())}")
    print(f"   File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Show sample
    print(f"\nSample articles:")
    for i, row in df.head(3).iterrows():
        print(f"  {i+1}. [{row['category']}] {row['short_description']}")

    print(f"\nCategory distribution:")
    print(df['category'].value_counts())

    print(f"\n✅ Sample dataset ready for testing!")
    print(f"   File: {filename}")
    print(f"   Ready to run: python main_pipeline.py")

if __name__ == "__main__":
    main()
