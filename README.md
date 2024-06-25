# Recsys Challenge 2024 - Tom3TK

This GitHub repository is for Team Tom3TK's code submission in the Recsys2024 Challenge. All our execution scripts are located on Jupyter Notebook. Furthermore, the execution environment utilizes official Kaggle Docker.

## Overview of Our Approach

We adopted a conventional approach of training LightGBM with LambdaRank for the article re-ranking problem.

The feature generation was created from the following four perspectives:
1. **Article Freshness**: The time elapsed since the article was published at the inview moment. This was the most crucial feature in news recommendation.
2. **Leaky Popularity**: Created features based on how many times each article was inview during a certain time period and how often certain article combinations appear simultaneously in inviews, using behavior data. However, this includes leaked information not available in a production environment.
3. **Content-Based**: Generated latent representations for each user based on the articles they had previously viewed. We enhanced accuracy by using what is considered state-of-the-art at the time of the competition, **multilingual-e5-large**, rather than pre-provided latent representations of articles. We generated numerous user latent representations combining various information like scroll percentage, read time, and category, and used the cosine similarity between these representations and those of articles as features.
4. **Collaborative Filtering**: While less effective than in past competitions due to the significant impact of the cold start problem in news recommendation, we used Word2Vec to treat articles as words to obtain latent representations of articles based on articles previously viewed by the user. We generated features similar to point 3. Other features considered included association rule-based features using NER/topics and graph representations, but they were not effective.

For each feature, we enhanced accuracy by calculating not only the absolute level of the feature but also its rank within impressions to provide relative information to the model.

Training was performed using LightGBM with LambdaRank. Features were generated using all the data, and 20% of it was randomly selected as the training dataset. We created eight models using this method and combined them in an ensemble.


## Directory Structure

- 0.precompute
  - a. get_article_embedding.ipynb
    - Utilizes 'multilingual-e5-large' to infer latent representations of articles.
  - b. item2vec.ipynb
    - Treats articles appearing in history as words, and employs word2vec to learn and infer latent representations of articles.
  - c. inview_occur.ipynb
    - Calculates combinations of articles that appear simultaneously in-view and counts them.
  - d. article_pop_inview.ipynb
    - Computes the count of each article appearing in-view over time.

- 1.feature engineering
- 2.train/inference


## How to Reproduce

- Run the notebooks under the "precompute" directory. Save the computed results at whereever you want. We have been using /home/data for this purpose.
- Run the feature engineering notebook. Ensure the loading directory is set correctly.
- Run the training and inference notebook.
