# Quick and Delicious: Does Cooking Time Predict Recipe Rating?

Final Project for the UCSD's DSC80 Course

Authors: Jeremy Cheng and Cedric Jeng

## Overview

Our data science project explores the relationship between the cooking time and rating of a given recipe.

## Introduction

As college students living on campus, finding good food is more challenging than it seems. Dining hall meals often fall short, and the fast-paced quarter system leaves little time for cooking, especially during exam-filled weeks. This makes us gravitate toward recipes that are both quick and satisfying. Motivated by this, we set out to explore whether shorter cooking times are linked to higher recipe ratings. Using two datasets from [food.com](https://www.food.com/), one containing over 80,000 recipes and the other with user reviews submitted since 2008, we aim to answer the question: Is cooking time a strong predictor of user satisfaction, as measured by average recipe ratings?

`'recipe'`, the first dataset, consist of 83782 rows (83782 unique recipes) and 12 columns, each representing information on the recipe:

| **Column**         |                                                                                                                                              **Description** |
| :----------------- | -----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| `'name'`           |                                                                                                                                                  recipe name |
| `'id'`             |                                                                                                                                                    recipe ID |
| `'minutes'`        |                                                                                                                                       minutes to cook recipe |
| `'contributor_id'` |                                                                                                                              user ID of who submitted recipe |
| `'submitted'`      |                                                                                                                                    date recipe was submitted |
| `'tags'`           |                                                                                                                                                food.com tags |
| `'nutrition'`      | `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV means "percentage of daily value" |
| `'n_steps'`        |                                                                                                                                    number of steps in recipe |
| `'steps'`          |                                                                                                                               text for in order recipe steps |
| `'description'`    |                                                                                                                                 recipe description from user |
| `'ingredients'`    |                                                                                                                                           recipe ingredients |
| `'n_ingredients'`  |                                                                                                                              number of ingredients in recipe |

`'interactions'`, our second dataset, consist of 731927 rows (731927 unique reviews) and 5 columns, each representing information on the review:

| **Column**    |           **Description** |
| :------------ | ------------------------: |
| `'user_id'`   |       user ID of reviewer |
| `'recipe_id'` |              ID of recipe |
| `'date'`      | date review was submitted |
| `'rating'`    |          rating of recipe |
| `'review'`    |        text of the review |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

In order to properly analyze and prepare the datasets, we performed the following steps:

1. Replaced ratings of 0 in the interactions dataset with np.nan.

   - Ratings on food.com range from 1 to 5, so a rating of 0 is invalid and likely indicates that the user didn't rate the recipe.
   - Replacing them with `np.nan` ensures they are excluded from the average rating calculation, which is how pandas handles missing values by default.

2. Computed the average rating per recipe of the interactions dataset.

   - We are primarily interested in how well each recipe is rated, rather than individual user reviews.
   - Grouping by `'recipe_id'` and computing the mean yields a single, representative value for each recipe's overall rating.

3. Mapped the average rating per recipe to the raw_recipes dataset
   - Instead of fully merging the datasets, we extracted only the information we care about (the average rating) and added it directly to the main recipes dataset.
   - This preserved all unique recipes in `raw_recipes` and avoided duplicate rows that would result from merging on one-to-many relationships.
   - Recipes with no valid ratings (either never rated or all ratings were 0) had their average rating set to `np.nan`

#### Result

These are the columns of our cleaned dataframe:

| **Column**            | **Type** |
| :-------------------- | -------: |
| `'name'`              |   object |
| `'id'`                |    int64 |
| `'minutes'`           |    int64 |
| `'contributor_id'`    |    int64 |
| `'submitted'`         |   object |
| `'tags'`              |   object |
| `'nutrition'`         |   object |
| `'n_steps'`           |    int64 |
| `'steps'`             |   object |
| `'description'`       |   object |
| `'ingredients'`       |   object |
| `'n_ingredients'`     |    int64 |
| `'avg_recipe_rating'` |  float64 |

And below is the head of our dataframe with only columns needed for our project:

|     | name                               |     id | minutes | tags                                                                                                                                                                                                                                                                                               | nutrition                                     | n_steps | n_ingredients | avg_recipe_rating |
| --: | :--------------------------------- | -----: | ------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- | ------: | ------------: | ----------------: |
|   0 | 1 brownies in the world best ever  | 333281 |      40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]      |      10 |             9 |                 4 |
|   1 | 1 in canada chocolate chip cookies | 453467 |      45 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0]  |      12 |            11 |                 5 |
|   2 | 412 broccoli casserole             | 306168 |      40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]     |       6 |             9 |                 5 |
|   3 | millionaire pound cake             | 286009 |     120 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] | [878.3, 63.0, 326.0, 13.0, 20.0, 123.0, 39.0] |       7 |             7 |                 5 |
|   4 | 2000 meatloaf                      | 475785 |      90 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             | [267.0, 30.0, 12.0, 12.0, 29.0, 48.0, 2.0]    |      17 |            13 |                 5 |

Our cleaned dataframe consist of 83782 rows × 9 columns.

---

### Exploratory Data Analysis

#### Univariate Analysis

We first did univariate analysis to see the distribution of our single variables.

Below is the distribution of cook times in minutes less than or equal to 250 minutes. Initally we plotted our entire minutes column but because of extreme outliers, it showed one large column. Thus we filtered it to `'<= 250'` minutes to see our trends better:

<iframe 
    src="graphs/fig_1.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>
As we can see in the histogram above, our cooking times show a right-skewed distribution. This means most of our filtered recipes hover around the lower end of our scale of 0 to 250 minutes.

Furthermore, the right-skeweness of the distribution suggests that logarithmic transformation of our minutes column could help reduce skewness and improve symmetry for better analysis:

<iframe 
    src="graphs/fig_2.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>
The resulting distribution is much more symmetric, with the center now around 3.75 on the log scale (corresponds to approximately 43 minutes on the original scale). This transformation reduces the influence of extreme cooking times and helps make the data more suitable for statistical modeling by improving normality and stabilizing variance across observations.

Next we wanted to look at the distribution of average recipe ratings:

<iframe 
    src="graphs/fig_3.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>
We can see that it's heavily skewed to the left with average ratings of 5 being by far the most common as it accounts for what looks like over 52,000 recipes. The next most frequent ratings are 4.0 and 4.5, with approximately 13,000 and 10,000 recipes respectively. Ratings below 4.0 are relatively rare, with fewer than 3,000 recipes for any given lower rating.This pattern indicates a strong positive bias in the dataset, which is common in online ratings systems where users are more likely to leave feedback only when they are especially satisfied.

#### Bivariate Analysis

Here we want to examine cooking time split into bins against average recipe rating:

<iframe 
    src="graphs/fig_4.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>

Above we see a faint trend that ratings decrease through each time bin. In order to confirm this we want to see the mean rating for each time bin:

| time_bin   | avg_recipe_rating |
| :--------- | ----------------: |
| 0-10 min   |           4.68662 |
| 10-30 min  |           4.63837 |
| 30-60 min  |           4.61215 |
| 60-120 min |            4.6205 |
| 120+ min   |           4.59511 |

There's not a significant trend and the `'60-120 min'` bin goes against this trend as it's mean average recipe rating is greater than the bin before it. This made us curious about the distribution of recipe counts per time bin so we visualized that below:

<iframe 
    src="graphs/fig_7.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>

Next we want to see the relationship between filtered cooking time (times <= 250) and average recipe rating:

<iframe 
    src="graphs/fig_5.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>
We see that a large concentration of recipes are 4 to 5 stars, with a even higher concentration around the lower end of the filtered cooking time scale. There doesn’t appear to be a strong or clear correlation between cooking time and average rating—recipes with short and long cooking times can receive both high and low ratings.

We also want to see the relationship between log cooking time and average recipe rating:

<iframe 
    src="graphs/fig_6.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>
Taking the log of cooking time helps to compress the wide range of values and highlight patterns that may be harder to see on a linear scale. Similar to the linear version, most recipes tend to have high ratings (4 or 5 stars), regardless of cooking time. Overall, even with log transformation, there still doesn’t appear to be a strong or consistent relationship between cooking time and average recipe rating. Most recipes, regardless of how long they take to cook, are still rated quite highly.

Finally, we want to see what the mean average recipe rating looks like for quick (1) and non-quick (0) recipes:

| is_quick |    mean | count |
| -------: | ------: | ----: |
|        0 | 4.60937 | 44108 |
|        1 | 4.64439 | 37065 |

The table above shows the mean average rating and the number of recipes in each category. Quick recipes have a slightly higher average rating of 4.64 compared to 4.61 for non-quick recipes. Despite the small difference, both types of recipes tend to receive high ratings overall. The dataset includes a substantial number of both quick (37,065) and non-quick (44,108) recipes, so the comparison is reasonably balanced. This suggests that quicker recipes may be slightly more favored by users, though the difference in mean ratings is relatively small.

## Assessment of Missingness

### NMAR Analysis

We believe that the missingness in the `'description'` column is Not Missing At Random (NMAR). This is because the likelihood of a description being provided likely depends on unobserved factors, such as the contributor’s effort, writing motivation, or belief in the recipe’s uniqueness or quality.

To convert this from NMAR to Missing At Random (MAR), we would need access to additional variables, such as contributor behavior or user profile information (e.g., how many recipes the user has uploaded, average engagement with their recipes, or whether they are verified contributors). These variables might explain the missingness in `'description'`, helping us determine if the absence of a description is related to something observable rather than unobservable.

### Missingness Dependency

To explore whether the missingness of `'description'` might be dependent on other observable columns, we conducted permutation tests with the following variables: `'minutes'`, `'avg_recipe_rating'`, and `'few_ingredients'`.

### Minutes and Description

Null Hypothesis: The missingness of 'description' does not depend on the cooking time of the recipe ('minutes').

Alternative Hypothesis: The missingness of 'description' does depend on the cooking time of the recipe.

Test Statistic: The absolute difference in the mean number of minutes between recipes with and without a description.

Significance Level: 0.05

After running a permutation test with 1000 shuffles, we obtained a p-value of 1.0, which is much greater than 0.05.

Conclusion: We fail to reject the null hypothesis. The missingness of 'description' does not depend on the cooking time in minutes.

<iframe 
    src="graphs/fig_8.html" 
    width="800"
    height="600"
    frameborder="0"
></iframe>

### Average Rating and Description

Null Hypothesis: The missingness of 'description' does not depend on the average rating of a recipe ('avg_recipe_rating').

Alternative Hypothesis: The missingness of 'description' does depend on the average rating of a recipe.

Test Statistic: The absolute difference in the mean average rating between recipes with and without a description.

Significance Level: 0.05

From the permutation test, we obtained a p-value of 0.45, which is greater than the significance level.
Conclusion: We fail to reject the null hypothesis. The missingness of 'description' does not depend on a recipe's average rating.

<iframe 
    src="graphs/fig_9.html" 
    width="800"
    height="600"
    frameborder="0"
></iframe>

### Number of Ingredients and Description

Null Hypothesis: The missingness of 'description' does not depend on the number of ingredients in the recipe.

Alternative Hypothesis: The missingness of 'description' does depend on the number of ingredients in the recipe.

Test Statistic: The absolute difference in the proportion of missing descriptions between recipes with fewer ingredients and those with more.

Significance Level: 0.05

We created a binary column 'few_ingredients' to indicate whether a recipe has fewer than or equal to the median number of ingredients.

The permutation test produced a p-value of 0.011, which is less than the significance level.
Conclusion: We reject the null hypothesis. The missingness of 'description' does depend on the number of ingredients in the recipe. Recipes with fewer ingredients are more likely to have missing descriptions, possibly because they are simpler and don’t require much explanation.

<iframe 
    src="graphs/fig_11.html" 
    width="800"
    height="600"
    frameborder="0"
></iframe>

## Hypothesis Testing

Null Hypothesis:
Cooking time has no association with average recipe rating.

Alternative Hypothesis:
Cooking time is associated with average recipe rating.

Test Statistic:
We used Pearson’s correlation coefficient (r) to quantify the linear relationship between cooking time and average recipe rating. This is an appropriate choice because both variables are continuous, and we are specifically interested in the strength and direction of their linear relationship.

Significance Level: 0.05

We chose to run a permutation test because we do not have information about the population distribution of recipe ratings or cooking times, and we want to determine whether the observed association between them could have occurred by chance. Our hypothesis is that there is a relationship between how long a recipe takes to cook and how highly it is rated. Since we’re interested in how these two continuous variables relate, we used Pearson’s correlation coefficient as our test statistic. To make the test robust to distributional assumptions, we used a permutation-based approach. This helps us simulate what the distribution of correlation values might look like under the null hypothesis that cooking time and rating are not related.

To run the test, we computed the observed Pearson correlation between cooking time and average recipe rating. Our results were **approximately 0.035**. This indicates a weak positive relationship. We then shuffled the ratings 1,000 times, recalculating the correlation each time to generate a null distribution. The number of times the permuted correlation was as extreme or more extreme than the observed one gave us a **p-value of 0.001**.

We also calculated the average recipe ratings for each group:
Quick recipes (≤ median time): 4.644
Non quick recipes (> median time): 4.609
Mean difference: 0.035

<iframe 
    src="graphs/fig_12.html" 
    width="800"
    height="600"
    frameborder="0"
></iframe>

<iframe 
    src="graphs/fig_13.html" 
    width="800"
    height="600"
    frameborder="0"
></iframe>

Since the p-value we obtained (0.001) is lower than the significance level of 0.05, we reject the null hypothesis. This suggests that cooking time and recipe rating are not independent. While the observed effect is statistically significant, the actual difference in mean ratings is quite small, which indicates a weak relationship.

## Framing a Prediction Problem

We aim to predict whether a user will enjoy a recipe based on its features through binary classification. This is because our response variable is a binary column called `'enjoy'`, which is 1 if a recipe’s average rating is greater than or equal to the median rating across all recipes, and 0 otherwise. This choice allows us to frame the prediction task as identifying “liked” vs. “not liked” recipes using user-provided ratings. This makes the task a binary classification problem with the model’s job being to assign a new recipe to one of two categories: liked or not liked. We use only information that would be available at the time of recipe publication to simulate a real-world use case such as suggesting new recipes to users.
<br>I need to explain metrics and justify

## Baseline Model

Our goal was to build a predictive model that classifies whether a user is likely to enjoy a recipe. We defined the target variable `'enjoy'` as a binary outcome: recipes with an average rating above or equal to the median rating were labeled as 1 (enjoyed), and those below the median were labeled as 0 (not enjoyed).

We trained a logistic regression model using these three numerical features:

- `'minutes'`: the cooking time of the recipe
- `'n_steps'`: the number of cooking steps in the recipe
- `'n_ingredients'`: the number of ingredients used in the recipe

All three of these features are quantitative. There were no ordinal or nominal (categorical) variables in this baseline model, so no encodings such as one-hot or ordinal encoding were necessary.

Before training the model, we standardized all numeric features using StandardScaler within a ColumnTransformer to ensure the model treats all features on the same scale. The full preprocessing and model training pipeline was implemented using scikit-learn’s Pipeline functionality.

We evaluated the model using a stratified 80/20 train-test split to preserve the distribution of the target variable. The `'enjoy'` variable was slightly imbalanced, with approximately:
**59% of samples labeled as enjoyed (1)** and **41% labeled as not enjoyed (0)**. The performance metrics on the test set were:

- **Accuracy: 0.589**
- **F1 Score: 0.741**

The model achieves an accuracy of approximately 58.9%, which is only slightly better than always predicting the majority class (enjoyed or 1). Meaning our baseline model isn’t adding much predictive value in terms of accuracy. However, our F1 Score of 0.741 indicates that the model is capturing patterns that allow it to distinguish the two classes. So while our accuracy is low, our high F1 suggest that the model is better at identifying minority class than accuracy alone indicates.

## Final Model

### Feature Engineering

To improve our prediction of whether a recipe would be `'enjoyed'` (rated at or above the median), we engineered a set of features grounded in the data-generating process and domain logic:

| **Feature**                           | **Description**                                                                                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `protein_to_fat`, `sugar_to_calories` | Nutritional ratios capturing relative nutrient densities. Users may prefer recipes with high protein-to-fat ratios or lower sugar relative to calories.              |
| `is_quick`                            | Indicates whether a recipe is quick to prepare. May influence user ratings due to convenience. Not used directly in the final model but informed feature design.     |
| `tag_` features                       | One-hot encoded tags like “easy”, “vegan”, or “holiday” provide semantic context about recipe type, complexity, or occasion—often aligned with user preferences.     |
| `n_ratings`, `rating_std`             | Aggregated rating metrics. `n_ratings` reflects popularity; `rating_std` captures variability. High count and low variability suggest reliability.                   |
| `encoded_contributor`                 | Encodes the historical performance of contributors using Bayesian shrinkage. Contributors with strong track records are more likely to produce highly-rated recipes. |
| `has_<ingredient>`                    | Binary indicators for the top 10 most common ingredients (e.g., garlic, onion, butter). These capture taste preferences or user familiarity with ingredients.        |

These features were selected to reflect meaningful patterns in how users might evaluate recipes, not based solely on model performance.

### Model and Hyperparameter Selection

We chose a Random Forest Classifier as the final model due to its ability to capture nonlinear relationships, natural handling of categorical/binary variables, and built-in feature importance.

We used a grid search with 3-fold cross-validation to tune the following hyperparameters:

- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of each tree
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node

The best performing combination from our grid was selected based on the F1 score, as it balances precision and recall, which is appropriate for slightly imbalanced classes.

### Model Performance

Compared to our baseline logistic regression model, which achieved:

- **Accuracy: 0.589**
- **F1 Score: 0.741**

Our final Random Forest model achieved:

- **Test Accuracy: 0.844**
- **Test F1 Score: 0.875**

This is a substantial improvement, suggesting that the engineered features and nonlinear modeling approach better captured the complexity of what makes a recipe enjoyable.

The performance boost of our final model validates the added value of our feature engineering. By incorporating metadata (tags), user behavior (ratings), and content-based features (ingredients and nutrition), we successfully built a model that closely mimics the nuanced decision process behind user ratings.

## Fairness Analysis

For this fairness test, we compared model performance across two groups:

- Group X (Quick Recipes): Recipes with cooking time below the median (`is_quick == 1`)
- Group Y (Not Quick Recipes): Recipes with cooking time above the median (`is_quick == 0`)

These groups reflect differences in user time constraints, and we want to ensure our model performs similarly across them.

### Evaluation Metric

We used F1 Score as our evaluation metric because it balances precision and recall, which is appropriate given the class imbalance in the target variable.

### Hypotheses

Null Hypothesis (H₀): The model is fair with respect to cooking time. Any difference in F1 scores between the two groups is due to random chance.

Alternative Hypothesis (H₁): The model is unfair with respect to cooking time. The F1 score differs significantly between quick and not-quick recipes.

### Test Statistic

We used the difference in F1 scores between the two groups as our test statistic:
Test Statistic = F1<sub>quick</sub> − F1<sub>not quick</sub>

### Permutation Test

We ran a permutation test with 1,000 iterations, randomly shuffling the `'is_quick'` labels and computing the difference in F1 scores each time. This generated a null distribution of F1 differences under the assumption of fairness.

The observed difference in F1 score was:

- F1_quick = 0.8788
- F1_not_quick = 0.8701
- Observed difference = 0.0087

The resulting p-value was **0.236**, which is greater than our significance threshold of α = 0.05.

<iframe 
    src="graphs/fig_14.html" 
    width="800"
    height="600"
    frameborder="0"
></iframe>

### Conclusion

The observed difference in F1 score between the two groups was not statistically significant (p_val > 0.05). Thus, we fail to reject the null hypothesis and conclude that our model does include evidence of unfairness with respect to cooking time.
