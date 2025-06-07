# Quick and Delicious: Does Cooking Time Predict Recipe Rating?

Final Project for the UCSD's DSC80 Course

Authors: Jeremy Cheng and Cedric Jeng

## Overview

Our data science project explores the relationship between the cooking time and rating of a given recipe.

## Introduction

As college students living on campus, finding good food is more challenging than it seems. Dining hall meals often fall short, and the fast-paced quarter system leaves little time for cooking, especially during exam-filled weeks. This makes us gravitate toward recipes that are both quick and satisfying. Motivated by this, we set out to explore whether shorter cooking times are linked to higher recipe ratings. Using two datasets from [food.com](https://www.food.com/), one containing over 80,000 recipes and the other with user reviews submitted since 2008, we aim to answer the question: Is cooking time a strong predictor of user satisfaction, as measured by average recipe ratings?

`'recipe'`, the first dataset, consists of 83782 rows (83782 unique recipes) and 12 columns, each representing information on the recipe:

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

`'interactions'`, our second dataset, consists of 731927 rows (731927 unique reviews) and 5 columns, each representing information on the review:

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

1. Left merge the recipe and interaction datasets.
   - This ensured that every recipe from raw_recipes was preserved, even if it didn’t have any user interactions.
   - The result was a single dataframe (`merged`) containing both recipe information and associated user ratings where available.

2. Replaced invalid ratings with `NaN`.
   - We replaced all ratings of 0 in the rating column with `np.nan`.
   - Food.com ratings range from 1 to 5, so a rating of 0 is not valid and likely indicates the user didn’t submit a rating.
   - Replacing these with `NaN` ensures that they are excluded from average rating calculations.
  
3. Computed the average rating per recipe.
   - We grouped the merged dataset by id (`'recipe ID'`) and calculated the mean rating for each recipe, ignoring any `NaN` values.
   - This gave us a single average rating for each recipe that reflects its overall reception by users.
   
4. Mapped the average rating to the recipes dataset.
   - We created a new copy of the original raw_recipes dataframe and added a new column called `'avg_recipe_rating'`, which contains the average ratings we just computed.
   - This was done using the `.map()` function to match each recipe's ID with its corresponding average rating.
   - Recipes with no valid ratings (no interactions or only invalid ratings) received a value of `NaN` in this column, indicating missing data. 

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

Our cleaned dataframe consists of 83782 rows × 8 columns.

---

### Exploratory Data Analysis

#### Univariate Analysis

We began our univariate analysis by examining the distribution of cooking times. Initially, we plotted all values in the minutes column, but extreme outliers distorted the visualization, producing a single column for all values. To better observe the data, we filtered the data to include only recipes with cooking times of 250 minutes or less. We chose 250 minutes (just over 4 hours) as a reasonable upper limit because recipes beyond that length are not as common as shorter ones and typically not practical for everyday cooking.

Below is the histogram of the filtered cooking times:

<iframe 
    src="graphs/fig_1.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>
As shown in the histogram above, the distribution of cooking times is heavily right-skewed. Most recipes fall within the lower end of the 0 to 250-minutes range, with only a small number of recipes requiring significantly longer cooking times.

Furthermore, the right-skewness of the distribution suggests that applying a logarithmic transformation to the minutes column could reduce skewness and improve symmetry, making the data more suitable for analysis.

<iframe 
    src="graphs/fig_2.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>

After applying the log transformation to the '`minutes`' column, the distribution becomes much more symmetric. The center of the distribution shifts to around 3.75 on the log scale, which corresponds to roughly 43 minutes on the original scale. This transformation removes the influence of extreme values, making the data more suitable for statistical modeling by improving normality and stabilizing variance across observations.

Next we examined the distribution of average recipe ratings:

<iframe 
    src="graphs/fig_3.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>

As shown in the histogram, the distribution is heavily left-skewed, with a concentration of recipes rated a perfect 5.0. Over 52,000 recipes recipes received this top rating, while the next most common ratings, 4.0 and 4.5, trail significantly, with around 13,000 and 10,000 recipes, respectively. Ratings below 4.0 are extremely rare, each appearing fewer than 3,000 times. This distribution reflects a strong positive bias, which is typical in online review platforms. Users are more likely to submit ratings when they've had a particularly positive experience, inflating ratings.

#### Bivariate Analysis

Here, we examine how average recipe ratings vary across different cooking time bins:

<iframe 
    src="graphs/fig_4.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>

The boxplot above suggests a faint trend: recipes with shorter cooking times tend to have slightly higher ratings. To confirm this trend, we computed the mean average rating for each time bin:

| time_bin   | avg_recipe_rating |
| :--------- | ----------------: |
| 0-10 min   |           4.68662 |
| 10-30 min  |           4.63837 |
| 30-60 min  |           4.61215 |
| 60-120 min |            4.6205 |
| 120+ min   |           4.59511 |

While there appears to be a slight downward trend in ratings as cooking time increases, it's not particularly strong, The `'60-120 min'` bin slightly breaks the pattern, with a higher mean rating than the `'30-60 min'` bin. To investigate further, we examined how many recipes fall into each cooking time bin:

<iframe 
    src="graphs/fig_7.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>

The bar chart above shows that the `'30-60 min'` bin contains the highest number of recipes, followed by the `'10-30 min'` bin. In contrast, both very short (0-10 minutes) and very long (120+ min) recipes are less common. This uneven distribution could partially explain the irregularity in the average ratings, as bins with fewer recipes may have more volatile averages due to smaller sample sizes.

Next, we examined the relationship between filtered cooking time (<= 250 minutes) and average recipe rating:

<iframe 
    src="graphs/fig_5.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>

The scatterplot above shows a strong concentration of recipes with ratings between 4 and 5 stars, particularly clustered at the lower end of the cooking time axis. However, there is no clear or strong correlation between cooking time and average rating; recipes with both short and long cook times span the full range of ratings. This suggests that cooking time alone is not a strong predictor of how users rate a recipe. While quicker recipes are slightly more frequent and often highly rated, long recipes can also receive top scores, and short ones can be rated poorly.

We also examined the relationship between log-transformed cooking time and average recipe rating:

<iframe 
    src="graphs/fig_6.html" 
    width="900"
    height="600"
    frameborder="0"
></iframe>

Taking the logarithm of cooking times compresses the wide range of values and helps highlight patterns that may be harder to identify on a linear scale. Similar to the original scatter plot, most recipes cluster between 4 and 5 stars, regardless of cooking time. Overall, even after log transformation, there is still no strong or consistent relationship between cooking time and average recipe rating. Most recipes, whether quick or time-intensive, are rated highly.

Finally, we explored how average ratings differ between quick (1) and non-quick (0) recipes:

| is_quick |    mean | count |
| -------: | ------: | ----: |
|        0 | 4.60937 | 44108 |
|        1 | 4.64439 | 37065 |

The table above shows the mean average rating and the number of recipes in each category. Quick recipes have a slightly higher average (4.64) compared to non-quick recipes (4.61). Despite this small difference, both groups tend to receive high ratings overall. The dataset contains a substantial number of both quick (37,065) and non-quick (44,108) recipes, making the comparison reasonably balanced. This suggests that quicker recipes may be slightly more favored by users, though the difference in average ratings is relatively minor.

## Assessment of Missingness

### NMAR Analysis

We believe that the missingness in the `description` column is Not Missing At Random (NMAR). This is because whether a recipe includes a description likely depends on the unobserved factors such as the effort or motivation of the person submitting it, or their confidence in the recipe's uniqueness or quality.

To convert this from NMAR to Missing At Random (MAR), we would need access to additional variables related to contributor behavior or user profile information(e.g. how many recipes a user has submitted, how often others engage with their content, or whether they are verified contributors). These variables could help explain the missingness in `description` and allow us to determine whether the absence of a description is linked to something observable rather than unobservable.

### Missingness Dependency

To explore whether the missingness of `description` might be dependent on other observable columns, we conducted permutation tests with the following variables: `'minutes'`, `'avg_recipe_rating'`, and `'few_ingredients'`.

### Minutes and Description

Null Hypothesis: The missingness of '`description`' does not depend on the cooking time of the recipe ('minutes').

Alternative Hypothesis: The missingness of '`description`' does depend on the cooking time of the recipe.

Test Statistic: The absolute difference in the mean number of minutes between recipes with and without a description.

Significance Level: 0.05

After running a permutation test with 1000 shuffles, we obtained a p-value of 1.0, which is much greater than 0.05.

Conclusion: We fail to reject the null hypothesis. The missingness of `description` does not appear to depend on cooking time.

<iframe 
    src="graphs/fig_8.html" 
    width="800"
    height="600"
    frameborder="0"
></iframe>

The smoothed density plot above supports this conclusion, showing little difference between the distributions of cooking time for recipes with and without a description. Both curves follow a similar right-skewed shape, indicating that cooking duration likely plays no role in whether a description is included.

### Average Rating and Description

Null Hypothesis: The missingness of '`description`' does not depend on the average rating of a recipe ('avg_recipe_rating').

Alternative Hypothesis: The missingness of '`description`' does depend on the average rating of a recipe.

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

The smoothed density plot supports this conclusion: the distributions of average ratings for recipes with and without descriptions are quite similar. Although the blue curve (not missing) shows a slightly sharper peak near 5.0, the overall shape of both distributions indicates that the presence or absence of a description is not strongly associated with how highly a recipe is rated.

### Number of Ingredients and Description

Null Hypothesis: The missingness of `'description'` does not depend on the number of ingredients in the recipe.

Alternative Hypothesis: The missingness of `'description'` does depend on the number of ingredients in the recipe.

Test Statistic: The absolute difference in the proportion of missing descriptions between recipes with fewer ingredients and those with more.

Significance Level: 0.05

We created a binary column `'few_ingredients'` to indicate whether a recipe has fewer than or equal to the median number of ingredients.

The permutation test produced a p-value of 0.011, which is less than the significance level.

Conclusion: We reject the null hypothesis. The missingness of `'description'` does depend on the number of ingredients in the recipe. Recipes with fewer ingredients are more likely to have missing descriptions, possibly because they are simpler and don’t require much explanation.

<iframe 
    src="graphs/fig_11.html" 
    width="800"
    height="600"
    frameborder="0"
></iframe>

The bar chart supports this conclusion, showing a higher proportion of missing descriptions among recipes with fewer ingredients (those below or equal to the median). The difference, though small, is statistically significant, indicating that ingredient count is an observable factor influencing description presence.

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

We aim to predict whether a user will enjoy a recipe based on its features through binary classification. This is because our response variable is a binary column called `'enjoy'`, which is 1 if a recipe’s average rating is greater than or equal to the median rating across all recipes, and 0 otherwise. This choice allows us to frame the prediction task as identifying “liked” vs. “not liked” recipes using user-provided ratings. This makes the task a binary classification problem with the model’s job being to assign a new recipe to one of two categories: liked or not liked.

To evaluate our model’s performance, we used two metrics: accuracy and F1 score. Accuracy measures the overall proportion of correct predictions and is useful as a baseline when the classes are relatively balanced. However, since the two classes are not perfectly balanced (e.g., around 59% of the recipes are labeled as "enjoyed"), accuracy alone can be misleading. For this reason, we also report the F1 score, which is the harmonic mean of precision and recall. F1 is especially useful when we care about both false positives and false negatives, which is relevant in our case: incorrectly recommending a disliked recipe or missing a good one both harm user experience. Therefore, F1 provides a more informative view of performance under potential class imbalance and reflects the model’s ability to correctly identify enjoyable recipes without overpredicting.

We made sure that all input features are based only on information that would be known before or at the time of prediction (when a recipe is first published). This includes metadata like cooking time, ingredients, nutritional information, contributor history, and recipe tags, but excludes any user interaction data such as average rating or number of ratings for new, unseen recipes. This ensures our model can be realistically deployed to make predictions about new recipes as soon as they’re posted, before any user feedback is available.

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
- **Precision: 0.589**
- **Recall: 1.0**

The model achieves an accuracy of approximately 58.9%, which is only slightly better than always predicting the majority class (enjoyed or 1). Meaning our baseline model isn’t adding much predictive value in terms of accuracy. However, our F1 Score of 0.741 indicates that the model is capturing patterns that allow it to distinguish the two classes. So while our accuracy is low, our high F1 suggest that the model is better at identifying minority class than accuracy alone indicates.

The model achieves an accuracy of about 58.9%, which is only slightly better than always predicting the majority class (which is 59%). However, the F1 score of 0.741 and perfect recall (1.0) suggest that the model successfully identifies all actually enjoyed recipes in the test set. This high recall indicates that the model never misses a positive case, but the relatively low precision and accuracy shows that it also makes many false positive predictions: labeling recipes as "enjoyed" when they actually weren't. Overall, the baseline model is heavily skewed toward predicting the majority class but still captures useful trends and signals from the features.

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

Our final model significantly outperformed the baseline logistic regression model across all key metrics:

- **Accuracy: 0.589**
- **F1 Score: 0.741**
- **Precision: 0.589**
- **Recall: 1.0**

Our final Random Forest model achieved:

- **Accuracy: 0.844**
- **F1 Score: 0.875**
- **Precision: 0.832**
- **Recall: 0.922**

These improvements show that the Random Forest model is far better at identifying both enjoyed and not-enjoyed recipes. The high recall means the model successfully detects most recipes users enjoy, while the increased precision means it avoids over-predicting enjoyment. The strong F1 score confirms a healthy balance between the two.

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

The observed difference in F1 score between the two groups was not statistically significant (p_val > 0.05). Thus, we fail to reject the null hypothesis and conclude that our model does include evidence of unfairness with respect to cooking time.

### Conclusion

In this project, we set out to answer a simple but meaningful question: what makes a recipe enjoyable? By modeling whether a recipe's average rating would fall above or below the median, we framed the task as a binary classification problem. Our baseline regression model, using only three basic numerical features, achieved an F1 score of 0.741. While this provided a solid starting point, it lacked the complexity needed to fully capture the underlying trends and nuances of user preferences.

To improve performance, we engineered a range of features informed by domain knowledge: nutritional ratios, user behavior signals, tag encodings, and ingredient-based indicators. Using a Random Forest Classifier with tuned hyperparameters, our final model achieved a significantly higher F1 score of 0.87 and strong precision and recall. These results suggest that user enjoyment is influenced not just by simplicity or cook time, but also by patterns in ingredient choices, tag context, and user familiarity with recipes.

We also ran a fairness analysis to evaluate whether our model treated quick and non-quick recipes equally. The small, statistically insignificant difference in F1 scores between the two groups indicated that the model performed consistently regardless of quickness (cooking time).

Ultimately, our findings show that enjoyment is a multifaceted concept. By combining different features, we were able to model the process behind rating a recipe. This project not only strengthened our understanding of recipe data, but also showed us the power of feature engineering and evaluation in building models.
