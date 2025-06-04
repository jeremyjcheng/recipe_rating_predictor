# Cooking Time vs. Recipe Ratings. A Data-Driven analysis on Recipes

Final Project for the University of California, San Diego DSC80 Course
<br>by: Jeremy Cheng and Cedric Jeng



## Overview
Our data science project explores the relationship between the cooking time and rating of a given recipe.


## Introduction
As college students living on campus, finding good food is much harder than it seems. Dining hall meals often fall short, and in the fast-paced quarter system, it can be difficult to find the time to cook. Thus, we find ourself drawn to recipes that are both quick and satisfying. This is why we were excited to explore this dataset: to determine whether shorter cooking times are linked to higher recipe ratings. To do so, we are analyzing two datasets that contain recipes and their ratings from on [food.com](https://www.food.com/) since 2008. Through this project, we hope to better understand or even answer the question: Is cooking time a strong predictor of user satisfaction, as measured by average recipe ratings?

```'recipe'```, the first dataset, consist of 83782 rows (83782 unique recipes) and 12 columns, each representing information on the recipe:<br>
| Column                | Description |
|-----------------------|-------------|
| `'name'`              | recipe name |
| `'id'`                | recipe ID |
| `'minutes'`           | minutes to cook recipe |
| `'contributor_id'`    | user ID of who submitted recipe |
| `'submitted'`         | date recipe was submitted |
| `'tags'`              | food.com tags |
| `'nutrition'`         | `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV means "percentage of daily value" |
| `'n_steps'`           | number of steps in recipe |
| `'steps'`             | text for in order recipe steps |
| `'description'`       | recipe description from user |
| `'ingredients'`       | recipe ingredients |
| `'n_ingredients'`     | number of ingredients in recipe |
<br>


```'interactions'```, our second dataset, consist of 731927 rows (731927 unique reviews) and 5 columns, each representing information on the review:<br>
| Column       | Description               |
|:-------------|--------------------------:|
| `'user_id'`  | user ID of reviewer       |
| `'recipe_id'`| ID of recipe              |
| `'date'`     | date review was submitted |
| `'rating'`   | rating of recipe          |
| `'review'`   | text of the review        |


## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
In order to properly analyze the dataset, we did the following to clean our data:

1. Replace ratings of 0 in interactions with np.nan.
    * Ratings on food.com range from 1 to 5, so a rating of 0 does not represent a valid user review, which likely means the user didn’t rate the recipe.
    * Replacing them with ```np.nan``` ensures they are excluded from the average calculation, which is how pandas handles missing values by default.

2. Compute average rating per recipe of the interactions dataset.
    * We're primarily interested in how well each recipe is rated, not the full list of individual interactions.
    * Grouping by ```'recipe_id'``` and computing the mean gives us a single, representative value for each recipe's overall rating.

3. Map the average rating per recipe to raw_recipes
    * Instead of fully merging the datasets, we extract only the information we care about (the average rating) and add it directly to the main recipes dataset.
    * This keeps all unique recipes in ```raw_recipes``` and avoids duplicate rows that would result from merging on one-to-many relationships.
    * If a recipe has no ratings in the interactions dataset (never rated or all ratings were 0), its mapped average will be ```np.nan``` — accurately reflecting the absence of rating data.

4. Add new column ```'high_rating'``` to the dataframe.
    * ```'high_rating'``` is a binary column indicating whether a recipe has an average rating of 4.5 or higher. Recipes that meet this threshold are assigned a 1, while all others receive a 0. This separates the dataset into two groups: highly rated recipes and the rest. It allows for easy comparison of characteristics (e.g., time, ingredients, sugar content) between top-rated recipes and lower-rated ones, helping identify what makes a recipe particularly successful.

#### Result
These are the columns of our cleaned dataframe:<br>
| Column                | Type     |
|:----------------------|---------:|
| `'name'`              | object   |
| `'id'`                | int64    |
| `'minutes'`          | int64    |
| `'contributor_id'`   | int64    |
| `'submitted'`        | object   |
| `'tags'`             | object   |
| `'nutrition'`        | object   |
| `'n_steps'`          | int64    |
| `'steps'`            | object   |
| `'description'`      | object   |
| `'ingredients'`      | object   |
| `'n_ingredients'`    | int64    |
| `'avg_recipe_rating'`| float64  |
| `'high_rating'`      | int64    |

And below is the head of our dataframe with only columns needed for our project:<br>
|    | name                                 |     id |   minutes | tags                                                                                                                                                                                                                                                                                               | nutrition                                     |   n_steps |   n_ingredients |   avg_recipe_rating |   high_rating |
|---:|:-------------------------------------|-------:|----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|----------:|----------------:|--------------------:|--------------:|
|  0 | 1 brownies in the world    best ever | 333281 |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]      |        10 |               9 |                   4 |             0 |
|  1 | 1 in canada chocolate chip cookies   | 453467 |        45 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0]  |        12 |              11 |                   5 |             1 |
|  2 | 412 broccoli casserole               | 306168 |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]     |         6 |               9 |                   5 |             1 |
|  3 | millionaire pound cake               | 286009 |       120 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] | [878.3, 63.0, 326.0, 13.0, 20.0, 123.0, 39.0] |         7 |               7 |                   5 |             1 |
|  4 | 2000 meatloaf                        | 475785 |        90 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             | [267.0, 30.0, 12.0, 12.0, 29.0, 48.0, 2.0]    |        17 |              13 |                   5 |             1 |

Our cleaned dataframe consist of 83782 rows × 9 columns.

### Exploratory Data Analysis
#### Univariate Analysis

We first did univariate analysis to see the distribution of our single variables.

Below is the distribution of cook times in minutes less than or equal to 250 minutes. Initally we plotted our entire minutes column but because of extreme outliers, it showed one large column. Thus we filtered it to ```'<= 250'``` minutes to see our trends better:
<iframe 
    src="graphs/fig_1.html" 
    width="1000"
    height="800"
    frameborder="0"
></iframe>

As we can see in the histogram above, our cooking times show a right-skewed distribution, which suggests logarithmic transformation of our minutes column could help reduce skewness and improve symmetry for better analysis:
<iframe 
    src="graphs/fig_2.html" 
    width="1000"
    height="800"
    frameborder="0"
></iframe>

Next we wanted to look at the distribution of average recipe ratings:
<iframe 
    src="graphs/fig_3.html" 
    width="1000"
    height="800"
    frameborder="0"
></iframe>

#### Bivariate Analysis

Here we want to examine cooking time split into bins against average recipe rating:
<iframe 
    src="graphs/fig_4.html" 
    width="1000"
    height="800"
    frameborder="0"
></iframe>

Above we see a faint trend that ratings decrease through each time bin. In order to confirm this we want to see the mean rating for each time bin:<br>
| time_bin   |   avg_recipe_rating |
|:-----------|--------------------:|
| 0-10 min   |             4.68662 |
| 10-30 min  |             4.63837 |
| 30-60 min  |             4.61215 |
| 60-120 min |             4.6205  |
| 120+ min   |             4.59511 |

There's not a significant trend and the ```'60-120 min'``` bin goes against this trend as it's mean average recipe rating is greater than the bin before it. This made us curious about the distribution of recipe counts per time bin so we visualized that below:

<iframe 
    src="graphs/fig_7.html" 
    width="1000"
    height="800"
    frameborder="0"
></iframe>

Next we want to see the relationship between filtered cooking time (times <= 250) and average recipe rating:
<iframe 
    src="graphs/fig_5.html" 
    width="1000"
    height="800"
    frameborder="0"
></iframe>

We also want to see the relationship between log cooking time and average recipe rating:
<iframe 
    src="graphs/fig_6.html" 
    width="1000"
    height="800"
    frameborder="0"
></iframe>

Finally, we want to see what the mean average recipe rating looks like for quick and non-quick recipes:<br>
|   is_quick |    mean |   count |
|-----------:|--------:|--------:|
|          0 | 4.60937 |   44108 |
|          1 | 4.64439 |   37065 |

## Assessment of Missingness

In our dataset, we have three columns with missing data: ```'date'```, ```'rating'```, and ```'review'```. These columns have a considerable amount of missing values, so we must assess the missingness of the dataset.


## Hypothesis Testing


## Framing a Prediction Problem


## Baseline Model


## Final Model


## Fairness Analysis



