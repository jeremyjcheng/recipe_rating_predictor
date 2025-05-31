# Cooking Time vs. Recipe Ratings. A Data-Driven analysis on Recipes

Final Project for the University of California, San Diego DSC80 Course
by: Jeremy Cheng and Cedric Jeng



## Overview
Our data science project explores the relationship between the cooking time and rating of a given recipe.


## Introduction
As college students living on campus, finding good food is much harder than it seems. Dining hall meals often fall short, and in the fast-paced quarter system, it can be difficult to find the time to cook. Thus, we find ourself drawn to recipes that are both quick and satisfying. This is why we were excited to explore this dataset: to determine whether shorter cooking times are linked to higher recipe ratings. To do so, we are analyzing two datasets that contain recipes and their ratings from on [food.com](https://www.food.com/) since 2008. Through this project, we hope to better understand how cooking times could affect recipe ratings.

```'recipe'```, the first dataset, consist of 83782 rows (83782 unique recipes) and 12 columns, each representing information on the recipe:
| Column                   | Description |
|:-------------------------|------------:|
| ```'name'```          | recipe name |
| ```'id'```            | recipe ID |
| ```'minutes'```       | minutes to cook recipe |
| ```'contributor_id'```| user ID of who submitted recipe |
| ```'submitted'```     | date recipe was submitted |
| ```'tags'```          | food.com tags |
| ```'nutrition'```     | nutrition information formatted as: \[calores (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)\]; PDV means “percentage of daily value” |
| ```'n_steps'```       | number of steps in recipe|
| ```'steps'```         | text for in order recipe steps |
| ```'description'```   | recipe description from user |
| ```'ingredients'```   | recipe ingredients |
| ```'n_ingredients'``` | number of ingredients in recipe |



```'interactions'```, our second dataset, consist of 731927 rows (731927 unique reviews) and 5 columns, each representing information on the review:
| Column                   | Description |
|:-------------------------|------------:|
| ```'user_id'```  | user ID of reviewer |
| ```'recipe_id'```| ID of recipe |
| ```'date'```     | date review was submitted |
| ```'rating'```   | rating of recipe |
| ```'review'```   | text of the review |


## Data Cleaning and Exploratory Data Analysis


## Assessment of Missingness


## Hypothesis Testing


## Framing a Prediction Problem


## Baseline Model


## Final Model


## Fairness Analysis



