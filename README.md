# Music Artist Discography Predictor
An end-to-end Machine Learning project to predict an artist's total album count using MusicBrainz data.

---

## Project Overview
This project aims to predict the total number of albums an artist has released based on their biographical and stylistic data (such as active years, genre, and artist type).
The data was sourced directly from the **MusicBrainz API**.

**Key Characteristics of the Dataset:**
* **The "Genre Icons" Bias:** The dataset was constructed by querying the top 200 artists across various musical genres. Because the API defaults to sorting by relevance and popularity,
the dataset inherently suffers from survivorship bias. It predominantly represents musical legends and highly established acts rather than the average, everyday musician.
* **The 25-Album Wall (Censored Data):** A crucial discovery during the Exploratory Data Analysis (EDA) was an API limitation that capped the retrieved album count at exactly 25 releases per artist.
Consequently, our target variable (`album_count`) is heavily right-censored. In practice, our model predicts an artist's output up to this threshold,
determining how quickly or likely they are to hit this "legendary" 25-album cap.

** Tech Stack:**
* **Data Manipulation:** `pandas`, `numpy`, `ast`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Preprocessing:** `scikit-learn` (MultiLabelBinarizer, train_test_split, metrics), `xgboost` (XGBRegressor)

## Dataset Description
The dataset was constructed by querying the **MusicBrainz** database, an open-source, community-maintained encyclopedia that collects highly structured music metadata. 
By fetching the top 200 artists across various predefined genres, I gathered a robust sample of prominent musical acts.

Here is a breakdown of the key features extracted and utilized in this project:

### Categorical Features
* **`type`**: The entity type of the artist (e.g., *Person*, *Group*). Extremely rare categories (like *Character* or *Orchestra*) were consolidated or removed to reduce noise.
* **`gender`**: The gender of the artist (e.g., *Male*, *Female*, *Not applicable* for groups).
* **`area_name`**: The geographical origin of the artist (e.g., *United Kingdom*, *North America*, *Europe*, *Rest of World*).
* **`genre_label`**: A list of genres associated with the artist (e.g., `['rock', 'pop']`). Because many artists cross genres, this was treated as a multi-label feature.

### Numerical Features
* **`begin_year` & `end_year`**: The raw dates representing the start and end of an artist's career (or birth/death for solo artists). These were heavily engineered into a more accurate representation of career length.
* **`label_count`**: The number of distinct record labels the artist has been associated with.
* **`tag_count`**: The number of user-defined tags attached to the artist in the MusicBrainz database (a proxy for community engagement or stylistic complexity).
* **`release_rel_count`**: The number of related release entities in the database.

### Target Variable
* **`album_count`**: The total number of primary album releases associated with the artist (capped at 25 due to API pagination limits).

## Data Cleaning & Feature Engineering
Raw data extracted from APIs is rarely ready for modeling. A significant portion of this project was dedicated to cleaning, standardizing, 
and engineering domain-specific features to maximize the model's predictive power.

* **Deduplication via Multi-Labeling:** Artists appearing multiple times (due to being tagged with multiple genres) were deduplicated. Instead of dropping valuable information,
their genres were aggregated into comprehensive lists (e.g., `['rock', 'pop', 'electronic']`), which were later transformed into sparse binary features using `MultiLabelBinarizer`.
* **Manual Curation over Blind Imputation:** For critical missing values in the `type` column (e.g., the artist "Sirch"), manual research was conducted to fill in the gaps.
This deliberate choice prioritized data integrity over statistical imputation.
* **Categorical Consolidation:**
  * **`gender`:** Logically handled by assigning 'Not applicable' to all *Groups* and *Orchestras*. Missing values and minority categories (like Non-Binary) were grouped into an 'Other/Unknown' bucket to reduce noise.
  * **`area_name`:** Highly granular locations were standardized to country levels (e.g., California -> United States, Scotland -> United Kingdom) and further aggregated into broader geographical regions.
  Placeholder noise like 'Worldwide' and unresolved NaNs (like 'Various Artists') were completely dropped.
* **Temporal Engineering (Domain Knowledge Application):**
  * Raw `begin_date` and `end_date` strings were parsed into float years.
  * **Grouped Imputation:** Missing `begin_year` values were intelligently imputed using the median of their specific `genre_label` and `type` combination,
  acknowledging that a jazz soloist's timeline differs vastly from a rock band's.
  * **The 18-Year Rule:** Since the API uses birth dates for Soloists (`Person`) and formation dates for `Groups`, calculating active years directly would heavily skew the data
  (giving a 70-year-old soloist a 70-year music career). A domain-specific correction was applied: subtracting 18 years from a soloist's biological age to represent their *professional* active years (`years_active_pro`).
  * Missing `end_year` values for active acts were filled with the current year (2026), while logical inconsistencies (e.g., `ended == True` but missing `end_date`) were corrected based on median career lengths.
 
## Key Insights from Exploratory Data Analysis (EDA)
Before building the predictive model, a thorough EDA was conducted to understand the underlying distributions and relationships within the dataset.

* **The Target Ceiling:** The distribution of the target variable (`album_count`) clearly visualized the aforementioned API pagination limit, with a massive spike at the 25-album mark. 
* **Embracing Skewness:** Univariate analysis revealed severe right-skewness in numerical features like `tag_count` and `label_count`.
Instead of forcing normalization (e.g., via Log or Box-Cox transformations), these distributions were kept intact. This decision seamlessly aligns with the choice of tree-based models (like XGBoost),
which are inherently robust to monotonic transformations and non-normal data.
* **Noise Reduction:** Bivariate analysis highlighted that certain rare categories, specifically the 'Other' `type`, were associated with exactly 0 albums.
These records were identified as database noise/placeholders and completely removed to prevent the model from learning false patterns.
* **Feature Correlations:** The engineered `years_active_pro` feature proved to be the strongest predictor, yielding a solid positive correlation (~0.41) with the `album_count`. 
* **Soloists vs. Groups:** Boxplot comparisons revealed an interesting trend within this "legends" dataset: Soloists (`Person`)
generally possessed a higher median album count and reached the 25-album cap more frequently than `Groups`, showcasing different release dynamics between the two entity types.


## Modeling
With the features cleaned, engineered, and encoded, the dataset was split into a standard 80/20 Train-Test split. 

For the predictive modeling phase, an **XGBoost Regressor (`XGBRegressor`)** was selected as the primary algorithm. The choice of a tree-based ensemble model was deliberate and driven by the EDA findings:
* **Robustness to Skewness:** XGBoost handles heavily right-skewed features (like `tag_count` and `label_count`) natively, eliminating the need for arbitrary log transformations or feature scaling.
* **Non-linear Relationships:** Tree-based models are excellent at capturing non-linear relationships and interactions between sparse binary categories
(like our `MultiLabelBinarizer` genre outputs) and continuous variables.
* **Baseline Approach:** The focus of this project was on robust data engineering and domain-specific preprocessing rather than extensive hyperparameter tuning.
The model was trained using baseline parameters to evaluate the pure predictive power of the engineered features.

---

## Model Performance & Evaluation
The model was evaluated on the unseen test set using standard regression metrics. The results reflect both the quality of the engineered features and the inherent unpredictability of the music industry.

**Results:**
* **Mean Absolute Error (MAE): 5.62 albums**
* **Root Mean Squared Error (RMSE): 7.29 albums**
* **R-squared ($R^2$): 0.16**

**Interpretation of Results:**
* An MAE of ~5.6 indicates that, on average, the model's predictions are off by about 5 to 6 albums. Considering the hard cap at 25 albums and the volatility of artistic output, this represents a solid, logical baseline.
* The $R^2$ score of 0.16 highlights the reality of modeling human creativity: artistic output is driven by countless unquantifiable factors
(inspiration, label disputes, personal life). Furthermore, the hard right-censoring of the target variable (capped at 25) heavily limits the model's ability to capture the true variance of legendary acts.

**Feature Importance:**
An analysis of the XGBoost `Gain` feature importance confirmed our EDA hypotheses:
1. **`years_active_pro`** overwhelmingly served as the strongest predictor, validating the custom "18-Year Rule" engineering for soloists.
2. **`type_Person`** emerged as highly significant, confirming that the entity type (Soloist vs. Group) fundamentally alters release dynamics.
3. **Specific Genres** (such as Jazz and Electronic) provided substantial information gain, proving that binarizing the multi-label genre lists was a highly effective strategy.

---

## Conclusion & Future Work
This project successfully demonstrates an end-to-end data science pipeline—from handling raw, nested API data to domain-specific feature engineering and predictive modeling. 
The analysis revealed that while time in the industry (`years_active_pro`) is the engine of an artist's discography, 
the API limits and survivorship bias redefine the modeling goal from "predicting total lifetime output" to "predicting the velocity towards the 25-album cap."

**Future Improvements:**
* **Breaking the 25-Album Wall:** Implementing pagination in the API requests to fetch the true, uncensored total album counts for legendary artists.
* **Hyperparameter Tuning:** Utilizing `GridSearchCV` or `Optuna` to optimize the XGBoost parameters and squeeze out better performance.
* **Integrating Popularity Metrics:** Merging the MusicBrainz dataset with external APIs (like Spotify or Last.fm) to include listener counts or streaming metrics as predictive features.

---
*Created by Mateusz Homik
