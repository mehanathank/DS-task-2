import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

FILE_PATH = "Data.csv"

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET INFORMATION ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\n========== DATA CLEANING ==========")

duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

df = df.drop_duplicates()

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df["release_year"] = df["release_date"].dt.year

numeric_columns = [
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df["runtime"] = df["runtime"].replace(0, np.nan)
df["vote_average"] = df["vote_average"].replace(0, np.nan)

print("\nMissing values:")
print(df.isnull().sum())

df = df[df["revenue"] > 0].copy()

df["original_language"] = df["original_language"].fillna("Unknown")
df["genres"] = df["genres"].fillna("Unknown")
df["production_countries"] = df["production_countries"].fillna("Unknown")

for col in ["runtime", "vote_average", "vote_count", "popularity"]:
    df[col] = df[col].fillna(df[col].median())

print("\nDataset after cleaning:")
print(df.shape)

def get_first_genre(value):
    try:
        values = ast.literal_eval(value)

        if isinstance(values, list) and len(values) > 0:
            return values[0]

    except:
        pass

    return "Unknown"

df["main_genre"] = df["genres"].apply(get_first_genre)

revenue_threshold = df["revenue"].median()

df["success"] = (
    df["revenue"] >= revenue_threshold
).astype(int)

df["success_label"] = df["success"].map({
    0: "Less Successful",
    1: "Successful"
})

print("\n========== PERFORMANCE ==========")
print("Revenue threshold:",
      f"${revenue_threshold:,.0f}")

print("\nPerformance distribution:")
print(df["success_label"].value_counts())

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(
    df[
        [
            "budget",
            "revenue",
            "runtime",
            "vote_average",
            "vote_count",
            "popularity"
        ]
    ].describe()
)

genre_performance = (
    df.groupby("main_genre")
    .agg(
        Movies=("id", "count"),
        Average_Revenue=("revenue", "mean"),
        Average_Rating=("vote_average", "mean"),
        Average_Popularity=("popularity", "mean")
    )
    .sort_values(
        "Average_Revenue",
        ascending=False
    )
)

print("\n========== GENRE PERFORMANCE ==========")
print(genre_performance.head(10))

language_performance = (
    df.groupby("original_language")
    .agg(
        Movies=("id", "count"),
        Average_Revenue=("revenue", "mean"),
        Average_Rating=("vote_average", "mean")
    )
    .sort_values(
        "Average_Revenue",
        ascending=False
    )
)

print("\n========== LANGUAGE PERFORMANCE ==========")
print(language_performance.head(10))

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

sns.histplot(
    df["revenue"],
    bins=50,
    kde=True
)

plt.axvline(
    revenue_threshold,
    linestyle="--",
    label="Median Revenue"
)

plt.title("Distribution of Movie Revenue")
plt.xlabel("Revenue")
plt.ylabel("Number of Movies")
plt.legend()
plt.tight_layout()
plt.show()

top_genres = (
    genre_performance
    .head(10)
    .sort_values("Average_Revenue")
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_genres.reset_index(),
    x="Average_Revenue",
    y="main_genre"
)

plt.title("Top Genres by Average Revenue")
plt.xlabel("Average Revenue")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df.sample(
        min(3000, len(df)),
        random_state=42
    ),
    x="vote_average",
    y="revenue",
    alpha=0.6
)

plt.title("Rating vs Revenue")
plt.xlabel("Rating")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="success_label"
)

plt.title("Successful vs Less-Successful Movies")
plt.xlabel("Performance")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.show()

print("\n========== RELATIONSHIP ANALYSIS ==========")

correlation = df[
    [
        "budget",
        "revenue",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity"
    ]
].corr()

print("\nCorrelation with Revenue:")
print(
    correlation["revenue"]
    .sort_values(
        ascending=False
    )
)

rating_revenue_corr = (
    df["vote_average"]
    .corr(df["revenue"])
)

print(
    "\nRating vs Revenue correlation:",
    round(rating_revenue_corr, 3)
)

model_df = df[
    [
        "budget",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "release_year",
        "original_language",
        "main_genre",
        "success"
    ]
].copy()

model_df = model_df.dropna(
    subset=["release_year"]
)

X = model_df.drop(
    "success",
    axis=1
)

y = model_df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

categorical_features = [
    "original_language",
    "main_genre"
]

numeric_features = [
    "budget",
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "release_year"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train
)

y_pred = model.predict(
    X_test_processed
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n========== RANDOM FOREST RESULTS ==========")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Less Successful",
            "Successful"
        ],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Less Successful",
        "Successful"
    ],
    yticklabels=[
        "Less Successful",
        "Successful"
    ]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

feature_names = (
    preprocessor
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)

print("\n========== TOP FEATURES ==========")
print(
    importance_df.head(15)
)

plt.figure(figsize=(10, 7))

top_features = (
    importance_df
    .head(15)
    .sort_values("Importance")
)

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\n")
print("=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

highest_genre = genre_performance.index[0]

print(
    f"\n1. {highest_genre} has the highest average revenue "
    f"among the analyzed genres."
)

highest_language = language_performance.index[0]

print(
    f"2. {highest_language} has the highest average revenue "
    f"among the analyzed languages."
)

print(
    f"3. The correlation between rating and revenue is "
    f"{rating_revenue_corr:.3f}."
)

budget_revenue_corr = (
    df["budget"]
    .corr(df["revenue"])
)

print(
    f"4. The correlation between budget and revenue is "
    f"{budget_revenue_corr:.3f}."
)

runtime_revenue_corr = (
    df["runtime"]
    .corr(df["revenue"])
)

print(
    f"5. The correlation between runtime and revenue is "
    f"{runtime_revenue_corr:.3f}."
)

popularity_revenue_corr = (
    df["popularity"]
    .corr(df["revenue"])
)

print(
    f"6. The correlation between popularity and revenue is "
    f"{popularity_revenue_corr:.3f}."
)

print(
    f"7. The Random Forest achieved an F1-score of "
    f"{f1:.3f} when classifying high- and low-performance movies."
)

print("\n")
print("=" * 60)
print("PREDICT PERFORMANCE OF A NEW MOVIE")
print("=" * 60)

print(
    f"\nRevenue threshold used by the model: "
    f"${revenue_threshold:,.0f}"
)

budget_input = float(
    input("Budget ($): ")
)

runtime_input = float(
    input("Runtime (minutes): ")
)

rating_input = float(
    input("Rating (0-10): ")
)

vote_count_input = int(
    input("Vote Count: ")
)

popularity_input = float(
    input("Popularity: ")
)

release_year_input = int(
    input("Release Year: ")
)

language_input = input(
    "Original Language (example: en, fr, hi): "
).strip()

genre_input = input(
    "Main Genre (example: Action, Drama, Comedy): "
).strip()

user_data = pd.DataFrame({
    "budget": [budget_input],
    "runtime": [runtime_input],
    "vote_average": [rating_input],
    "vote_count": [vote_count_input],
    "popularity": [popularity_input],
    "release_year": [release_year_input],
    "original_language": [language_input],
    "main_genre": [genre_input]
})

user_processed = preprocessor.transform(
    user_data
)

prediction = model.predict(
    user_processed
)[0]

probability = model.predict_proba(
    user_processed
)[0]

print("\n")
print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

if prediction == 1:

    print(
        "\nPredicted Performance: SUCCESSFUL"
    )

    print(
        f"Probability of Successful: "
        f"{probability[1] * 100:.2f}%"
    )

else:

    print(
        "\nPredicted Performance: LESS SUCCESSFUL"
    )

    print(
        f"Probability of Less Successful: "
        f"{probability[0] * 100:.2f}%"
    )

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Revenue threshold: ${revenue_threshold:,.0f}"
)

print("=" * 60)

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison["Actual"] = comparison["Actual"].map({
    0: "Less Successful",
    1: "Successful"
})

comparison["Predicted"] = comparison["Predicted"].map({
    0: "Less Successful",
    1: "Successful"
})

print("\n========== ACTUAL VS PREDICTED ==========")

print(
    comparison.head(20).to_string(
        index=False
    )
)