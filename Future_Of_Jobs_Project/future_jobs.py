import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Download latest version
#path = kagglehub.dataset_download("name")

#print("Path to dataset files:", path)

# Load job postings dataset
jobs_df = pd.read_csv("./data/linkedin_job_postings.csv")

# Load skills dataset
skills_df = pd.read_csv("./data/job_skills.csv")

# Lets Merge on job_link
merged_df = jobs_df.merge(skills_df, on="job_link", how="left")  # 'left' keeps all job postings

# Convert date_posted to datetime format
merged_df["last_processed_time"] = pd.to_datetime(merged_df["last_processed_time"], errors="coerce")

# Extract year and month
merged_df["year"] = merged_df["last_processed_time"].dt.year
merged_df["month"] = merged_df["last_processed_time"].dt.month

# Aggregate job postings by job title, job skills and location
job_trends = merged_df.groupby(["job_skills", "job_title", "job_location", "year", "month"]).size().reset_index(name="job_count")

# Sort by time
job_trends = job_trends.sort_values(by=["job_skills", "job_title", "job_location", "year", "month"])

# Shift job count to use the previous month's demand as a feature
job_trends["job_count_last_month"] = job_trends.groupby(["job_skills", "job_title", "job_location"])["job_count"].shift(1)

# Rolling average of job demand for the past 3 months
job_trends["job_count_avg_3m"] = job_trends.groupby(["job_skills", "job_title", "job_location"])["job_count"].rolling(3).mean().reset_index(drop=True)

# Encode categorical variables
label_encoders = {}
for col in ["job_skills", "job_title", "job_location", "year", "month"]:
    le = LabelEncoder()
    job_trends[col] = le.fit_transform(job_trends[col])
    label_encoders[col] = le  # Store encoders for later use

# Drop rows with NaN values
job_trends.dropna(subset=["job_count_last_month", "job_count_avg_3m"])

# Select features & target
features = ["year", "month", "job_skills", "job_title", "job_location", "job_count_last_month", "job_count_avg_3m"]
X = job_trends[features]
y = job_trends["job_count"]

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# Predicting future job postings
future_months = 6  # Predict for the next 6 months
last_date = merged_df["last_processed_time"].max()
future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, future_months + 1)]

# Create a DataFrame for future job postings
future_jobs = []
for date in future_dates:
    for _, row in job_trends.iterrows():
        future_jobs.append({
            "year": date.year,
            "month": date.month,
            "job_skills": row["job_skills"],
            "job_title": row["job_title"],
            "job_location": row["job_location"],
            "job_count_last_month": row["job_count"],  # Last available job count
            "job_count_avg_3m": row["job_count_avg_3m"],  # Rolling average
        })

future_df = pd.DataFrame(future_jobs)

# Combine training and future datasets for encoding
combined_df = pd.concat([job_trends, future_df], ignore_index=True)

# Encode categorical variables using combined data
for col in ["year", "month", "job_skills", "job_title", "job_location"]:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    label_encoders[col] = le  # Update encoders with new fit

# Split the combined data back into training and future datasets
job_trends_encoded = combined_df.iloc[:len(job_trends)]
future_df_encoded = combined_df.iloc[len(job_trends):]

# Update the original dataframes with encoded values
job_trends.update(job_trends_encoded)
future_df.update(future_df_encoded)

# Select features & target
features = ["year", "month", "job_skills", "job_title", "job_location", "job_count_last_month", "job_count_avg_3m"]
X = job_trends[features]
y = job_trends["job_count"]

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predict job postings for future months
future_df["predicted_job_count"] = rf.predict(future_df[features])

# Display predictions
print(future_df[["year", "month", "job_skills", "job_title", "job_location", "predicted_job_count"]])


plt.figure(figsize=(12, 6))
for skill in future_df["job_skills"].unique():
    subset = future_df[future_df["job_skills"] == skill]
    plt.plot(subset["month"], subset["predicted_job_count"], label=f"Skill {skill}")

plt.xlabel("Month")
plt.ylabel("Predicted Job Count")
plt.title("Predicted Job Demand Over Time")
plt.legend()
plt.show()

# Group by year and month to get the total job count per month
monthly_trends = job_trends.groupby(["year", "month"])["job_count"].sum().reset_index()

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(monthly_trends["year"].astype(str) + '-' + monthly_trends["month"].astype(str), monthly_trends["job_count"], marker='o')
plt.title("Job Postings Trend Over Time")
plt.xlabel("Time (Year-Month)")
plt.ylabel("Number of Job Postings")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

