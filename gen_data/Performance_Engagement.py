import pandas as pd
import numpy as np

np.random.seed(42)

employee_data = pd.read_csv("../data/Employees_Core.csv")

# Assume you already have a list of employee IDs from employee_data
employee_ids = employee_data["Employee_ID"].tolist()
n = len(employee_ids)

survey_data = {
    "Employee_ID": employee_ids,
    "Survey_Date": pd.to_datetime("2024-06-30"),
    "Engagement_Score": np.round(np.random.normal(loc=4.0, scale=0.5, size=n), 1),
    "Satisfaction_Score": np.round(np.random.normal(loc=4.0, scale=0.5, size=n), 1),
    "Work_Life_Balance": np.round(np.random.normal(loc=3.5, scale=0.7, size=n), 1),
    "Relationship_with_Manager": np.round(np.random.normal(loc=4.2, scale=0.4, size=n), 1),
    "Career_Development_Opportunities": np.round(np.random.normal(loc=3.0, scale=0.6, size=n), 1),
    "Performance_Rating_2024": np.round(np.random.normal(loc=4.0, scale=0.3, size=n), 1),
    "Performance_Rating_2023": np.round(np.random.normal(loc=3.8, scale=0.4, size=n), 1),
    "Training_Hours_Last_Year": np.random.randint(10, 100, size=n),
    "Projects_Completed": np.random.poisson(lam=5, size=n)
}