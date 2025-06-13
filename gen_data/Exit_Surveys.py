import pandas as pd
import numpy as np


employee_data = pd.read_csv("../data/Employees_Core.csv")
# Get former employees
former_employees = employee_data[employee_data["Termination_Date"].notna()].copy()

# Randomly select those who filled the exit survey
exit_survey_sample = former_employees.sample(frac=0.8, random_state=42)

# Simulate exit survey responses
n_exit = len(exit_survey_sample)

np.random.seed(42)

primary_reasons = ["Compensation", "Career Growth", "Management", "Work-Life Balance", "Job Role Mismatch"]
secondary_reasons = ["Commute", "Lack of Training", "Burnout", "Poor Leadership", "Company Culture"]

likes = [
    "The people and the challenging projects.",
    "Flexible work arrangements.",
    "Supportive team environment.",
    "Learning opportunities.",
    "Company values and mission."
]

improvements = [
    "More transparency in promotion decisions.",
    "Better leadership communication.",
    "Improve compensation structure.",
    "Offer more career development programs.",
    "Reduce micromanagement."
]

exit_survey_df = pd.DataFrame({
    "Employee_ID": exit_survey_sample["Employee_ID"].values,
    "Exit_Date": exit_survey_sample["Termination_Date"].values,
    "Primary_Reason_for_Leaving": np.random.choice(primary_reasons, size=n_exit),
    "Secondary_Reason": np.random.choice(secondary_reasons, size=n_exit),
    "Like_Most_About_Company": np.random.choice(likes, size=n_exit),
    "Improvement_Areas": np.random.choice(improvements, size=n_exit),
})