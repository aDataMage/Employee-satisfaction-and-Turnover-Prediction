import pandas as pd
import numpy as np
from employee_turnover_app.utils.helper import random_date

np.random.seed(42)

# Simulating 100 employees
n = 10000
employee_data = pd.read_csv("../data/Employees_Core.csv")
employee_ids = employee_data["Employee_ID"].values
temination_dates = pd.to_datetime(employee_data["Termination_Date"])
# Generate the compensation data


def base_salary(salary_grade):
    """Generate base salary based on salary grade."""
    if salary_grade == 'A':
        return np.random.normal(loc=12_000_000, scale=2_000_000)
    elif salary_grade == 'B':
        return np.random.normal(loc=15_000_000, scale=3_000_000)
    elif salary_grade == 'C':
        return np.random.normal(loc=18_000_000, scale=4_000_000)
    elif salary_grade == 'D':
        return np.random.normal(loc=22_000_000, scale=5_000_000)
    elif salary_grade == 'E':
        return np.random.normal(loc=30_000_000, scale=6_000_000)
    else:
        return np.nan


def bonus(salary_grade):
    """Generate bonus based on salary grade."""
    if salary_grade == 'A':
        return np.random.normal(loc=1_000_000, scale=200_000)
    elif salary_grade == 'B':
        return np.random.normal(loc=2_000_000, scale=400_000)
    elif salary_grade == 'C':
        return np.random.normal(loc=3_000_000, scale=600_000)
    elif salary_grade == 'D':
        return np.random.normal(loc=4_000_000, scale=800_000)
    elif salary_grade == 'E':
        return np.random.normal(loc=5_000_000, scale=1_000_000)
    else:
        return np.nan
    
    
compensation_data = []
for employee_id in employee_ids:
    hire_date = employee_data.loc[employee_data["Employee_ID"] == employee_id, "Hire_Date"].values[0]
    
    # Get termination date if available, else set to a fixed future date
    term_date = employee_data.loc[employee_data["Employee_ID"] == employee_id, "Termination_Date"].values[0]
    if pd.isna(term_date):
        end_date = pd.to_datetime("2025-01-01")
    else:
        end_date = pd.to_datetime(term_date)
    comp_data = {
        "Employee_ID": employee_id,
        # Effective date should be after the hire date and before termination date if applicable
        "Effective_Date": random_date(pd.to_datetime(hire_date), end_date),
        "Salary_Grade": np.random.choice(['A', 'B', 'C', 'D', 'E']),
        # Base salary should be a normal distribution around 15 million NGN with a standard deviation of 3 million 
        # NGN and respect salary grade
        "Stock_Options": np.random.randint(0, 1000),
        "Last_Promotion_Date": pd.to_datetime(np.random.choice(pd.date_range("2019-01-01", "2023-12-31")))
    }
    
    comp_data["Base_Salary"] = [base_salary(sg) for sg in comp_data["Salary_Grade"]][0]
    comp_data["Bonus"] = [bonus(sg) for sg in comp_data["Salary_Grade"]][0]
    comp_data["Total_Compensation"] = comp_data["Base_Salary"] + comp_data["Bonus"] + (comp_data["Stock_Options"] * 1000
                                                                                       )
    # Assuming each stock option is worth 1000 NGN

    compensation_data.append(comp_data)
    print(f"Processed compensation data for Employee ID: {employee_id}")