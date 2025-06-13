from sqlite3 import Date
from tkinter import E
import pandas as pd
import numpy as np
from utils.random_date import random_date
# Set seed for reproducibility
np.random.seed(42)

n = 10000  # Number of employees



df = pd.DataFrame(columns=[
    "Employee_ID",
    "Date_of_Birth",
    "Gender",
    "Ethnicity",
    "Hire_Date",
    "Termination_Date",
    "Termination_Reason",
    "Department",
    "Job_Title",
    "Job_Level",
    "Manager_ID",
    "Location",
    "Employment_Type",
    "Tenure_in_Months",
    "Time_in_Current_Role"
])

employees_list = []

for i in range(n):
    # Generate random employee data
    Employee_ID = f"E-{i+1:05d}"
    Date_of_Birth = random_date(pd.to_datetime('1975-01-01'), pd.to_datetime('2000-12-31'))
    Gender = np.random.choice(['Male', "Female", "Non-Binary"], p=[0.45, 0.45, 0.1])
    Ethnicity = np.random.choice(['Black', 'White', 'Asian', 'Hispanic', 'Other'])
    Hire_Date = random_date(pd.to_datetime('2010-01-01'), pd.to_datetime('2023-01-01'))
    Termination_Date = random_date(Hire_Date, pd.to_datetime('2025-01-01')) if np.random.rand() > 0.3 else np.nan
    Termination_Reason = np.random.choice(['Voluntary', 'Involuntary', 'Retirement'], p=[0.5, 0.3, 0.2]) if not pd.isna(Termination_Date) else np.nan
    Department = np.random.choice(['Engineering', 'Sales', 'HR', 'Finance', 'Marketing'])
    Job_Title = np.random.choice(['Analyst', 'Manager', 'Engineer', 'Executive'])
    Job_Level = np.random.choice(['L1', 'L2', 'L3', 'L4', 'L5'])
    Manager_ID = f"E-{np.random.randint(1, n+1):05d}"
    Location = np.random.choice(['Ikorodu, Lagos', 'VI, Lagos', 'Abuja', 'Port Harcourt'])
    Employment_Type = np.random.choice(['Full-Time', 'Part-Time', 'Contract'])
    Tenure_in_Months = (Termination_Date - Hire_Date).days // 30 if not pd.isna(Termination_Date) else (pd.to_datetime('2025-01-01') - Hire_Date).days // 30
    # Time in current role in months must be less than or equal to Tenure_in_Months
    Time_in_Current_Role = np.random.randint(1, Tenure_in_Months + 1) if Tenure_in_Months > 0 else 0
    
    data = {
        "Employee_ID": Employee_ID,
        "Date_of_Birth": Date_of_Birth,
        "Gender": Gender,
        "Ethnicity": Ethnicity,
        "Hire_Date": Hire_Date,
        "Termination_Date": Termination_Date,
        "Termination_Reason": Termination_Reason,
        "Department": Department,
        "Job_Title": Job_Title,
        "Job_Level": Job_Level,
        "Manager_ID": Manager_ID,
        "Location": Location,
        "Employment_Type": Employment_Type,
        "Tenure_in_Months": Tenure_in_Months,
        "Time_in_Current_Role": Time_in_Current_Role
    }
    # Append to the DataFrame
    print("🤖\tGenerating data for employee:", Employee_ID)
    employees_list.append(data)
