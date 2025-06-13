import os
import sys
from compensation_history import *
from gen_employees import *
from Performance_Engagement import *
from Exit_Surveys import *

def main():
    """Run all data generation scripts."""
    print("Starting data generation process...")

    # Generate compensation history data
    print("\nGenerating compensation history data...")
    try:
        # Save the DataFrame to a CSV file

        survey_df = pd.DataFrame(survey_data)
        survey_df.to_csv('../data/Performance_Engagement.csv', index=False)
        compensation_df = pd.DataFrame(compensation_data)
        compensation_df.to_csv("../data/Compensation_History.csv", index=False)
        print("Compensation history data generation complete.")
        exit_survey_df.to_csv("../data/Exit_Surveys.csv", index=False)
        print("Exit survey data generation complete.")
    except Exception as e:
        print(f"Error generating compensation history data: {str(e)}")
        sys.exit(1)

    print("\nAll data generation complete!")


if __name__ == "__main__":
    employee_df = pd.DataFrame(employees_list)
    employee_df.to_csv("../data/Employees_Core.csv", index=False)
    print("Data saved to Employees_Core.csv")
    main()
# Create DataFrame
