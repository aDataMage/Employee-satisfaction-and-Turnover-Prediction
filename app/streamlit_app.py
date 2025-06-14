import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
import os

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Enhanced Employee Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-high { border-left-color: #d62728 !important; }
    .risk-medium { border-left-color: #ff7f0e !important; }
    .risk-low { border-left-color: #2ca02c !important; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_enhanced_data():
    """Load data with all available columns properly utilized"""
    try:
        base_path = "../data"
        required_files = ["Employees_Core.csv", "Compensation_History.csv",
                          "Exit_Surveys.csv", "Performance_Engagement.csv"]

        for file in required_files:
            if not os.path.exists(os.path.join(base_path, file)):
                st.error(f"Missing file: {file}")
                raise FileNotFoundError

        # Load files
        employee_data = pd.read_csv(os.path.join(base_path, "Employees_Core.csv"))
        compensation_data = pd.read_csv(os.path.join(base_path, "Compensation_History.csv"))
        exit_survey_data = pd.read_csv(os.path.join(base_path, "Exit_Surveys.csv"))
        performance_data = pd.read_csv(os.path.join(base_path, "Performance_Engagement.csv"))

        # Merge all datasets
        full_data = employee_data.merge(compensation_data, on="Employee_ID", how="left")
        full_data = full_data.merge(exit_survey_data, on="Employee_ID", how="left")
        full_data = full_data.merge(performance_data, on="Employee_ID", how="left")

        # Process all date columns
        date_columns = ['Date_of_Birth', 'Hire_Date', 'Termination_Date', 'Effective_Date',
                        'Last_Promotion_Date', 'Exit_Date', 'Survey_Date']
        for col in date_columns:
            if col in full_data.columns:
                full_data[col] = pd.to_datetime(full_data[col], errors='coerce')

        # Enhanced derived features using ALL available columns
        current_date = pd.Timestamp.now()

        # Age and tenure calculations
        full_data['Age'] = (current_date - full_data['Date_of_Birth']).dt.days / 365.25
        full_data['Tenure_Years'] = full_data['Tenure_in_Months'] / 12
        full_data['Time_in_Role_Years'] = full_data['Time_in_Current_Role'] / 12

        # Employment status
        full_data['Has_Left'] = full_data['Termination_Date'].notna()
        full_data['Is_Voluntary_Exit'] = full_data['Termination_Reason'] == 'Voluntary'

        # Promotion and career progression
        full_data['Months_Since_Promotion'] = (current_date - full_data['Last_Promotion_Date']).dt.days / 30.44
        full_data['Has_Been_Promoted'] = full_data['Last_Promotion_Date'].notna()

        # Compensation metrics
        full_data['Bonus_to_Salary_Ratio'] = full_data['Bonus'] / full_data['Base_Salary']
        full_data['Total_Comp_Rank'] = full_data['Total_Compensation'].rank(pct=True)

        # Performance trends
        full_data['Performance_Trend'] = full_data['Performance_Rating_2024'] - full_data['Performance_Rating_2023']
        full_data['High_Performer'] = full_data['Performance_Rating_2024'] >= 4.0
        full_data['Training_Intensity'] = pd.cut(full_data['Training_Hours_Last_Year'],
                                                 bins=[0, 20, 40, 80, float('inf')],
                                                 labels=['Low', 'Medium', 'High', 'Very High'])

        # Demographics and diversity
        full_data['Age_Group'] = pd.cut(full_data['Age'],
                                        bins=[0, 30, 40, 50, 100],
                                        labels=['<30', '30-40', '40-50', '50+'])

        # Location analysis
        full_data['Is_Remote'] = full_data['Location'].str.contains('Remote', na=False)

        # Exit survey insights (for those who left)
        full_data['Exit_Reason_Category'] = full_data['Primary_Reason_for_Leaving'].fillna('Active Employee')

        return full_data

    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return generate_enhanced_sample_data()


def generate_enhanced_sample_data():
    """Generate comprehensive sample data matching your schema"""
    np.random.seed(42)
    n_employees = 1000

    # Employee IDs
    employee_ids = [f"E-{str(i).zfill(5)}" for i in range(1, n_employees + 1)]

    # Demographics
    genders = ['Male', 'Female', 'Non-Binary']
    ethnicities = ['White', 'Asian', 'Black', 'Hispanic', 'Other']
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
    job_titles = ['Analyst', 'Specialist', 'Manager', 'Director', 'VP', 'Executive']
    job_levels = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    locations = ['New York, NY', 'San Francisco, CA', 'Chicago, IL', 'Austin, TX', 'Remote', 'VI, Lagos']
    employment_types = ['Full-Time', 'Part-Time', 'Contract']
    salary_grades = ['A', 'B', 'C', 'D', 'E', 'F']
    termination_reasons = ['Voluntary', 'Involuntary', 'Layoff', 'Retirement']

    # Generate base employee data
    data = {
        'Employee_ID': employee_ids,
        'Date_of_Birth': pd.date_range('1960-01-01', '2000-12-31', periods=n_employees),
        'Gender': np.random.choice(genders, n_employees, p=[0.45, 0.50, 0.05]),
        'Ethnicity': np.random.choice(ethnicities, n_employees, p=[0.55, 0.20, 0.15, 0.08, 0.02]),
        'Hire_Date': pd.date_range('2015-01-01', '2024-12-31', periods=n_employees),
        'Department': np.random.choice(departments, n_employees),
        'Job_Title': np.random.choice(job_titles, n_employees),
        'Job_Level': np.random.choice(job_levels, n_employees),
        'Location': np.random.choice(locations, n_employees, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]),
        'Employment_Type': np.random.choice(employment_types, n_employees, p=[0.85, 0.10, 0.05]),
        'Tenure_in_Months': np.random.exponential(24, n_employees),
        'Time_in_Current_Role': np.random.exponential(18, n_employees),

        # Compensation data
        'Salary_Grade': np.random.choice(salary_grades, n_employees),
        'Stock_Options': np.random.poisson(100, n_employees),
        'Base_Salary': np.random.normal(75000, 25000, n_employees),
        'Bonus': np.random.normal(8000, 5000, n_employees),

        # Performance and engagement data
        'Engagement_Score': np.random.normal(7, 1.5, n_employees),
        'Satisfaction_Score': np.random.normal(7.2, 1.3, n_employees),
        'Work_Life_Balance': np.random.normal(6.8, 1.4, n_employees),
        'Relationship_with_Manager': np.random.normal(7.5, 1.2, n_employees),
        'Career_Development_Opportunities': np.random.normal(6.5, 1.6, n_employees),
        'Performance_Rating_2024': np.random.normal(3.2, 0.8, n_employees),
        'Performance_Rating_2023': np.random.normal(3.1, 0.8, n_employees),
        'Training_Hours_Last_Year': np.random.poisson(35, n_employees),
        'Projects_Completed': np.random.poisson(6, n_employees),

        # Exit survey data (for some employees)
        'Primary_Reason_for_Leaving': np.random.choice(
            ['Work-Life Balance', 'Career Growth', 'Compensation', 'Management', 'Company Culture', None],
            n_employees, p=[0.05, 0.04, 0.03, 0.02, 0.01, 0.85]
        ),
        'Secondary_Reason': np.random.choice(
            ['Commute', 'Workload', 'Limited Opportunities', 'Stress', None],
            n_employees, p=[0.02, 0.03, 0.02, 0.01, 0.92]
        ),
    }

    df = pd.DataFrame(data)

    # Add termination data for some employees
    termination_mask = np.random.choice([True, False], n_employees, p=[0.18, 0.82])
    df.loc[termination_mask, 'Termination_Date'] = pd.date_range('2020-01-01', '2024-12-31',
                                                                 periods=termination_mask.sum())
    df.loc[termination_mask, 'Termination_Reason'] = np.random.choice(termination_reasons, termination_mask.sum(),
                                                                      p=[0.70, 0.15, 0.10, 0.05])

    # Add promotion dates
    promotion_mask = np.random.choice([True, False], n_employees, p=[0.60, 0.40])
    df.loc[promotion_mask, 'Last_Promotion_Date'] = pd.date_range('2020-01-01', '2024-06-30',
                                                                  periods=promotion_mask.sum())

    # Calculate derived fields
    current_date = pd.Timestamp.now()
    df['Age'] = (current_date - df['Date_of_Birth']).dt.days / 365.25
    df['Tenure_Years'] = df['Tenure_in_Months'] / 12
    df['Time_in_Role_Years'] = df['Time_in_Current_Role'] / 12
    df['Has_Left'] = df['Termination_Date'].notna()
    df['Is_Voluntary_Exit'] = df['Termination_Reason'] == 'Voluntary'
    df['Months_Since_Promotion'] = (current_date - df['Last_Promotion_Date']).dt.days / 30.44
    df['Has_Been_Promoted'] = df['Last_Promotion_Date'].notna()
    df['Total_Compensation'] = df['Base_Salary'] + df['Bonus']
    df['Bonus_to_Salary_Ratio'] = df['Bonus'] / df['Base_Salary']
    df['Total_Comp_Rank'] = df['Total_Compensation'].rank(pct=True)
    df['Performance_Trend'] = df['Performance_Rating_2024'] - df['Performance_Rating_2023']
    df['High_Performer'] = df['Performance_Rating_2024'] >= 4.0
    df['Training_Intensity'] = pd.cut(df['Training_Hours_Last_Year'],
                                      bins=[0, 20, 40, 80, float('inf')],
                                      labels=['Low', 'Medium', 'High', 'Very High'])
    df['Age_Group'] = pd.cut(df['Age'],
                             bins=[0, 30, 40, 50, 100],
                             labels=['<30', '30-40', '40-50', '50+'])
    df['Is_Remote'] = df['Location'].str.contains('Remote', na=False)
    df['Exit_Reason_Category'] = df['Primary_Reason_for_Leaving'].fillna('Active Employee')

    # Ensure realistic ranges
    numeric_cols = ['Age', 'Engagement_Score', 'Satisfaction_Score', 'Work_Life_Balance',
                    'Relationship_with_Manager', 'Career_Development_Opportunities',
                    'Performance_Rating_2024', 'Performance_Rating_2023', 'Base_Salary', 'Bonus']

    for col in numeric_cols:
        if col in ['Age']:
            df[col] = np.clip(df[col], 22, 65)
        elif 'Score' in col or 'Balance' in col or 'Manager' in col or 'Opportunities' in col:
            df[col] = np.clip(df[col], 1, 10)
        elif 'Rating' in col:
            df[col] = np.clip(df[col], 1, 5)
        elif col == 'Base_Salary':
            df[col] = np.clip(df[col], 35000, 200000)
        elif col == 'Bonus':
            df[col] = np.clip(df[col], 0, 50000)

    return df


def calculate_enhanced_risk_score(row):
    """Enhanced risk calculation using more data points"""
    score = 0

    # Core engagement factors (40% weight)
    if row['Engagement_Score'] < 3:
        score += 4
    elif row['Engagement_Score'] < 6:
        score += 2
    elif row['Engagement_Score'] < 7:
        score += 1

    if row['Satisfaction_Score'] < 3:
        score += 4
    elif row['Satisfaction_Score'] < 6:
        score += 2
    elif row['Satisfaction_Score'] < 7:
        score += 1

    # Work environment factors (30% weight)
    if row['Work_Life_Balance'] < 3:
        score += 3
    elif row['Work_Life_Balance'] < 6:
        score += 1.5

    if row['Relationship_with_Manager'] < 3:
        score += 3
    elif row['Relationship_with_Manager'] < 6:
        score += 1.5

    # Career factors (20% weight)
    if row['Career_Development_Opportunities'] < 3:
        score += 2
    elif row['Career_Development_Opportunities'] < 6:
        score += 1

    # Time-based risk factors (10% weight)
    if pd.notna(row['Months_Since_Promotion']) and row['Months_Since_Promotion'] > 48: score += 1
    if row['Time_in_Role_Years'] > 5: score += 0.5

    # Performance factors
    if row['Performance_Rating_2024'] < 2.5: score += 1
    if row['Performance_Trend'] < -0.5: score += 0.5

    # Compensation factors
    if row['Total_Comp_Rank'] < 0.25: score += 0.5  # Bottom quartile compensation

    # Training and development
    if row['Training_Hours_Last_Year'] < 10: score += 0.5

    # Normalize to 0-100 scale
    risk_score = min(100, (score / 20) * 100)
    return risk_score


def get_risk_category(score):
    """Categorize risk score"""
    if score >= 70:
        return "Critical"
    elif score >= 50:
        return "High"
    elif score >= 30:
        return "Medium"
    else:
        return "Low"


@st.cache_data
def load_and_prepare_enhanced_data():
    """Load and prepare data with enhanced risk calculations"""
    data = load_enhanced_data().copy()
    data['Risk_Score'] = data.apply(calculate_enhanced_risk_score, axis=1)
    data['Risk_Category'] = data['Risk_Score'].apply(get_risk_category)
    return data


def enhanced_filters(data):
    """Enhanced filtering options"""
    st.sidebar.header("🔍 Advanced Filters")

    # Multi-select filters
    departments = st.sidebar.multiselect(
        "Department", sorted(data['Department'].unique()),
        default=sorted(data['Department'].unique())
    )

    job_levels = st.sidebar.multiselect(
        "Job Level", sorted(data['Job_Level'].unique()),
        default=sorted(data['Job_Level'].unique())
    )

    locations = st.sidebar.multiselect(
        "Location", sorted(data['Location'].unique()),
        default=sorted(data['Location'].unique())
    )

    employment_types = st.sidebar.multiselect(
        "Employment Type", sorted(data['Employment_Type'].unique()),
        default=sorted(data['Employment_Type'].unique())
    )

    # Age and tenure sliders
    age_range = st.sidebar.slider(
        "Age Range",
        int(data['Age'].min()), int(data['Age'].max()),
        (int(data['Age'].min()), int(data['Age'].max()))
    )

    tenure_range = st.sidebar.slider(
        "Tenure Range (Years)",
        0.0, float(data['Tenure_Years'].max()),
        (0.0, float(data['Tenure_Years'].max()))
    )

    # Performance filters
    performance_range = st.sidebar.slider(
        "2024 Performance Rating",
        1.0, 5.0, (1.0, 5.0), 0.1
    )

    # Compensation filter
    comp_percentile = st.sidebar.slider(
        "Minimum Compensation Percentile",
        0, 100, 0
    )

    return {
        'departments': departments,
        'job_levels': job_levels,
        'locations': locations,
        'employment_types': employment_types,
        'age_range': age_range,
        'tenure_range': tenure_range,
        'performance_range': performance_range,
        'comp_percentile': comp_percentile
    }


def apply_enhanced_filters(data, filters):
    """Apply all selected filters"""
    filtered_data = data[
        (data['Department'].isin(filters['departments'])) &
        (data['Job_Level'].isin(filters['job_levels'])) &
        (data['Location'].isin(filters['locations'])) &
        (data['Employment_Type'].isin(filters['employment_types'])) &
        (data['Age'] >= filters['age_range'][0]) &
        (data['Age'] <= filters['age_range'][1]) &
        (data['Tenure_Years'] >= filters['tenure_range'][0]) &
        (data['Tenure_Years'] <= filters['tenure_range'][1]) &
        (data['Performance_Rating_2024'] >= filters['performance_range'][0]) &
        (data['Performance_Rating_2024'] <= filters['performance_range'][1]) &
        (data['Total_Comp_Rank'] >= filters['comp_percentile'] / 100)
        ]
    return filtered_data


def enhanced_executive_summary(data):
    """Enhanced executive summary using all data points"""
    st.header("📊 Enhanced Executive Dashboard")

    # Key metrics calculation
    total_employees = len(data)
    turnover_rate = (data['Has_Left'].sum() / total_employees) * 100 if total_employees > 0 else 0
    voluntary_turnover = (data['Is_Voluntary_Exit'].sum() / total_employees) * 100 if total_employees > 0 else 0
    high_risk_count = len(data[data['Risk_Category'].isin(['High', 'Critical'])])
    avg_tenure = data['Tenure_Years'].mean()
    avg_compensation = data['Total_Compensation'].mean()
    promotion_rate = (data['Has_Been_Promoted'].sum() / total_employees) * 100 if total_employees > 0 else 0
    high_performer_retention = ((~data['Has_Left'] & data['High_Performer']).sum() / data[
        'High_Performer'].sum()) * 100 if data['High_Performer'].sum() > 0 else 0

    # Display enhanced metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Employees", f"{total_employees:,}")
        st.metric("Overall Turnover", f"{turnover_rate:.1f}%")

    with col2:
        st.metric("Voluntary Turnover", f"{voluntary_turnover:.1f}%")
        st.metric("High Risk Employees", high_risk_count)

    with col3:
        st.metric("Avg Tenure", f"{avg_tenure:.1f} years")
        st.metric("Promotion Rate", f"{promotion_rate:.1f}%")

    with col4:
        st.metric("Avg Total Compensation", f"${avg_compensation:,.0f}")
        st.metric("High Performer Retention", f"{high_performer_retention:.1f}%")

    # Enhanced visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Turnover by Exit Reason")
        exit_reasons = data[data['Has_Left']]['Primary_Reason_for_Leaving'].value_counts()
        if not exit_reasons.empty:
            fig = px.pie(values=exit_reasons.values, names=exit_reasons.index,
                         title="Primary Reasons for Leaving")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No exit data available")

    with col2:
        st.subheader("💼 Performance Distribution")
        fig = px.histogram(data, x='Performance_Rating_2024', nbins=20,
                           title="2024 Performance Ratings Distribution")
        fig.update_layout(xaxis_title="Performance Rating", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Demographics insights
    st.subheader("👥 Workforce Demographics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Gender Distribution**")
        gender_dist = data['Gender'].value_counts()
        fig = px.bar(x=gender_dist.index, y=gender_dist.values)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Age Groups**")
        age_dist = data['Age_Group'].value_counts()
        fig = px.bar(x=age_dist.index, y=age_dist.values)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("**Employment Type**")
        emp_type_dist = data['Employment_Type'].value_counts()
        fig = px.bar(x=emp_type_dist.index, y=emp_type_dist.values)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)


def compensation_analysis(data):
    """Detailed compensation analysis"""
    st.header("💰 Compensation & Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💵 Compensation vs Performance")
        fig = px.scatter(data, x='Performance_Rating_2024', y='Total_Compensation',
                         color='Risk_Category', size='Tenure_Years',
                         hover_data=['Department', 'Job_Level'],
                         title="Compensation vs Performance Relationship")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Salary Grade Distribution")
        salary_grade_stats = data.groupby('Salary_Grade').agg({
            'Base_Salary': 'mean',
            'Total_Compensation': 'mean',
            'Employee_ID': 'count'
        }).round(0)
        salary_grade_stats.columns = ['Avg Base Salary', 'Avg Total Comp', 'Count']
        st.dataframe(salary_grade_stats)

    # Compensation equity analysis
    st.subheader("⚖️ Compensation Equity Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**By Gender**")
        gender_comp = data.groupby('Gender')['Total_Compensation'].agg(['mean', 'median', 'count']).round(0)
        st.dataframe(gender_comp)

    with col2:
        st.markdown("**By Ethnicity**")
        ethnicity_comp = data.groupby('Ethnicity')['Total_Compensation'].agg(['mean', 'median', 'count']).round(0)
        st.dataframe(ethnicity_comp)

    # Bonus analysis
    st.subheader("🎁 Bonus Analysis")
    fig = px.box(data, x='Department', y='Bonus_to_Salary_Ratio',
                 title="Bonus to Salary Ratio by Department")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def career_progression_analysis(data):
    """Career progression and development analysis"""
    st.header("🚀 Career Progression Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Time Since Last Promotion")
        promotion_data = data[data['Has_Been_Promoted']].copy()
        if not promotion_data.empty:
            fig = px.histogram(promotion_data, x='Months_Since_Promotion', nbins=20,
                               title="Distribution of Months Since Last Promotion")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No promotion data available")

    with col2:
        st.subheader("🎯 Training Hours vs Performance")
        fig = px.scatter(data, x='Training_Hours_Last_Year', y='Performance_Rating_2024',
                         color='Risk_Category',
                         title="Training Investment vs Performance")
        st.plotly_chart(fig, use_container_width=True)

    # Career development insights
    st.subheader("📚 Training and Development Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        training_avg = data.groupby('Department')['Training_Hours_Last_Year'].mean().sort_values(ascending=False)
        st.markdown("**Training Hours by Department**")
        st.bar_chart(training_avg)

    with col2:
        projects_avg = data.groupby('Job_Level')['Projects_Completed'].mean().sort_values(ascending=False)
        st.markdown("**Projects Completed by Level**")
        st.bar_chart(projects_avg)

    with col3:
        # Performance trend analysis
        perf_trend_dist = data['Performance_Trend'].value_counts(bins=5)
        st.markdown("**Performance Trend Distribution**")
        st.bar_chart(perf_trend_dist)


def location_remote_analysis(data):
    """Location and remote work analysis"""
    st.header("🌍 Location & Remote Work Analysis")


    st.subheader("📍 Employee Distribution by Location")
    location_dist = data['Location'].value_counts()
    fig = px.bar(x=location_dist.values, y=location_dist.index, orientation='h',
                     title="Employee Count by Location")
    st.plotly_chart(fig, use_container_width=True)

    # Location-based risk analysis
    st.subheader("⚠️ Risk Analysis by Location")
    location_risk = data.groupby('Location')['Risk_Score'].agg(['mean', 'count']).round(1)
    location_risk.columns = ['Avg Risk Score', 'Employee Count']
    location_risk = location_risk.sort_values('Avg Risk Score', ascending=False)

    # Continuing from the location_remote_analysis function
    fig = px.scatter(location_risk, x='Employee Count', y='Avg Risk Score',
                     text=location_risk.index, size='Employee Count',
                     title="Risk Score vs Employee Count by Location")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)


def advanced_risk_analysis(data):
    """Advanced risk analysis with multiple dimensions"""
    st.header("⚠️ Advanced Risk Analysis")

    # Risk distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Risk Score Distribution")
        fig = px.histogram(data, x='Risk_Score', nbins=20, color='Risk_Category',
                           title="Distribution of Risk Scores")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Risk Categories")
        risk_counts = data['Risk_Category'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                     title="Risk Category Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # High-risk employee analysis
    st.subheader("🚨 High-Risk Employee Analysis")

    high_risk_data = data[data['Risk_Category'].isin(['High', 'Critical'])]

    if not high_risk_data.empty:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**By Department**")
            dept_risk = high_risk_data['Department'].value_counts()
            st.bar_chart(dept_risk)

        with col2:
            st.markdown("**By Job Level**")
            level_risk = high_risk_data['Job_Level'].value_counts()
            st.bar_chart(level_risk)

        with col3:
            st.markdown("**By Tenure**")
            tenure_risk = high_risk_data['Tenure_Years'].describe()
            st.dataframe(tenure_risk)

        # Detailed high-risk table
        st.subheader("📋 High-Risk Employees Detail")
        high_risk_display = high_risk_data[['Employee_ID', 'Department', 'Job_Level',
                                            'Risk_Score', 'Engagement_Score', 'Satisfaction_Score',
                                            'Tenure_Years', 'Performance_Rating_2024']].sort_values('Risk_Score',
                                                                                                    ascending=False)
        st.dataframe(high_risk_display, use_container_width=True)
    else:
        st.info("No high-risk employees found in the current dataset.")


def predictive_insights(data):
    """Predictive insights and recommendations"""
    st.header("🔮 Predictive Insights & Recommendations")

    # Turnover prediction factors
    st.subheader("📈 Key Turnover Predictors")

    # Calculate correlation with turnover
    turnover_factors = data[['Has_Left', 'Engagement_Score', 'Satisfaction_Score',
                             'Work_Life_Balance', 'Relationship_with_Manager',
                             'Career_Development_Opportunities', 'Performance_Rating_2024',
                             'Tenure_Years', 'Age', 'Total_Comp_Rank']].corr()['Has_Left'].sort_values()

    fig = px.bar(x=turnover_factors.values[:-1], y=turnover_factors.index[:-1],
                 orientation='h', title="Correlation with Turnover")
    st.plotly_chart(fig, use_container_width=True)

    # Actionable insights
    st.subheader("💡 Actionable Insights")

    insights = []

    # Low engagement insight
    low_engagement = len(data[data['Engagement_Score'] < 5])
    if low_engagement > 0:
        insights.append(
            f"🔴 {low_engagement} employees have low engagement scores (<5). Consider targeted engagement initiatives.")

    # Performance trend insight
    declining_perf = len(data[data['Performance_Trend'] < -0.5])
    if declining_perf > 0:
        insights.append(
            f"📉 {declining_perf} employees show declining performance. Implement performance improvement plans.")

    # Promotion stagnation insight
    stagnant_promotions = len(data[(data['Months_Since_Promotion'] > 36) & (data['Has_Been_Promoted'])])
    if stagnant_promotions > 0:
        insights.append(
            f"⏰ {stagnant_promotions} employees haven't been promoted in 3+ years. Review career development paths.")

    # Compensation insight
    low_comp = len(data[data['Total_Comp_Rank'] < 0.25])
    if low_comp > 0:
        insights.append(
            f"💰 {low_comp} employees are in the bottom compensation quartile. Consider compensation review.")

    # Training insight
    low_training = len(data[data['Training_Hours_Last_Year'] < 20])
    if low_training > 0:
        insights.append(
            f"📚 {low_training} employees received minimal training (<20 hours). Increase development opportunities.")

    for i, insight in enumerate(insights, 1):
        st.markdown(f"**{i}.** {insight}")

    # Recommendations by department
    st.subheader("🎯 Department-Specific Recommendations")

    dept_recommendations = {}
    for dept in data['Department'].unique():
        dept_data = data[data['Department'] == dept]
        avg_risk = dept_data['Risk_Score'].mean()
        high_risk_count = len(dept_data[dept_data['Risk_Category'].isin(['High', 'Critical'])])

        if avg_risk > 50:
            dept_recommendations[
                dept] = f"High average risk ({avg_risk:.1f}). Focus on engagement and retention strategies."
        elif high_risk_count > 5:
            dept_recommendations[dept] = f"{high_risk_count} high-risk employees. Implement targeted interventions."
        else:
            dept_recommendations[dept] = "Department performing well. Continue current practices."

    for dept, rec in dept_recommendations.items():
        st.markdown(f"**{dept}:** {rec}")


def engagement_deep_dive(data):
    """Deep dive into engagement and satisfaction metrics"""
    st.header("💫 Engagement Deep Dive")

    # Engagement vs other factors
    st.subheader("📊 Engagement Correlation Matrix")

    engagement_cols = ['Engagement_Score', 'Satisfaction_Score', 'Work_Life_Balance',
                       'Relationship_with_Manager', 'Career_Development_Opportunities',
                       'Performance_Rating_2024', 'Risk_Score']

    corr_matrix = data[engagement_cols].corr()

    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Engagement Factors Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    # Engagement trends by demographics
    st.subheader("👥 Engagement by Demographics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Engagement by Age Group**")
        age_engagement = data.groupby('Age_Group')['Engagement_Score'].mean().sort_values(ascending=False)
        st.bar_chart(age_engagement)

    with col2:
        st.markdown("**Engagement by Tenure**")
        # Create tenure bins for analysis
        data['Tenure_Bins'] = pd.cut(data['Tenure_Years'], bins=[0, 1, 3, 5, 10, 100],
                                     labels=['<1yr', '1-3yrs', '3-5yrs', '5-10yrs', '10+yrs'])
        tenure_engagement = data.groupby('Tenure_Bins')['Engagement_Score'].mean()
        st.bar_chart(tenure_engagement)

    # Satisfaction drivers
    st.subheader("🎯 Satisfaction Drivers Analysis")

    # Identify top and bottom performers in each satisfaction dimension
    satisfaction_metrics = ['Work_Life_Balance', 'Relationship_with_Manager',
                            'Career_Development_Opportunities']

    for metric in satisfaction_metrics:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Top 10% in {metric.replace('_', ' ')}**")
            top_performers = data[data[metric] >= data[metric].quantile(0.9)]
            st.metric("Average Engagement", f"{top_performers['Engagement_Score'].mean():.2f}")
            st.metric("Average Risk Score", f"{top_performers['Risk_Score'].mean():.1f}")

        with col2:
            st.markdown(f"**Bottom 10% in {metric.replace('_', ' ')}**")
            bottom_performers = data[data[metric] <= data[metric].quantile(0.1)]
            st.metric("Average Engagement", f"{bottom_performers['Engagement_Score'].mean():.2f}")
            st.metric("Average Risk Score", f"{bottom_performers['Risk_Score'].mean():.1f}")


def main():
    """Main application function"""
    st.title("🚀 Enhanced Employee Analytics Dashboard")
    st.markdown("*Comprehensive workforce analytics with advanced risk assessment*")

    # Load data
    with st.spinner("Loading employee data..."):
        data = load_and_prepare_enhanced_data()

    # Sidebar filters
    filters = enhanced_filters(data)
    filtered_data = apply_enhanced_filters(data, filters)

    # Display filter results
    st.sidebar.markdown(f"**Filtered Results:** {len(filtered_data)} employees")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Executive Summary",
        "💰 Compensation",
        "🚀 Career Progression",
        "🌍 Location Analysis",
        "⚠️ Risk Analysis",
        "🔮 Predictive Insights",
        "💫 Engagement Deep Dive"
    ])

    with tab1:
        enhanced_executive_summary(filtered_data)

    with tab2:
        compensation_analysis(filtered_data)

    with tab3:
        career_progression_analysis(filtered_data)

    with tab4:
        location_remote_analysis(filtered_data)

    with tab5:
        advanced_risk_analysis(filtered_data)

    with tab6:
        predictive_insights(filtered_data)

    with tab7:
        engagement_deep_dive(filtered_data)

    # Footer
    st.markdown("---")
    st.markdown("*Enhanced Employee Analytics Dashboard - Powered by Advanced Data Science*")


if __name__ == "__main__":
    main()