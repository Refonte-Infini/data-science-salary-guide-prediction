import requests
import pandas as pd
from transformers import pipeline

# Mock API URLs (Replace these with real API endpoints)
SALARY_API_URL = "https://api.mockdatasalary.com/salaries"
DEMAND_API_URL = "https://api.mockjobdemand.com/demand"
GEOGRAPHIC_API_URL = "https://api.mockgeographic.com/factors"
SKILLS_API_URL = "https://api.mockskills.com/premiums"

# Step 1: Fetch salary data
def fetch_salary_data():
    response = requests.get(SALARY_API_URL)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception("Failed to fetch salary data from the API")

# Step 2: Fetch demand adjustment data
def fetch_demand_data():
    response = requests.get(DEMAND_API_URL)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception("Failed to fetch demand data from the API")

# Step 3: Fetch geographic adjustment data
def fetch_geographic_data():
    response = requests.get(GEOGRAPHIC_API_URL)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception("Failed to fetch geographic adjustment data from the API")

# Step 4: Fetch skills premium data
def fetch_skills_data():
    response = requests.get(SKILLS_API_URL)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch skills premium data from the API")

# Step 5: Apply the BERT model for role-specific salary extraction
def extract_salary_details(description):
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    ner_results = ner_pipeline(description)
    # Extract salary-related entities
    roles, levels, salary_range = [], [], None
    for entity in ner_results:
        if "Role" in entity["entity"]:
            roles.append(entity["word"])
        elif "Level" in entity["entity"]:
            levels.append(entity["word"])
        elif "$" in entity["word"]:  # Salary range detection
            salary_range = entity["word"] if not salary_range else f"{salary_range}{entity['word']}"
    return {"Role": " ".join(roles), "Level": " ".join(levels), "Salary Range": salary_range}

# Step 6: Compound Annual Growth Rate (CAGR) model
def cagr(fv, pv, n):
    """Calculate CAGR"""
    return (fv / pv) ** (1 / n) - 1

# Inflation Adjustment model
def inflation_adjustment(salary, inflation_rate):
    """Adjust salary for inflation"""
    return salary * (1 + inflation_rate)

# Skills Premium model
def skills_premium_adjustment(base_salary, skill_factors):
    """Adjust salary for skills premium"""
    total_premium = sum(skill_factors.values())
    return base_salary * (1 + total_premium)

# Demand and Geographic Growth model
def demand_geographic_adjustment(base_salary, demand_factor, geographic_factor):
    """Adjust salary for demand and geography"""
    return base_salary * (1 + demand_factor + geographic_factor)

# Weighted Regression model (example coefficients)
def weighted_regression(experience, skills_premium, location_factor, base_salary):
    """Predict salary using regression"""
    return base_salary + 10000 * experience + 20000 * skills_premium + 30000 * location_factor

# Fetch and prepare data
try:
    salary_data = fetch_salary_data()
    demand_data = fetch_demand_data()
    geographic_data = fetch_geographic_data()
    skills_data = fetch_skills_data()
except Exception as e:
    print(f"Error fetching data: {e}")
    # Mock fallback data
    salary_data = pd.DataFrame({
        "Role": ["Data Analyst", "Data Scientist", "BI Analyst", "Data Engineer", "ML Engineer"],
        "Entry-Level 2024": [70000, 90000, 75000, 85000, 95000],
        "Mid-Level 2024": [95000, 120000, 100000, 110000, 130000],
        "Senior-Level 2024": [120000, 150000, 130000, 140000, 175000]
    })
    demand_data = pd.DataFrame({
        "Role": ["Data Analyst", "Data Scientist", "BI Analyst", "Data Engineer", "ML Engineer"],
        "Demand Factor": [0.1, 0.12, 0.08, 0.09, 0.11]
    })
    geographic_data = pd.DataFrame({
        "Role": ["Data Analyst", "Data Scientist", "BI Analyst", "Data Engineer", "ML Engineer"],
        "Geographic Factor": [0.05, 0.07, 0.04, 0.06, 0.08]
    })
    skills_data = {"Python": 0.05, "SQL": 0.03, "Machine Learning": 0.02}

# Merge dataframes
merged_data = salary_data.merge(demand_data, on="Role").merge(geographic_data, on="Role")

# Apply all models to calculate 2025 salaries
entry_level_2025, mid_level_2025, senior_level_2025 = [], [], []

for _, row in merged_data.iterrows():
    entry_inflated = inflation_adjustment(row["Entry-Level 2024"], 0.025)
    mid_inflated = inflation_adjustment(row["Mid-Level 2024"], 0.025)
    senior_inflated = inflation_adjustment(row["Senior-Level 2024"], 0.025)
    
    entry_skills = skills_premium_adjustment(entry_inflated, skills_data)
    mid_skills = skills_premium_adjustment(mid_inflated, skills_data)
    senior_skills = skills_premium_adjustment(senior_inflated, skills_data)
    
    entry_final = demand_geographic_adjustment(entry_skills, row["Demand Factor"], row["Geographic Factor"])
    mid_final = demand_geographic_adjustment(mid_skills, row["Demand Factor"], row["Geographic Factor"])
    senior_final = demand_geographic_adjustment(senior_skills, row["Demand Factor"], row["Geographic Factor"])
    
    entry_level_2025.append(entry_final)
    mid_level_2025.append(mid_final)
    senior_level_2025.append(senior_final)

# Add results to DataFrame
merged_data["Entry-Level 2025"] = entry_level_2025
merged_data["Mid-Level 2025"] = mid_level_2025
merged_data["Senior-Level 2025"] = senior_level_2025

# Select output columns
output_columns = ["Role", "Entry-Level 2025", "Mid-Level 2025", "Senior-Level 2025"]
final_table = merged_data[output_columns]

# Display the table
print(final_table)
