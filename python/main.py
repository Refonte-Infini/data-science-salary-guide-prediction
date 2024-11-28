import requests
import pandas as pd
from transformers import pipeline

# API URLs (Replace with actual endpoints)
SALARY_API_URL = "https://api.mockdatasalary.com/salaries"
DEMAND_API_URL = "https://api.mockjobdemand.com/demand"
GEOGRAPHIC_API_URL = "https://api.mockgeographic.com/factors"
SKILLS_API_URL = "https://api.mockskills.com/premiums"

# Utility functions for fetching data with fallback
def fetch_data(api_url, fallback_data):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Failed to fetch data from {api_url}. Using fallback data.")
            return pd.DataFrame(fallback_data)
    except Exception as e:
        print(f"Error fetching data from {api_url}: {e}. Using fallback data.")
        return pd.DataFrame(fallback_data)

# Models
def calculate_cagr(fv, pv, n):
    """Compound Annual Growth Rate (CAGR)"""
    return (fv / pv) ** (1 / n) - 1

def inflation_adjustment(salary, inflation_rate):
    """Adjust salary for inflation"""
    return salary * (1 + inflation_rate)

def skills_premium_adjustment(base_salary, skill_factors):
    """Adjust salary for skills premium"""
    total_premium = sum(skill_factors.values())
    return base_salary * (1 + total_premium)

def demand_geographic_adjustment(base_salary, demand_factor, geographic_factor):
    """Adjust salary for demand and geography"""
    return base_salary * (1 + demand_factor + geographic_factor)

def weighted_regression(base_salary, experience, skills_premium, location_factor):
    """Predict salary using regression"""
    return base_salary + 10000 * experience + 20000 * skills_premium + 30000 * location_factor

# Initialize the BERT pipeline
def initialize_bert():
    return pipeline("ner", model="google-bert/bert-base-cased")

def extract_salary_details(bert_pipeline, job_description):
    """Use BERT to extract role, level, and salary range from job descriptions"""
    ner_results = bert_pipeline(job_description)
    extracted_data = {"Role": [], "Level": [], "SalaryRange": None}

    for entity in ner_results:
        word = entity["word"]
        if "ROLE" in entity["entity"]:
            extracted_data["Role"].append(word)
        elif "LEVEL" in entity["entity"]:
            extracted_data["Level"].append(word)
        elif "$" in word:
            if extracted_data["SalaryRange"] is None:
                extracted_data["SalaryRange"] = word
            else:
                extracted_data["SalaryRange"] += word

    return {
        "Role": " ".join(extracted_data["Role"]),
        "Level": " ".join(extracted_data["Level"]),
        "SalaryRange": extracted_data["SalaryRange"],
    }

# Mock Data
salary_mock = [
    {"Role": "Data Analyst", "Entry-Level 2024": 70000, "Mid-Level 2024": 95000, "Senior-Level 2024": 120000},
    {"Role": "Data Scientist", "Entry-Level 2024": 90000, "Mid-Level 2024": 120000, "Senior-Level 2024": 150000},
]
demand_mock = [{"Role": "Data Analyst", "Demand Factor": 0.1}, {"Role": "Data Scientist", "Demand Factor": 0.12}]
geographic_mock = [
    {"Role": "Data Analyst", "Geographic Factor": 0.05},
    {"Role": "Data Scientist", "Geographic Factor": 0.07},
]
skills_mock = {"Python": 0.05, "SQL": 0.03, "MachineLearning": 0.02}

# Mock Job Descriptions
job_descriptions = [
    "Looking for an entry-level Data Scientist. The salary range is $90,000–$120,000.",
    "Hiring a senior Data Engineer with experience in cloud platforms. Salary up to $175,000.",
]

# Main function
def main():
    # Fetch data from APIs with fallbacks
    salary_data = fetch_data(SALARY_API_URL, salary_mock)
    demand_data = fetch_data(DEMAND_API_URL, demand_mock)
    geographic_data = fetch_data(GEOGRAPHIC_API_URL, geographic_mock)
    skills_data = skills_mock  # No need for fallback; directly used

    use_bert = False
    if use_bert:
         # Initialize BERT pipeline
        bert_pipeline = initialize_bert()

        # Process job descriptions with BERT
        bert_results = []
        for description in job_descriptions:
            bert_results.append(extract_salary_details(bert_pipeline, description))

        print("BERT Extracted Data:")
        print(pd.DataFrame(bert_results))

    # Merge data
    merged_data = pd.merge(salary_data, demand_data, on="Role")
    merged_data = pd.merge(merged_data, geographic_data, on="Role")

    # Calculate 2025 salaries
    inflation_rate = 0.025
    entry_2025, mid_2025, senior_2025 = [], [], []

    for _, row in merged_data.iterrows():
        entry_inflated = inflation_adjustment(row["Entry-Level 2024"], inflation_rate)
        mid_inflated = inflation_adjustment(row["Mid-Level 2024"], inflation_rate)
        senior_inflated = inflation_adjustment(row["Senior-Level 2024"], inflation_rate)

        entry_skills = skills_premium_adjustment(entry_inflated, skills_data)
        mid_skills = skills_premium_adjustment(mid_inflated, skills_data)
        senior_skills = skills_premium_adjustment(senior_inflated, skills_data)

        entry_final = demand_geographic_adjustment(entry_skills, row["Demand Factor"], row["Geographic Factor"])
        mid_final = demand_geographic_adjustment(mid_skills, row["Demand Factor"], row["Geographic Factor"])
        senior_final = demand_geographic_adjustment(senior_skills, row["Demand Factor"], row["Geographic Factor"])

        entry_2025.append(entry_final)
        mid_2025.append(mid_final)
        senior_2025.append(senior_final)

    # Add results to DataFrame
    merged_data["Entry-Level 2025"] = entry_2025
    merged_data["Mid-Level 2025"] = mid_2025
    merged_data["Senior-Level 2025"] = senior_2025

    # Display final results
    print("Predicted Salaries for 2025:")
    print(merged_data[["Role", "Entry-Level 2025", "Mid-Level 2025", "Senior-Level 2025"]])

# Run the program
if __name__ == "__main__":
    main()
