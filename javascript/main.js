// Import required libraries
import { pipeline } from "@huggingface/transformers"; // Hugging Face Transformers for Node.js
import fetch from "node-fetch"; // For fetching API data

// API URLs (Replace with actual endpoints)
const SALARY_API_URL = "https://api.mockdatasalary.com/salaries";
const DEMAND_API_URL = "https://api.mockjobdemand.com/demand";
const GEOGRAPHIC_API_URL = "https://api.mockgeographic.com/factors";
const SKILLS_API_URL = "https://api.mockskills.com/premiums";

// Utility Functions for Models

// Compound Annual Growth Rate (CAGR)
const calculateCAGR = (futureValue, presentValue, years) => {
    return Math.pow(futureValue / presentValue, 1 / years) - 1;
};

// Inflation Adjustment
const adjustForInflation = (salary, inflationRate) => {
    return salary * (1 + inflationRate);
};

// Skills Premium Adjustment
const adjustForSkillsPremium = (baseSalary, skillFactors) => {
    const totalPremium = Object.values(skillFactors).reduce((a, b) => a + b, 0);
    return baseSalary * (1 + totalPremium);
};

// Demand and Geographic Growth Adjustment
const adjustForDemandAndGeography = (
    baseSalary,
    demandFactor,
    geographicFactor
) => {
    return baseSalary * (1 + demandFactor + geographicFactor);
};

// Fetch Data from APIs
const fetchData = async (url) => {
    try {
        const response = await fetch(url);
        if (response.ok) {
            return await response.json();
        } else {
            throw new Error(`Failed to fetch data from ${url}`);
        }
    } catch (error) {
        console.error(error.message);
        return null;
    }
};

// Initialize the BERT-based named entity recognition pipeline
async function initializeBERT() {
    const nerPipeline = await pipeline("ner", "Xenova/bert-base-cased");
    return nerPipeline;
}

// Extract salary details using BERT
async function extractSalaryDetails(nerPipeline, jobDescription) {
    const nerResults = await nerPipeline(jobDescription);
    const extractedData = {
        Role: [],
        Level: [],
        SalaryRange: null,
    };

    nerResults.forEach((entity) => {
        const text = entity.word;
        if (entity.entity.startsWith("B-")) {
            if (entity.entity.includes("ROLE")) extractedData.Role.push(text);
            else if (entity.entity.includes("LEVEL"))
                extractedData.Level.push(text);
            else if (text.includes("$")) extractedData.SalaryRange = text;
        }
    });

    return {
        Role: extractedData.Role.join(" "),
        Level: extractedData.Level.join(" "),
        SalaryRange: extractedData.SalaryRange,
    };
}

// Main function to integrate BERT and salary prediction models
async function main() {
    // Fetch data from APIs with fallback to mock data
    const salaryData = (await fetchData(SALARY_API_URL)) || [
        {
            Role: "Data Analyst",
            "Entry-Level 2024": 70000,
            "Mid-Level 2024": 95000,
            "Senior-Level 2024": 120000,
        },
        {
            Role: "Data Scientist",
            "Entry-Level 2024": 90000,
            "Mid-Level 2024": 120000,
            "Senior-Level 2024": 150000,
        },
    ];

    const demandData = (await fetchData(DEMAND_API_URL)) || [
        { Role: "Data Analyst", DemandFactor: 0.1 },
        { Role: "Data Scientist", DemandFactor: 0.12 },
    ];

    const geographicData = (await fetchData(GEOGRAPHIC_API_URL)) || [
        { Role: "Data Analyst", GeographicFactor: 0.05 },
        { Role: "Data Scientist", GeographicFactor: 0.07 },
    ];

    const skillFactors = (await fetchData(SKILLS_API_URL)) || {
        Python: 0.05,
        SQL: 0.03,
        MachineLearning: 0.02,
    };

    // Mock job descriptions
    const jobDescriptions = [
        "Looking for an entry-level Data Scientist. The salary range is $90,000–$120,000.",
        "Hiring a senior Data Engineer with experience in cloud platforms. Salary up to $175,000.",
    ];

    // Initialize the BERT pipeline
    const nerPipeline = await initializeBERT();

    // Process each job description with BERT
    const bertResults = [];
    for (const description of jobDescriptions) {
        const result = await extractSalaryDetails(nerPipeline, description);
        bertResults.push(result);
    }

    console.log("BERT Extracted Data:");
    console.table(bertResults);

    // Merge Data and Apply Models
    const mergedData = salaryData.map((role) => {
        const demandFactor =
            demandData.find((d) => d.Role === role.Role)?.DemandFactor || 0;
        const geographicFactor =
            geographicData.find((g) => g.Role === role.Role)
                ?.GeographicFactor || 0;
        return {
            ...role,
            DemandFactor: demandFactor,
            GeographicFactor: geographicFactor,
        };
    });

    const inflationRate = 0.025; // 2.5% inflation rate

    // Calculate 2025 salaries
    const results = mergedData.map((role) => {
        // Apply inflation adjustment
        const entryInflated = adjustForInflation(
            role["Entry-Level 2024"],
            inflationRate
        );
        const midInflated = adjustForInflation(
            role["Mid-Level 2024"],
            inflationRate
        );
        const seniorInflated = adjustForInflation(
            role["Senior-Level 2024"],
            inflationRate
        );

        // Apply skills premium adjustment
        const entryWithSkills = adjustForSkillsPremium(
            entryInflated,
            skillFactors
        );
        const midWithSkills = adjustForSkillsPremium(midInflated, skillFactors);
        const seniorWithSkills = adjustForSkillsPremium(
            seniorInflated,
            skillFactors
        );

        // Apply demand and geographic adjustments
        const entryFinal = adjustForDemandAndGeography(
            entryWithSkills,
            role.DemandFactor,
            role.GeographicFactor
        );
        const midFinal = adjustForDemandAndGeography(
            midWithSkills,
            role.DemandFactor,
            role.GeographicFactor
        );
        const seniorFinal = adjustForDemandAndGeography(
            seniorWithSkills,
            role.DemandFactor,
            role.GeographicFactor
        );

        return {
            Role: role.Role,
            "Entry-Level 2025": entryFinal.toFixed(2),
            "Mid-Level 2025": midFinal.toFixed(2),
            "Senior-Level 2025": seniorFinal.toFixed(2),
        };
    });

    // Display Results
    console.log("Predicted Salaries for 2025:");
    console.table(results);
}

// Run the main function
main();
