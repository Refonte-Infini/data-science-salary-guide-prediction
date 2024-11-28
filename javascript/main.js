// Import required libraries
import { pipeline } from "@huggingface/transformers"; // Hugging Face Transformers for Node.js
import fetch from "node-fetch"; // For fetching API data

// API URLs (Replace with actual endpoints to get full data)
const SALARY_API_URL = "https://api.mockdatasalary.com/salaries";
const DEMAND_API_URL = "https://api.mockjobdemand.com/demand";
const GEOGRAPHIC_API_URL = "https://api.mockgeographic.com/factors";
const SKILLS_API_URL = "https://api.mockskills.com/premiums";

// Utility Functions for Models

// CAGR Calculation
const calculateCAGR = (presentValue, futureValue, years) => {
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

// Initialize BERT Pipeline
const initializeBERT = async () => {
    return await pipeline("ner", "bert-base-cased");
};

// Use BERT to Extract Salary Details
const extractSalaryDetails = async (nerPipeline, jobDescription) => {
    const nerResults = await nerPipeline(jobDescription);
    const extractedData = {
        Role: [],
        Level: [],
        SalaryRange: null,
    };

    nerResults.forEach((entity) => {
        const word = entity.word;
        if (entity.entity.startsWith("B-")) {
            if (entity.entity.includes("ROLE")) extractedData.Role.push(word);
            else if (entity.entity.includes("LEVEL"))
                extractedData.Level.push(word);
            else if (word.includes("$")) {
                extractedData.SalaryRange = extractedData.SalaryRange
                    ? `${extractedData.SalaryRange}${word}`
                    : word;
            }
        }
    });

    return {
        Role: extractedData.Role.join(" "),
        Level: extractedData.Level.join(" "),
        SalaryRange: extractedData.SalaryRange,
    };
};

// Fetch Data from APIs
const fetchData = async (url, fallbackData) => {
    try {
        const response = await fetch(url);
        if (response.ok) {
            return await response.json();
        } else {
            console.error(
                `Failed to fetch data from ${url}. Using fallback data.`
            );
            return fallbackData;
        }
    } catch (error) {
        console.error(`Error fetching data from ${url}: ${error.message}`);
        return fallbackData;
    }
};

// Main function to integrate BERT, CAGR, and other models
const main = async () => {
    // Mock data for fallbacks
    const salaryMock = [
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
    const demandMock = [
        { Role: "Data Analyst", "Demand Factor": 0.1 },
        { Role: "Data Scientist", "Demand Factor": 0.12 },
    ];
    const geographicMock = [
        { Role: "Data Analyst", "Geographic Factor": 0.05 },
        { Role: "Data Scientist", "Geographic Factor": 0.07 },
    ];
    const skillsMock = { Python: 0.05, SQL: 0.03, MachineLearning: 0.02 };

    // Fetch data from APIs or use fallback data
    const salaryData = await fetchData(SALARY_API_URL, salaryMock);
    const demandData = await fetchData(DEMAND_API_URL, demandMock);
    const geographicData = await fetchData(GEOGRAPHIC_API_URL, geographicMock);
    const skillsData = skillsMock;

    // Mock job descriptions
    const jobDescriptions = [
        "Looking for an entry-level Data Scientist. The salary range is $90,000–$120,000.",
        "Hiring a senior Data Engineer with experience in cloud platforms. Salary up to $175,000.",
    ];

    const use_bert = false;
    if (use_bert) {
        // Initialize BERT pipeline
        const nerPipeline = await initializeBERT();

        // Process each job description with BERT
        const bertResults = [];
        for (const description of jobDescriptions) {
            const result = await extractSalaryDetails(nerPipeline, description);
            bertResults.push(result);
        }

        console.log("BERT Extracted Data:");
        console.table(bertResults);
    }

    // Inflation rate
    const inflationRate = 0.025; // 2.5% inflation

    // Calculate 2025 salaries
    const results = salaryData.map((roleData) => {
        const role = roleData.Role;

        const entrySalary2024 = roleData["Entry-Level 2024"];
        const midSalary2024 = roleData["Mid-Level 2024"];
        const seniorSalary2024 = roleData["Senior-Level 2024"];

        const demandFactor =
            demandData.find((d) => d.Role === role)?.["Demand Factor"] || 0;
        const geographicFactor =
            geographicData.find((g) => g.Role === role)?.[
                "Geographic Factor"
            ] || 0;

        // Apply CAGR based on historical data
        const entryCAGR = calculateCAGR(
            entrySalary2024 * 0.95,
            entrySalary2024,
            1
        ); // Assuming 5% increase historically
        const midCAGR = calculateCAGR(midSalary2024 * 0.9, midSalary2024, 1); // Assuming 10% increase historically
        const seniorCAGR = calculateCAGR(
            seniorSalary2024 * 0.85,
            seniorSalary2024,
            1
        ); // Assuming 15% increase historically

        // Inflation-adjusted salaries
        const entryInflated = adjustForInflation(
            entrySalary2024,
            inflationRate + entryCAGR
        );
        const midInflated = adjustForInflation(
            midSalary2024,
            inflationRate + midCAGR
        );
        const seniorInflated = adjustForInflation(
            seniorSalary2024,
            inflationRate + seniorCAGR
        );

        // Skills premium adjustments
        const entryWithSkills = adjustForSkillsPremium(
            entryInflated,
            skillsData
        );
        const midWithSkills = adjustForSkillsPremium(midInflated, skillsData);
        const seniorWithSkills = adjustForSkillsPremium(
            seniorInflated,
            skillsData
        );

        // Demand and geographic adjustments
        const entryFinal = adjustForDemandAndGeography(
            entryWithSkills,
            demandFactor,
            geographicFactor
        );
        const midFinal = adjustForDemandAndGeography(
            midWithSkills,
            demandFactor,
            geographicFactor
        );
        const seniorFinal = adjustForDemandAndGeography(
            seniorWithSkills,
            demandFactor,
            geographicFactor
        );

        return {
            Role: role,
            "Entry-Level 2025": entryFinal.toFixed(2),
            "Mid-Level 2025": midFinal.toFixed(2),
            "Senior-Level 2025": seniorFinal.toFixed(2),
        };
    });

    // Display Results
    console.log("Predicted Salaries for 2025:");
    console.table(results);
};

// Run the main function
main();
