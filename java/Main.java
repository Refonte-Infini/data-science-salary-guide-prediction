import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.*;
import org.json.JSONArray;
import org.json.JSONObject;

public class Main {

    // Fetch data from API with fallback to mock data
    public static JSONArray fetchData(String apiUrl, JSONArray fallbackData) {
        try {
            URL url = new URL(apiUrl);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            int responseCode = connection.getResponseCode();

            if (responseCode == 200) {
                BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                String inputLine;
                StringBuilder response = new StringBuilder();

                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();
                return new JSONArray(response.toString());
            } else {
                System.out.println("API call failed. Using fallback data.");
            }
        } catch (Exception e) {
            System.out.println("Error fetching data: " + e.getMessage());
        }
        return fallbackData;
    }

    // Simulate BERT-based extraction via an API
    public static JSONObject extractSalaryDetails(String jobDescription) {
        // Mock BERT API response
        JSONObject bertResult = new JSONObject();
        if (jobDescription.contains("Data Scientist")) {
            bertResult.put("Role", "Data Scientist");
            bertResult.put("Level", "Entry-Level");
            bertResult.put("SalaryRange", "$90,000–$120,000");
        } else if (jobDescription.contains("Data Engineer")) {
            bertResult.put("Role", "Data Engineer");
            bertResult.put("Level", "Senior");
            bertResult.put("SalaryRange", "$175,000");
        }
        return bertResult;
    }

    // Models

    // CAGR Calculation
    public static double calculateCAGR(double presentValue, double futureValue, int years) {
        return Math.pow(futureValue / presentValue, 1.0 / years) - 1.0;
    }

    // Inflation Adjustment
    public static double adjustForInflation(double salary, double inflationRate) {
        return salary * (1 + inflationRate);
    }

    // Skills Premium Adjustment
    public static double adjustForSkillsPremium(double baseSalary, Map<String, Double> skillFactors) {
        double totalPremium = skillFactors.values().stream().mapToDouble(Double::doubleValue).sum();
        return baseSalary * (1 + totalPremium);
    }

    // Demand and Geographic Growth Adjustment
    public static double adjustForDemandAndGeography(double baseSalary, double demandFactor, double geographicFactor) {
        return baseSalary * (1 + demandFactor + geographicFactor);
    }

    // Main function
    public static void main(String[] args) {
        // Mock data
        JSONArray salaryMock = new JSONArray("[{\"Role\":\"Data Analyst\",\"Entry-Level 2024\":70000,\"Mid-Level 2024\":95000,\"Senior-Level 2024\":120000},{\"Role\":\"Data Scientist\",\"Entry-Level 2024\":90000,\"Mid-Level 2024\":120000,\"Senior-Level 2024\":150000}]");
        JSONArray demandMock = new JSONArray("[{\"Role\":\"Data Analyst\",\"Demand Factor\":0.1},{\"Role\":\"Data Scientist\",\"Demand Factor\":0.12}]");
        JSONArray geographicMock = new JSONArray("[{\"Role\":\"Data Analyst\",\"Geographic Factor\":0.05},{\"Role\":\"Data Scientist\",\"Geographic Factor\":0.07}]");
        Map<String, Double> skillsMock = Map.of("Python", 0.05, "SQL", 0.03, "Machine Learning", 0.02);

        // Mock job descriptions
        List<String> jobDescriptions = Arrays.asList(
                "Looking for an entry-level Data Scientist. The salary range is $90,000–$120,000.",
                "Hiring a senior Data Engineer with experience in cloud platforms. Salary up to $175,000."
        );

        // Fetch data
        JSONArray salaryData = fetchData("https://api.mockdatasalary.com/salaries", salaryMock);
        JSONArray demandData = fetchData("https://api.mockjobdemand.com/demand", demandMock);
        JSONArray geographicData = fetchData("https://api.mockgeographic.com/factors", geographicMock);

        // Inflation rate
        double inflationRate = 0.025;

        // Process job descriptions with BERT
        boolean useBert = false;
        if(useBert) {
            List<JSONObject> bertResults = new ArrayList<>();
            for (String description : jobDescriptions) {
                bertResults.add(extractSalaryDetails(description));
            }

            System.out.println("BERT Extracted Data:");
            bertResults.forEach(System.out::println);
        }

        // Calculate 2025 salaries
        List<Map<String, Object>> results = new ArrayList<>();
        for (int i = 0; i < salaryData.length(); i++) {
            JSONObject roleData = salaryData.getJSONObject(i);
            String role = roleData.getString("Role");

            double entrySalary2024 = roleData.getDouble("Entry-Level 2024");
            double midSalary2024 = roleData.getDouble("Mid-Level 2024");
            double seniorSalary2024 = roleData.getDouble("Senior-Level 2024");

            double demandFactor = demandData.toList().stream()
                    .map(d -> (Map<?, ?>) d)
                    .filter(d -> d.get("Role").equals(role))
                    .map(d -> (double) d.get("Demand Factor"))
                    .findFirst().orElse(0.0);

            double geographicFactor = geographicData.toList().stream()
                    .map(g -> (Map<?, ?>) g)
                    .filter(g -> g.get("Role").equals(role))
                    .map(g -> (double) g.get("Geographic Factor"))
                    .findFirst().orElse(0.0);

            // Apply CAGR based on historical data
            double entryCAGR = calculateCAGR(entrySalary2024 * 0.95, entrySalary2024, 1);
            double midCAGR = calculateCAGR(midSalary2024 * 0.9, midSalary2024, 1);
            double seniorCAGR = calculateCAGR(seniorSalary2024 * 0.85, seniorSalary2024, 1);

            // Apply models
            double entryInflated = adjustForInflation(entrySalary2024, inflationRate + entryCAGR);
            double midInflated = adjustForInflation(midSalary2024, inflationRate + midCAGR);
            double seniorInflated = adjustForInflation(seniorSalary2024, inflationRate + seniorCAGR);

            double entryWithSkills = adjustForSkillsPremium(entryInflated, skillsMock);
            double midWithSkills = adjustForSkillsPremium(midInflated, skillsMock);
            double seniorWithSkills = adjustForSkillsPremium(seniorInflated, skillsMock);

            double entryFinal = adjustForDemandAndGeography(entryWithSkills, demandFactor, geographicFactor);
            double midFinal = adjustForDemandAndGeography(midWithSkills, demandFactor, geographicFactor);
            double seniorFinal = adjustForDemandAndGeography(seniorWithSkills, demandFactor, geographicFactor);

            // Store results
            Map<String, Object> result = new HashMap<>();
            result.put("Role", role);
            result.put("Entry-Level 2025", entryFinal);
            result.put("Mid-Level 2025", midFinal);
            result.put("Senior-Level 2025", seniorFinal);
            results.add(result);
        }

        // Display results
        System.out.println("Predicted Salaries for 2025:");
        results.forEach(System.out::println);
    }
}
