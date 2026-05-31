#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>

namespace py = pybind11;

class DataCalculator {
public:
    // Simple default constructor since we no longer load CSVs
    DataCalculator() = default;

    /**
     * FULL ALTMAN Z-SCORE
     * Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
     */
    double calculate_full_z_score(const std::map<std::string, double>& data) {
        try {
            double total_assets = data.at("Assets");
            if (total_assets <= 0) return 0.0;

            // X1: Working Capital / Total Assets (Liquidity)
            double x1 = (data.at("AssetsCurrent") - data.at("LiabilitiesCurrent")) / total_assets;
            
            // X2: Retained Earnings / Total Assets (Cumulative Profitability)
            double x2 = data.at("RetainedEarnings") / total_assets;
            
            // X3: EBIT / Total Assets (Operating Efficiency)
            double x3 = data.at("OperatingIncome") / total_assets;
            
            // X4: Market Value of Equity / Total Liabilities (Solvency/leverage)
            double x4 = data.at("MarketCap") / data.at("Liabilities");
            
            // X5: Sales / Total Assets (Asset Turnover)
            double x5 = data.at("Revenue") / total_assets;

            return (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5);
        } catch (...) {
            return -1.0; // Error indicator for missing tags
        }
    }

    /**
     * PFAI PREDICTIVE DRIVERS
     */
    std::map<std::string, double> calculate_pfai_drivers(double current_rev, double prev_rev, double net_income) {
        std::map<std::string, double> drivers;
        drivers["rev_velocity"] = (prev_rev != 0) ? (current_rev - prev_rev) / std::abs(prev_rev) : 0.0;
        drivers["op_margin"] = (current_rev > 0) ? net_income / current_rev : 0.0;
        return drivers;
    }

    double calculate_solvency_ratio(const std::map<std::string, double>& data) {
        try {
            double total_liabilities = data.at("Liabilities");
            if (total_liabilities <= 0) return 0.0;
            return data.at("CashFlowOps") / total_liabilities;
        } catch (...) {
            return 0.0;
        }
    }

    // Stripped out 'ticker' parameter since C++ doesn't need to look up sectors anymore
    std::vector<double> get_tensor(const std::map<std::string, double>& data) {
        std::vector<double> tensor;
        
        // --- GROUP A: THE KNOBS (Raw Dollars for Simulation) ---
        double rev = data.at("Revenue");
        tensor.push_back(rev);          // [1] Revenue
        tensor.push_back(data.at("COGS"));           // [2] Production Costs
        tensor.push_back(data.at("SGA"));            // [3] Salaries/Admin
        tensor.push_back(data.at("RD"));             // [4] Innovation
        tensor.push_back(data.at("CAPEX"));          // [5] Investment
        tensor.push_back(data.at("Inventory"));      // [6] Capacity
        
        // --- GROUP B: THE HEALTH (Ratios for Logic) ---
        double z = calculate_full_z_score(data);
        double solvency = calculate_solvency_ratio(data);
        double ebitda = data.at("OperatingIncome") + data.at("Depreciation");
        double ebitda_margin = (rev > 0) ? ebitda / rev : 0.0;
        double mcap = data.at("MarketCap");
        double ebitda_to_mcap = (mcap > 0) ? ebitda / mcap : 0.0;
        
        tensor.push_back(z);                         // [7] Risk Floor
        tensor.push_back(solvency);                  // [8] Cash Safety
        tensor.push_back(ebitda_margin);            // [9] Profitability
        tensor.push_back(ebitda_to_mcap);           // [10] Market Efficiency
        
        // --- GROUP C: THE MOMENTUM (PFAI Drivers) ---
        double prev_rev = data.at("PrevRevenue");
        double velocity = (prev_rev != 0) ? (data.at("Revenue") - prev_rev) / std::abs(prev_rev) : 0.0;
        double margin = (data.at("Revenue") > 0) ? data.at("NetIncome") / data.at("Revenue") : 0.0;
        tensor.push_back(velocity);                  // [11] Speed of growth
        tensor.push_back(margin);                    // [12] Profit efficiency

        // --- GROUP D: SCALE ---
        tensor.push_back(data.at("MarketCap"));      // [13] Relative size
        tensor.push_back(data.at("Assets"));         // [14] Total footprint

        // Returns only the 14 core features (Python handles appending the 9-dim sector vector)
        return tensor; 
    }
};

PYBIND11_MODULE(calculate_data, m) {
    py::class_<DataCalculator>(m, "DataCalculator")
        // Default init (no longer requires csv_path)
        .def(py::init<>())
        .def("calculate_z_score", &DataCalculator::calculate_full_z_score)
        .def("calculate_solvency_ratio", &DataCalculator::calculate_solvency_ratio)
        // get_tensor now only takes 'data', no more 'ticker' string
        .def("get_tensor", &DataCalculator::get_tensor, py::arg("data"));
}