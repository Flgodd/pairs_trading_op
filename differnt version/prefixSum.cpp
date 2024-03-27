#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
//#include <immintrin.h>'
#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <iostream>
#include <array>
#include <experimental/simd>
//#include <experimental/execution_policy>
#include <chrono>
//#include <experimental/numeric>
#include <arm_neon.h>
#include <array>


using namespace std;

namespace simd = std::experimental;

std::vector<double> stock1_prices;
std::vector<double> stock2_prices;


vector<double> readCSV(const string& filename);



void read_prices() {

    string gs_file = "GS.csv";
    string ms_file = "MS.csv";

    stock1_prices = readCSV(gs_file);
    stock2_prices = readCSV(ms_file);

}


vector<double> readCSV(const string& filename){
    std::vector<double> prices;
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(value);
        }

        double adjClose = std::stod(row[5]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, 2512> spread;
    vector<int> check(4, 0);

    spread[0] = stock1_prices[0] - stock2_prices[0];
    spread[1] = (stock1_prices[0] - stock2_prices[0])*(stock1_prices[0] - stock2_prices[0]);

    for(size_t i = 1; i<1256; i++){
        const int idx = i*2;
        double current_spread = stock1_prices[i] - stock2_prices[i];
        spread[idx] = current_spread + spread[idx -2];
        spread[idx + 1] = (current_spread*current_spread) + spread[idx -1];

    }

    const int idx = (N-1)*2;
    double mean = (spread[idx])/ N;
    double stddev = std::sqrt((spread[idx +1])/ N - mean * mean);
    double current_spread = stock1_prices[N] - stock2_prices[N];
    double z_score = (current_spread - mean) / stddev;


    if (z_score > 1.0) {
        check[0]++;  // Long and Short
    } else if (z_score < -1.0) {
        check[1]++;  // Short and Long
    } else if (std::abs(z_score) < 0.8) {
        check[2]++;  // Close positions
    } else {
        check[3]++;  // No signal
    }


    for (size_t i = N+1; i < stock1_prices.size(); ++i) {
        const int idx = (i-1)*2;
        double mean = (spread[idx] - spread[idx-(N*2)])/ N;
        double stddev = std::sqrt((spread[idx +1] - spread[idx+1-(N*2)])/ N - mean * mean);
        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) / stddev;


        if (z_score > 1.0) {
            check[0]++;  // Long and Short
        } else if (z_score < -1.0) {
            check[1]++;  // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            check[2]++;  // Close positions
        } else {
            check[3]++;  // No signal
        }

    }
    cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<endl;

}


template<size_t N>
void BM_PairsTradingStrategyOptimized(benchmark::State& state) {
    if (stock1_prices.empty() || stock2_prices.empty()) {
        read_prices();
    }
    for (auto _ : state) {
        pairs_trading_strategy_optimized<N>(stock1_prices, stock2_prices);
    }
}

BENCHMARK_TEMPLATE(BM_PairsTradingStrategyOptimized, 8);

BENCHMARK_MAIN();
