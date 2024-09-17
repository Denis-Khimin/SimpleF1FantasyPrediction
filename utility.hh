#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <random>
#include <xgboost/c_api.h>

#include "driver.hh"
#include "team.hh"

// Read files
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
std::vector<Team> read_teams(const std::string &filename) {
    std::cout<<"Reading teams from "<<filename<<std::endl;

    std::vector<Team> teams;
    std::ifstream file(filename);
    std::string line;
    
    // Check if file can be opened
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return teams;
    }

    Driver default_driver("default_name","default_team",0.0);
    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string team, price_str;

        if (std::getline(iss, team, ',') &&
            std::getline(iss, price_str)) {

            double price = std::stod(price_str);
            teams.push_back(Team(team, price,default_driver));
        }
    }

    file.close();
    return teams;
}

//---------------------------------------
std::vector<Driver> read_drivers(const std::string &filename) {
    std::cout<<"Reading drivers from "<<filename<<std::endl;

    std::vector<Driver> drivers;
    std::ifstream file(filename);
    std::string line;
    
    // Check if file can be opened
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return drivers;
    }

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string name, team, price_str;

        if (std::getline(iss, name, ',') &&
            std::getline(iss, team, ',') &&
            std::getline(iss, price_str)) {

            double price = std::stod(price_str);
            drivers.push_back(Driver(name, team, price));
        }
    }

    file.close();
    return drivers;
}

//---------------------------------------
void read_race_results(const std::string &filename_drivers,
                       const std::string &filename_teams,
                       std::vector<Driver> &drivers,
                       std::vector<Team> &teams) {

    std::cout<<"Reading results from "<<filename_drivers<<", "<<filename_teams<<std::endl;

    std::ifstream file_d(filename_drivers);
    std::string line_d;
    std::ifstream file_t(filename_teams);
    std::string line_t;
    
    // Check if file can be opened
    if (!file_d.is_open() || !file_t.is_open()) {
        std::cerr << "Error opening file: " << filename_drivers << std::endl;
        return;
    }

    // Skip the header line
    std::getline(file_d, line_d);

    while (std::getline(file_d, line_d)) {
        std::istringstream iss(line_d);
        std::string pos_str, name, team;

        std::getline(iss, pos_str, ','); // Skip the position number
        std::getline(iss, name, ',');
        std::getline(iss, team, ',');

        std::vector<int> race_points;
        // Read race points
        for (int i = 0; i < 30; ++i) { // Assuming there max. 30 result columns
            std::string point;
            std::getline(iss >> std::ws, point, ',');
            if (point.empty()) {
                break; // No more results to read
            } 
            else 
            {
                int race_point = std::stoi(point);
                race_points.push_back(race_point);
            }
        }

        // Process the race results for the driver
        for (auto &driver : drivers) {
            if (driver.get_name() == name && driver.get_team() == team) {
                for (const auto &point : race_points) {
                    driver.add_results(point);
                    driver.add_points(point);
                }
            }
            driver.compute_avg_points();
        }
    }

    file_d.close();

    // Skip the header line
    std::getline(file_t, line_t);

    while (std::getline(file_t, line_t)) {
        std::istringstream iss(line_t);
        std::string pos_str, teamname;
        std::getline(iss >> std::ws, pos_str, ',');
        std::getline(iss >> std::ws, teamname, ',');
        std::vector<int> race_points;
        
        // Read race positions
        for (int i = 0; i < 30; ++i) { // Assuming there max. 30 result columns
            std::string point;
            std::getline(iss >> std::ws, point, ',');
            if (point.empty()) {
                break; // No more results to read
            } 
            else {
                try {
                    int race_point = std::stoi(point);
                    race_points.push_back(race_point);
                } catch (const std::invalid_argument&) {
                    std::cerr << "Invalid point value: " << point << " for team " << teamname << "\n";
                }
            }
        }

        // Process the race results for the driver
        for (auto &team : teams) {
            if (team.get_team() == teamname) {
                for (const auto &point : race_points) {
                    team.add_results(point);
                    team.add_points(point);
                }
            }
            team.compute_avg_points();
        }
    }
    file_t.close();
}

// Assign
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
void assign_drivers_to_teams(std::vector<Driver> drivers, std::vector<Team>& teams) {
    for (size_t i = 0; i < teams.size(); ++i) {
        std::string search = teams[i].get_team();
        Driver tmp_1("", "", 0.0);
        Driver tmp_2("", "", 0.0);
        int found = 0;
        for (size_t j = 0; j < drivers.size(); ++j) {
            if (drivers[j].get_team() == search) {
                if (found == 0) {
                    tmp_1 = drivers[j];
                    found++;
                } else if (found == 1) {
                    tmp_2 = drivers[j];
                    found++;
                    break;
                }
            }
        }
        if (found == 2) {
            teams[i].set_drivers(tmp_1, tmp_2);
            found = 0;
        } else {
            std::cerr << "Not enough drivers found for team: " << search << std::endl;
        }
    }
}

// Machine Learning prediction
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

//---------------------------------------
// Helper function to calculate variance
double calculate_variance(const std::vector<int>& results, double mean) {
    double variance = 0.0;
    for (const auto& points : results) {
        variance += std::pow(points - mean, 2);
    }
    return variance / results.size();
}

//---------------------------------------
std::vector<float> extract_features(const std::vector<int>& results) {
    std::vector<float> features;

    // Feature 1: Recent performance (average over last 3 races)
    int window_size = 3;
    double recent_performance = 0.0;
    int count = 0;
    for (int i = std::max(0, (int)results.size() - window_size); i < results.size(); ++i) {
        recent_performance += results[i];
        count++;
    }
    recent_performance = (count > 0) ? recent_performance / count : 0.0;
    features.push_back(recent_performance);

    // Feature 2: Consistency (standard deviation of race results)
    double mean = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
    double variance = calculate_variance(results, mean);
    double consistency = std::sqrt(variance);
    features.push_back(consistency);

    // Feature 3: Overall average performance
    double avg_performance = mean;
    features.push_back(avg_performance);

    // Feature 4: Recent Improvement or Decline
    // Calculate the difference between the most recent race result and the one before it
    if (results.size() >= 2) {
        int recent_improvement = results.back() - results[results.size() - 2];
        features.push_back(recent_improvement);  // Positive if performance improved, negative if declined
    } else {
        features.push_back(0);  // Default to 0 if there are fewer than 2 results
    }

    // Feature 5: Time-Weighted Average of Race Results
    // More recent race results have more weight
    double weighted_sum = 0.0;
    double total_weight = 0.0;
    for (int i = 0; i < results.size(); ++i) {
        double weight = 1.0 / (i + 1);  // Most recent race has the largest weight
        weighted_sum += results[results.size() - 1 - i] * weight;
        total_weight += weight;
    }
    double time_weighted_avg = (total_weight > 0) ? weighted_sum / total_weight : 0.0;
    features.push_back(time_weighted_avg);

    return features;
}

//---------------------------------------
// Create the dataset and prepare it for XGBoost training
void prepare_dataset(std::vector<Driver>& drivers, std::vector<Team>& teams,
                     std::vector<float>& feature_matrix, std::vector<float>& labels) {
    // Prepare data for drivers
    for (auto& driver : drivers) {
        std::vector<int> results = driver.get_race_results();
        for (size_t i = 0; i < results.size() - 1; ++i) {
            // Extract features from the first i race results
            std::vector<float> features = extract_features(std::vector<int>(results.begin(), results.begin() + i + 1));
            
            // Add features to the feature matrix
            feature_matrix.insert(feature_matrix.end(), features.begin(), features.end());
            
            // The target label is the next race result (i+1)
            labels.push_back(results[i + 1]);
        }
    }

    // Prepare data for teams
    for (auto& team : teams) {
        std::vector<int> results = team.get_race_results();
        for (size_t i = 0; i < results.size() - 1; ++i) {
            // Extract features from the first i race results
            std::vector<float> features = extract_features(std::vector<int>(results.begin(), results.begin() + i + 1));
            
            // Add features to the feature matrix
            feature_matrix.insert(feature_matrix.end(), features.begin(), features.end());
            
            // The target label is the next race result (i+1)
            labels.push_back(results[i + 1]);
        }
    }
}

//---------------------------------------
// Function to train the XGBoost model
void train_xgboost_model(const std::vector<float>& feature_matrix, const std::vector<float>& labels, 
                         const std::string& model_path) {
    // Feature matrix dimensions (rows and columns)
    size_t num_rows = labels.size();
    size_t num_columns = feature_matrix.size() / num_rows;

    // Create DMatrix (XGBoost format) for training
    DMatrixHandle dtrain;
    XGDMatrixCreateFromMat(feature_matrix.data(), num_rows, num_columns, 0.0, &dtrain);
    XGDMatrixSetFloatInfo(dtrain, "label", labels.data(), labels.size());

    // Set training parameters
    const std::vector<std::pair<std::string, std::string>> params = {
        {"objective", "reg:squarederror"},   // Regression objective
        {"max_depth", "1000"},               // Maximum tree depth
        {"eta", "0.05"},                     // Learning rate
        {"lambda", "0.5"},                   // L2 regularization
        {"alpha", "0.5"},                    // L1 regularization
        {"eval_metric", "mae"}               // Mean Absolute Error for evaluation
    };
    
    // Create the booster
    BoosterHandle booster;
    XGBoosterCreate(&dtrain, 1, &booster);

    // Set the parameters for the booster
    for (const auto& param : params) {
        XGBoosterSetParam(booster, param.first.c_str(), param.second.c_str());
    }

    // Train the model with 100 boosting rounds
    for (int iter = 0; iter < 100; ++iter) {
        XGBoosterUpdateOneIter(booster, iter, dtrain);
    }

    // Save the trained model to file
    XGBoosterSaveModel(booster, model_path.c_str());

    // Clean up resources
    XGBoosterFree(booster);
    XGDMatrixFree(dtrain);
}

//---------------------------------------
// Function to fine-tune the XGBoost model with recent race results
void fine_tune_xgboost_model(std::vector<Driver>& drivers, std::vector<Team>& teams, 
                             const std::string& model_path, int num_recent_races = 3) {
    // Create feature matrix and labels for the most recent races
    std::vector<float> recent_feature_matrix;
    std::vector<float> recent_labels;
    
    // Prepare dataset for recent races
    for (auto& driver : drivers) {
        std::vector<int> results = driver.get_race_results();
        size_t num_races = results.size();
        
        // Take the last num_recent_races results if there are enough races
        if (num_races > num_recent_races) {
            std::vector<float> features = extract_features(
                std::vector<int>(results.end() - num_recent_races, results.end() - 1)
            );
            recent_feature_matrix.insert(recent_feature_matrix.end(), features.begin(), features.end());
            recent_labels.push_back(results.back());  // The most recent race is the target
        }
    }
    
    // Repeat the same process for teams
    for (auto& team : teams) {
        std::vector<int> results = team.get_race_results();
        size_t num_races = results.size();
        
        if (num_races > num_recent_races) {
            std::vector<float> features = extract_features(
                std::vector<int>(results.end() - num_recent_races, results.end() - 1)
            );
            recent_feature_matrix.insert(recent_feature_matrix.end(), features.begin(), features.end());
            recent_labels.push_back(results.back());
        }
    }
    
    // Load the existing model
    BoosterHandle booster;
    XGBoosterCreate(nullptr, 0, &booster);
    XGBoosterLoadModel(booster, model_path.c_str());

    // Create DMatrix for recent data
    size_t num_rows = recent_labels.size();
    size_t num_columns = recent_feature_matrix.size() / num_rows;
    DMatrixHandle drecent;
    XGDMatrixCreateFromMat(recent_feature_matrix.data(), num_rows, num_columns, 0.0, &drecent);
    XGDMatrixSetFloatInfo(drecent, "label", recent_labels.data(), recent_labels.size());

    // Fine-tune the model with the recent race data (lower learning rate for fine-tuning)
    std::vector<std::pair<std::string, std::string>> params = {
        {"objective", "reg:squarederror"},   // Regression objective
        {"max_depth", "1000"},                 // Maximum tree depth
        {"eta", "0.02"},                     // Learning rate
        {"lambda", "0.1"},                   // L2 regularization
        {"alpha", "0.1"},                    // L1 regularization
        {"eval_metric", "mae"}               // Mean Absolute Error for evaluation
    };

    // Set the parameters for the booster
    for (const auto& param : params) {
        XGBoosterSetParam(booster, param.first.c_str(), param.second.c_str());
    }
    
    // Fine-tune the model with 50 boosting rounds on recent data
    for (int iter = 0; iter < 50; ++iter) {
        XGBoosterUpdateOneIter(booster, iter, drecent);
    }

    // Save the fine-tuned model
    XGBoosterSaveModel(booster, model_path.c_str());

    // Clean up
    XGBoosterFree(booster);
    XGDMatrixFree(drecent);
}

//---------------------------------------
void calculate_expected_performance_ML(std::vector<Driver>& drivers, std::vector<Team>& teams, const std::string& model_path) {
    // Load the pre-trained XGBoost model
    BoosterHandle booster;
    XGBoosterCreate(nullptr, 0, &booster);
    XGBoosterLoadModel(booster, model_path.c_str());

    // Process drivers
    for (auto& driver : drivers) {
        std::vector<int> race_results = driver.get_race_results();
        
        // Extract features from race results
        std::vector<float> features = extract_features(race_results);
        
        // Convert features into DMatrix format (XGBoost input format)
        DMatrixHandle dmatrix;
        XGDMatrixCreateFromMat(features.data(), 1, features.size(), 0.0, &dmatrix);

        // Perform prediction
        bst_ulong out_len;
        const float* out_result;
        XGBoosterPredict(booster, dmatrix, 0, 0, 0, &out_len, &out_result);
        
        // Set expected points for the driver
        driver.set_expected_points(out_result[0]);

        // Clean up
        XGDMatrixFree(dmatrix);
    }

    // Process teams similarly
    for (auto& team : teams) {
        std::vector<int> race_results = team.get_race_results();
        
        // Extract features from race results
        std::vector<float> features = extract_features(race_results);
        
        // Convert features into DMatrix format (XGBoost input format)
        DMatrixHandle dmatrix;
        XGDMatrixCreateFromMat(features.data(), 1, features.size(), 0.0, &dmatrix);

        // Perform prediction
        bst_ulong out_len;
        const float* out_result;
        XGBoosterPredict(booster, dmatrix, 0, 0, 0, &out_len, &out_result);

        // Set expected points for the team
        team.set_expected_points(out_result[0]);

        // Clean up
        XGDMatrixFree(dmatrix);
    }

    // Cleanup XGBoost booster after usage
    XGBoosterFree(booster);
}


// Stochastic prediction
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

double combine_predictions(double ai_prediction, double stochastic_prediction, double average_points) {
    double combi_prediction  = 0;

    // Due to the insufficient training data, the ML prediction 
    // is too far off from the stochastic prediction --> We choose the stochastic prediction
    if(std::abs(ai_prediction - stochastic_prediction) > 7.0)
    {
        combi_prediction =  stochastic_prediction;
    }
    // Otherwise we take the middle
    else 
       combi_prediction = 0.5 * stochastic_prediction + 0.5 * ai_prediction;

    // If the prediction is too far away from the average points we correct it
    if(std::abs(combi_prediction - average_points) > 5.0)
    {
        combi_prediction = 0.75*combi_prediction + 0.25*average_points;
    }
    if(combi_prediction <0)
        combi_prediction = average_points;

    return combi_prediction;
}

//---------------------------------------
void calculate_expected_performace_combi(std::vector<Driver>& drivers, std::vector<Team>& teams) {

    const double decay_factor = 0.5; // Exponential decay for weights
    const int window_size = 3; // Moving average window size

    // Process drivers
    for (size_t i = 0; i < drivers.size(); ++i) {
        std::vector<int> race_points = drivers[i].get_race_results();
        int n = race_points.size();

        // Exponential decay approach
        double weighted_sum = 0.0;
        double total_weight = 0.0;

        for (int j = 0; j < n; ++j) {
            double weight = std::pow(decay_factor, (n - j - 1));
            weighted_sum += race_points[j] * weight;
            total_weight += weight;
        }
        double exp_points = weighted_sum / total_weight; 

        // Moving average approach
        double sum = 0.0;
        int count = 0;
        for (int j = std::max(0, n - window_size); j < n; ++j) {
            sum += race_points[j];
            count++;
        }
        double mov_avg_points = (count > 0) ? sum / count : 0;

        // Combine both methods
        double expected_points = (exp_points + mov_avg_points) / 2;

        // Combine AI and stochastic
        double expected_points_AI = drivers[i].get_expected_points();
        double expected_points_combi = combine_predictions(expected_points_AI, expected_points, drivers[i].get_average_points());

        drivers[i].set_expected_points(expected_points_combi);
    }

    // Process teams
    for (size_t i = 0; i < teams.size(); ++i) {
        std::vector<int> race_points = teams[i].get_race_results();
        int n = race_points.size();

        // Exponential decay approach
        double weighted_sum = 0.0;
        double total_weight = 0.0;

        for (int j = 0; j < n; ++j) {
            double weight = std::pow(decay_factor, (n - j - 1));
            weighted_sum += race_points[j] * weight;
            total_weight += weight;
        }
        double exp_points = weighted_sum / total_weight; 

        // Moving average approach
        double sum = 0.0;
        int count = 0;
        for (int j = std::max(0, n - window_size); j < n; ++j) {
            sum += race_points[j];
            count++;
        }
        double mov_avg_points = (count > 0) ? sum / count : 0;

        // Combine both methods
        double expected_points = (exp_points + mov_avg_points) / 2;

        // Combine AI and stochastic
        double expected_points_AI = drivers[i].get_expected_points();
        double expected_points_combi = combine_predictions(expected_points_AI,expected_points, teams[i].get_average_points());

        teams[i].set_expected_points(expected_points_combi);
    }
}


// Optimality (Here brute force)
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
void find_optimal_team(const std::vector<Driver> &drivers, 
                       const std::vector<Team> &teams, 
                       const std::string& model_path) {
    double budget = 100.0;
    int best_points = 0;
    std::vector<Driver> best_team_drivers;
    std::vector<Team> best_team_teams;
    size_t num_drivers = drivers.size();
    size_t num_teams = teams.size();

    // Iterate through all possible teams of 5 drivers and 2 teams
    for (size_t i = 0; i < num_drivers - 4; ++i) {
        for (size_t j = i + 1; j < num_drivers - 3; ++j) {
            for (size_t k = j + 1; k < num_drivers - 2; ++k) {
                for (size_t l = k + 1; l < num_drivers - 1; ++l) {
                    for (size_t m = l + 1; m < num_drivers; ++m) {
                        for (size_t n = 0; n < num_teams - 1; ++n) {
                            for (size_t p = n + 1; p < num_teams; ++p) {
                                std::vector<Driver> team_drivers = { drivers[i], drivers[j], drivers[k], drivers[l], drivers[m] };
                                std::vector<Team> team_teams = { teams[n], teams[p] };

                                double total_driver_price = 0;
                                int expected_driver_points = 0;
                                int expected_team_points = 0;
                                double total_team_price = 0;

                                // Calculate total price and points for teams
                                for (const auto &driver : team_drivers) {
                                    total_driver_price += driver.get_price();
                                    expected_driver_points += driver.get_expected_points();
                                }

                                for (const auto &team : team_teams) {
                                    total_team_price += team.get_price();
                                    expected_team_points += team.get_expected_points();
                                }

                                // Check each driver for doubling their points (turbo driver)
                                for (size_t d = 0; d < team_drivers.size(); ++d) {
                                    int driver_points_with_bonus = expected_driver_points;
                                    driver_points_with_bonus += team_drivers[d].get_expected_points(); // Add the driver's points once more for doubling

                                    double total_price = total_driver_price + total_team_price;

                                    // Check if within budget and if this configuration has more points
                                    if (total_price <= budget && (driver_points_with_bonus + expected_team_points) > best_points) {
                                        best_points = driver_points_with_bonus + expected_team_points;
                                        best_team_drivers = team_drivers;
                                        best_team_teams = team_teams;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (best_team_drivers.empty() || best_team_teams.empty()) {
        std::cout << "No team found within the budget.\n";
    } else {
        for(int i = 0; i < 105; ++i){
            std::cout << "*";
        }
        std::cout<<std::endl;
        for(int i = 0; i < 43; ++i){
            std::cout << "*";
        }
        std::cout<<" Found optimal team ";
        for(int i = 0; i < 42; ++i){
            std::cout << "*";
        }
        std::cout<<std::endl;
        for(int i = 0; i < 105; ++i){
            std::cout << "*";
        }
        std::cout<<"\n"<<std::endl;
        std::cout << std::left << std::setw(35) << "Team"
              << std::internal << std::setw(20) << "Price"
              << std::internal << std::setw(18) << "Total pt."
              << std::internal << std::setw(16) << "Average pt."
              << std::internal << std::setw(16) << "Expected pt."
              << std::endl;
        for(int i = 0; i < 105; ++i){
            std::cout << "-";
        }
        std::cout<<std::endl;

        double last_points = 0.0;
        double total_cost = 0.0;
        double double_driver = 0.0;
        for (const auto &driver : best_team_drivers) {
            driver.print();
            if(driver.get_race_results().back() > double_driver)
            {
                double_driver = driver.get_race_results().back();
            }
            total_cost += driver.get_price();
            last_points+= driver.get_race_results().back();
        }
        for (const auto &team : best_team_teams) {
            team.print();
            total_cost += team.get_price();
            last_points+= team.get_race_results().back();
        }
        for(int i = 0; i < 105; ++i){
            std::cout << "-";
        }
        std::cout<<std::endl;


        std::cout<<"Last pt. "<<last_points + double_driver << std::endl;
        std::cout<<"Expected pt. "<<best_points << std::endl;
        std::cout<<"Total price "<<total_cost << std::endl;
    }
}

// Output
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
void print_drivers(const std::vector<Driver> &drivers) {
    std::cout << std::left << std::setw(20) << "Name"
              << std::left << std::setw(15) << "Team"
              << std::internal << std::setw(20) << "Price"
              << std::internal << std::setw(18) << "Total pt."
              << std::internal << std::setw(16) << "Average pt."
              << std::internal << std::setw(16) << "Expected pt."
              << std::endl;
    for (int i = 0;i<105;++i)
    {
        std::cout<<"-";
    }
    std::cout<<std::endl;
    for (const auto &driver : drivers) {
        driver.print();
        //driver.print_history();
    }
    std::cout<<std::endl;
}

//---------------------------------------
void print_teams(const std::vector<Team> &teams) {
    std::cout << std::left << std::setw(35) << "Team"
              << std::internal << std::setw(20) << "Price"
              << std::internal << std::setw(18) << "Total pt."
              << std::internal << std::setw(16) << "Average pt."
              << std::internal << std::setw(16) << "Expected pt."
              << std::endl;
    for (int i = 0;i<105;++i)
    {
        std::cout<<"-";
    }
    std::cout<<std::endl;
    for (const auto &team : teams) {
        team.print();
    }
    std::cout<<std::endl;
}
