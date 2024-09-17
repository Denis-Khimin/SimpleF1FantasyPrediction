#include "driver.hh"
#include "team.hh"
#include "utility.hh"

int main() {   
    // Create vectors with all drivers and all teams
    std::vector<Driver> drivers = read_drivers("data/drivers.csv");
    std::vector<Team> teams = read_teams("data/teams.csv");

    // Read race results and compute the expected performace
    read_race_results("data/driver_results_2024.csv","data/team_results_2024.csv",drivers,teams);
    assign_drivers_to_teams(drivers,teams);
    
    // Feature matrix and labels (for XGBoost training)
    std::vector<float> feature_matrix;
    std::vector<float> labels;

    // Prepare the dataset for training
    prepare_dataset(drivers, teams, feature_matrix, labels);

    // Train the XGBoost model and save it to a file
    std::string model_path = "xgboost_model.bin";
    train_xgboost_model(feature_matrix, labels, model_path);
    fine_tune_xgboost_model(drivers, teams, model_path, 3);

    std::cout << "Model trained and saved to " << model_path << std::endl;

    // Compute the expected performance utilizing the ML model and simple stochastics
    calculate_expected_performance_ML(drivers,teams, model_path); 
    calculate_expected_performace_combi(drivers,teams);

    // Print out the drivers and teams for verification
    print_drivers(drivers);
    print_teams(teams);

    // Find the optimal team based on points and budget
    std::cout<<"Searching for optimal team\n"<<std::endl;
    find_optimal_team(drivers, teams, model_path);

    return 0;
}
