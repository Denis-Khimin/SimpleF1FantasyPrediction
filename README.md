# F1 Fantasy Points Prediction

This program predicts the F1 fantasy points for all drivers in the upcoming race using a combination of stochastic methods (moving averages and exponential decay) and machine learning (ML) techniques. Afterwards, an "optimal" fantasy team is chosen with maximal fantasy points under a specified budget.
**Note:** This program is intentionally kept very simple. As a result, the machine learning model is limited in its ability to make accurate, real-world predictions. Futhermore, it includes only the F1 races from the 2024 season until the 19th September. In order to get accurate results, one has to update the data files with the current points.

## Features

- **Data Input**: Utilizes provided CSV files containing information on drivers, teams, and previous race results.
- **Fantasy Points Prediction**: Predicts the number of F1 fantasy points each driver and team will earn in the upcoming race.
- **Machine Learning**: Uses the XGBoost algorithm to train a model on historical race data, enhancing prediction accuracy.
- **Stochastic Predictions**: Employs moving average and exponential decay methods to model driver performance and predict fantasy points.
- **Team Selection**: Finds the "optimal" team of drivers within a budget of $100, aiming to maximize total fantasy points.
- **Output**: Provides predicted fantasy points for each driver and the optimal team composition.

## Prediction Methodology

1. **Stochastic Modeling**:
    - **Moving Average**: Uses historical performance data to calculate a moving average for each driver, smoothing out fluctuations to identify consistent performance trends that contribute to fantasy points.
    - **Exponential Decay**: Applies an exponential decay factor to emphasize recent performance, reflecting its greater relevance to future fantasy points.

2. **Machine Learning (XGBoost)**:
    - Uses past race results (`driver_results_2024.csv`) to train an XGBoost model, which learns patterns between driver performance and fantasy points.
    - The simplified nature of the provided data restricts the model's ability to capture all factors affecting fantasy points in a real-world scenario.

3. **Fantasy Points Prediction**:
    - Combines the results of the stochastic modeling and machine learning to predict the number of fantasy points each driver will earn in the upcoming race.

4. **Optimal Team Selection**:
    - Finds the best combination of drivers under a budget of $100 to maximize total fantasy points using a simple brute-force method.

## Getting Started

### Prerequisites

- A C++ compiler that supports the C++17 standard or later.
- The [XGBoost library](https://github.com/dmlc/xgboost) installed and properly configured.

### Usage

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Denis-Khimin/SimpleF1FantasyPrediction.git
   ```
2. Install the XGBoost library. Follow the installation guide on the [XGBoost GitHub page](https://github.com/dmlc/xgboost).

3. Compile and run the program using a proper makefile (the one provided here might not necessarily work on your machine)

4. The program will:
   - Load driver and team data from the CSV files.
   - Train an XGBoost model using the historical race data.
   - Predict the fantasy points for each driver using stochastic methods and the machine learning model.
   - Find the "optimal" team of drivers within a $100 budget that maximizes total fantasy points.
   - Display the predicted fantasy points for each driver and the optimal team composition.

## Code Structure

- **`main.cpp`**: The main entry point of the program. It handles data loading, model training, stochastic predictions, fantasy points calculation, and team selection.
- **`driver.hh`**: Defines the `Driver` class and includes functions for managing driver-related data.
- **`team.hh`**: Defines the `Team` class and includes functions for managing team-related data.
- **`utility.hh`**: Utility functions for reading data files, preparing datasets, making the predictions and printing results.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the code or add new features.
