#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <numeric>

#ifndef DRIVER_H
#define DRIVER_H

class Driver {
public:
    Driver(std::string given_name, std::string given_team, double given_price)
        : name(given_name), team(given_team), price(given_price), 
        total_points(0), expected_points(0) {}

    void add_points(int points) {total_points += points;}
    void add_results(int points){results.push_back(points);}
    void set_expected_points(double points) {expected_points = points;}

    double get_price() const {return price; }
    int get_total_points() const {return total_points;}
    double get_expected_points() const {return expected_points;}
    double get_average_points() const {return average_points;}
    std::vector<int> get_race_results() const {return results;}
    std::string get_name() const {return name;}
    std::string get_team() const {return team;}

    void compute_avg_points(){average_points = 
                    std::accumulate(results.begin(), results.end(), 0.0)/results.size();} 

    
    void print() const {
        std::cout << std::left << std::setw(20) << name
                  << std::left << std::setw(20) << team
                  << std::internal << std::setw(15) << price
                  << std::internal << std::setw(15) << total_points
                  << std::internal << std::setw(15) <<std::setprecision(1)<<std::fixed<< average_points
                  << std::internal << std::setw(15) << expected_points
                  << std::endl;
    }
    void print_history() const {
        std::cout<<"[ ";
        for (const int& elem : results) {
            std::cout << elem << " ";
        }
        std::cout<< "]"<< std::endl;
    }
private:
    std::string name;
    std::string team;
    double price;
    int total_points;
    double average_points;
    double  expected_points;
    std::vector<int> results;
};
#endif