#include "driver.hh"

#ifndef TEAM_H
#define TEAM_H

class Team {
public:
    Team(std::string given_team, double given_price, const Driver& driver_default)
        : team(given_team), price(given_price), total_points(0), expected_points(0), 
        driver_1(driver_default), driver_2(driver_default) {}

    void add_results(int points){results.push_back(points);}
    void add_points(int points) {total_points += points;}
    void set_drivers(Driver first, Driver second){driver_1 = first; driver_2 = second;}
    void set_expected_points(double points) { expected_points = std::round(points);}
    void compute_avg_points(){average_points 
                            = std::accumulate(results.begin(), results.end(), 0.0)/results.size();} 
    std::array<Driver, 2> get_drivers() const {
        return { driver_1, driver_2 };
    }

    double get_price() const {return price;}
    std::string get_team() const {return team;}
    double get_expected_points()const {return expected_points;}
    double get_average_points()const {return average_points;}
    std::vector<int> get_race_results() const {return results;}

    void print_member() const {driver_1.print();driver_2.print();}
    void print() const {
        std::cout << std::left << std::setw(40) << team
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
    std::string team;
    double price;
    int total_points;
    double average_points;
    double expected_points;
    std::vector<int> results;
    Driver driver_1;
    Driver driver_2;
};
#endif