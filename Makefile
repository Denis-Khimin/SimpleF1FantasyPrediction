# Paths to XGBoost (adjust these to where you installed XGBoost)
XGBOOST_PATH = /path/to/xgboost
XGBOOST_INCLUDE = $(XGBOOST_PATH)/include
XGBOOST_LIB = $(XGBOOST_PATH)/lib

# Compile and link with XGBoost
# List of source files
SOURCES = main.cpp other_source1.cpp other_source2.cpp

# Compile and link with XGBoost
main: $(SOURCES)
	g++ -Wall -std=c++17 -lxgboost $(SOURCES) -o main

# Run the program
run: main
	./main
