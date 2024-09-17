# Paths to XGBoost (adjust these to where you installed XGBoost)
XGBOOST_PATH = /Users/macbookair2020/xgboost
XGBOOST_INCLUDE = $(XGBOOST_PATH)/include
XGBOOST_LIB = $(XGBOOST_PATH)/lib

# Compile and link with XGBoost
main: main.cpp
	g++ -Wall -std=c++17 -I$(XGBOOST_INCLUDE) -L$(XGBOOST_LIB) -lxgboost -Wl,-rpath,$(XGBOOST_LIB) main.cpp -o main

# Run the program
run: main
	./main
