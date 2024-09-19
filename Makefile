# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++17

# XGBoost library and include paths
XGBOOST_LIB = -lxgboost

# Source files and output executable
SOURCES = main.cpp
HEADERS = driver.hh team.hh utility.hh
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = main

# Default target
all: $(EXECUTABLE)

# Compile source files into object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link object files to create the executable
$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(XGBOOST_LIB) -o $(EXECUTABLE)

# Run the program
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Clean up
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
