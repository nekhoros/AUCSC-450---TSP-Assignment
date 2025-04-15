#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <limits>
#include <random>

using namespace std;

// Algorithm parameters
constexpr int N_ANTS = 100; //Number of ants
constexpr int N_ITER = 1000; //Number of iterations in finding the best path
constexpr int ALPHA = 1; // the relative importance of the trail
constexpr int BETA = 1; // the relative importance of the visibility
constexpr double EVAP_RATE = 0.5; // 1-p, p = trail persistance
constexpr double AMT_PHERO = 100.0; // the quantity of trail laid by ants

/**
* euclideanDistance
* 
* finds the absolute distance between two given points through
* euclidean formula
* 
* @parameters
*   the two points
* 
* @returns
*   the distance between them
* 
*/
double euclideanDistance(pair<int, int> a, pair<int, int> b) {
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

/**
* selectNextCity
* 
* With some randomness included, depending on the amount pheremones, the next city will be 
* selected. The formula for the calculated probabilities was provided through the Marco Dorigo 
* article. The randomness included ensures that ants do not travel on a single path that could
* lead the ants down a certain, and not entirely certain best path.
* 
* @params
*   the current city, tabu, the pheromone 2d-array, all the points, and the random seed
* @return
*   the next city index
* 
*/
int selectNextCity(int current, vector<int>& visited, vector<double>& pheromone,
    const vector<pair<int, int>>& points, mt19937& rng) {
    int numPoints = points.size();
    vector<double> probabilities(numPoints, 0.0);
    double total = 0.0;

    for (int i = 0; i < numPoints; ++i) {
        if (!visited[i] && !(points[current].first == points[i].first && points[current].second == points[i].second)) {
            double dist = euclideanDistance(points[current], points[i]);
            double tau = pow(pheromone[current * numPoints + i], ALPHA);
            double eta = pow(1.0 / dist, BETA);
            probabilities[i] = tau * eta;
            total += probabilities[i];
        }
    }

    for (int i = 0; i < numPoints; ++i) {
        probabilities[i] /= total;
    }

    uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cumulative = 0.0;
    for (int i = 0; i < numPoints; i++) {
        if (!visited[i]) {
            cumulative += probabilities[i];
            if (r <= cumulative) {
                return i;
            }
        }
    }

    // Fallback
    for (int i = 0; i < numPoints; i++) {
        if (!visited[i])
            return i;
    }

    return -1;
}

int main() {
    // Load coordinates
    vector<pair<int, int>> coordinates;
    ifstream infile("../Data/BerlinCities.txt");
    if (!infile) {
        cerr << "Error opening file '../Data/BerlinCities.txt'" << endl;
        return 1;
    }

    // converts BerlinCities, and any of the other txt files into data points
    string line;
    while (getline(infile, line)) {
        if (line == "-1" || line == "EOF") continue;
        istringstream iss(line);
        int x, y;
        if (iss >> x >> y) {
            coordinates.emplace_back(x, y);
        }
    }
    infile.close();

    int numPoints = coordinates.size();
    if (numPoints == 0) {
        cerr << "No coordinates loaded from file." << endl;
        return 1;
    }

    //set the threads for experimentation
    omp_set_num_threads(8);
    //start the timer
    auto start = chrono::high_resolution_clock::now();

    //initialization of shared data
    vector<double> pheromone(numPoints * numPoints, 1.0);
    vector<int> bestPath(numPoints);
    double bestLength = numeric_limits<double>::infinity();

    vector<vector<int>> paths(N_ANTS, vector<int>(numPoints));
    vector<double> lengths(N_ANTS);
    vector<double> iterationMin(N_ITER, 0.0);

    for (int iter = 0; iter < N_ITER; ++iter) {
        //initialization of the threads
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < N_ANTS; ++k) {
            //set the seed for all the ants
            mt19937 rng(42 + omp_get_thread_num() + iter * N_ANTS);
            uniform_int_distribution<int> dist(0, numPoints - 1);

            vector<int> visited(numPoints, 0);
            int current = dist(rng);
            visited[current] = 1;
            paths[k][0] = current;
            double pathLength = 0.0;

            //travel through the graph
            for (int step = 1; step < numPoints; ++step) {
                int nextCity = selectNextCity(current, visited, pheromone, coordinates, rng);
                pathLength += euclideanDistance(coordinates[current], coordinates[nextCity]);
                current = nextCity;
                paths[k][step] = current;
                visited[current] = 1;
            }

            // Return to start
            pathLength += euclideanDistance(coordinates[current], coordinates[paths[k][0]]);
            lengths[k] = pathLength;

            #pragma omp critical
            {
                //check if we need to update the best values
                if (pathLength < bestLength) {
                    bestLength = pathLength;
                    bestPath = paths[k];
                }
            }
        }

        // Record iteration best
        double iterationBest = numeric_limits<double>::infinity();
        for (int k = 0; k < N_ANTS; ++k) {
            if (lengths[k] < iterationBest)
                iterationBest = lengths[k];
        }
        iterationMin[iter] = iterationBest;

        // Evaporate pheromone
        #pragma omp parallel for
        for (int i = 0; i < numPoints * numPoints; ++i) {
            pheromone[i] *= EVAP_RATE;
        }

        // Update pheromone
        for (int k = 0; k < N_ANTS; ++k) {
            for (int i = 0; i < numPoints - 1; ++i) {
                int from = paths[k][i];
                int to = paths[k][i + 1];
                pheromone[from * numPoints + to] += AMT_PHERO / lengths[k];
                pheromone[to * numPoints + from] += AMT_PHERO / lengths[k];
            }
            int last = paths[k][numPoints - 1];
            int first = paths[k][0];
            pheromone[last * numPoints + first] += AMT_PHERO / lengths[k];
            pheromone[first * numPoints + last] += AMT_PHERO / lengths[k];
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto elapsed = end - start;

    cout << "Execution time: " << chrono::duration<double>(elapsed).count() << " seconds\n";
    cout << "Best path length: " << bestLength << "\nPath:\n";
    for (int i = 0; i < numPoints - 1; i++) {
        int index = bestPath[i];
        cout << "[" << coordinates[index].first << ", " << coordinates[index].second << "], ";
    }
    cout << "[" << coordinates[bestPath[numPoints - 1]].first << ", "
        << coordinates[bestPath[numPoints - 1]].second << "].\n";

    //output of the paths
    ofstream outfile("iteration_lengths.txt");
    if (!outfile) {
        cerr << "Error opening file 'iteration_lengths.txt'" << endl;
        return 1;
    }
    for (double minLength : iterationMin) {
        outfile << minLength << "\n";
    }
    outfile.close();

    return 0;
}
