#include <iostream>
#include <cmath>
#include <time.h>
#include <omp.h>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <limits>

using namespace std;

// Algorithm parameters
constexpr int N_ANTS = 8;           // Number of ants
constexpr int N_ITER = 1000;       // Number of iterations
constexpr int ALPHA = 1;            // Pheromone importance
constexpr int BETA = 1;             // Distance importance
constexpr double EVAP_RATE = 0.5;   // Pheromone evaporation rate
constexpr double AMT_PHERO = 100.0; // Amount of pheromone deposited

// Euclidean distance function
double euclideanDistance(pair<int, int> a, pair<int, int> b) {
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

// Select next city based on probabilities computed from distances and pheromone values.
int selectNextCity(int current, vector<int>& visited, vector<double>& pheromone, const vector<pair<int, int>>& points) {
    int numPoints = points.size();
    vector<double> probabilities(numPoints, 0.0);
    double total = 0.0;

    // Parallel loop to calculate probabilities
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < numPoints; ++i) {
        if (!visited[i] && !(points[current].first == points[i].first && points[current].second == points[i].second)) {
            double dist = euclideanDistance(points[current], points[i]);
            double tau = pow(pheromone[current * numPoints + i], ALPHA);
            double eta = pow(1.0 / dist, BETA);
            probabilities[i] = tau * eta;
            total += probabilities[i];
        }
    }

    // Normalize probabilities
    #pragma omp parallel for
    for (int i = 0; i < numPoints; ++i) {
        probabilities[i] /= total;
    }

    double r = ((double)rand()) / RAND_MAX;
    double cumulative = 0.0;
    for (int i = 0; i < numPoints; i++) {
        if (!visited[i]) {
            cumulative += probabilities[i];
            if (r <= cumulative) {
                return i;
            }
        }
    }

    // Fallback: return the first unvisited city
    for (int i = 0; i < numPoints; i++) {
        if (!visited[i])
            return i;
    }
    return -1;
}

int main() {
    // Read coordinates from file
    vector<pair<int, int>> coordinates;
    ifstream infile("../Data/BerlinCities.txt");
    if (!infile) {
        cerr << "Error opening file 'Data/BerlinCities.txt'" << endl;
        return 1;
    }
    
    string line;
    while (getline(infile, line)) {
        // Skip the last two lines if they are "-1" or "EOF"
        if (line == "-1" || line == "EOF")
            continue;
        istringstream iss(line);
        int x, y;
        if (iss >> x >> y) {
            coordinates.push_back(make_pair(x, y));
        }
    }
    infile.close();
    
    int numPoints = coordinates.size();
    if (numPoints == 0) {
        cerr << "No coordinates loaded from file." << endl;
        return 1;
    }

    omp_set_num_threads(4); // Set number of threads (adjust as needed)

    auto start = chrono::high_resolution_clock::now();

    srand(42);
    // Initialize pheromone matrix; dimensions: numPoints x numPoints
    vector<double> pheromone(numPoints * numPoints, 1.0);
    vector<int> bestPath(numPoints);
    double bestLength = numeric_limits<double>::infinity();

    vector<vector<int>> paths(N_ANTS, vector<int>(numPoints));
    vector<double> lengths(N_ANTS);

    // Vector to store the minimum (best) path length for each iteration.
    vector<double> iterationMin(N_ITER, 0.0);

    for (int iter = 0; iter < N_ITER; ++iter) {
        // Parallel ant path construction
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < N_ANTS; ++k) {
            unsigned int seed = 42 + omp_get_thread_num();
            vector<int> visited(numPoints, 0);
            int current = rand_r(&seed) % numPoints;

            visited[current] = 1;
            paths[k][0] = current;
            double pathLength = 0.0;

            for (int step = 1; step < numPoints; ++step) {
                int nextCity = selectNextCity(current, visited, pheromone, coordinates);
                pathLength += euclideanDistance(coordinates[current], coordinates[nextCity]);
                current = nextCity;
                paths[k][step] = current;
                visited[current] = 1;
            }

            // Complete the cycle by returning to the starting city.
            pathLength += euclideanDistance(coordinates[current], coordinates[paths[k][0]]);
            lengths[k] = pathLength;

            // Update the global best if applicable.
            #pragma omp critical
            {
                if (pathLength < bestLength) {
                    bestLength = pathLength;
                    bestPath = paths[k];
                }
            }
        }
        
        // Find the minimum path length from the ants in this iteration.
        double iterationBest = numeric_limits<double>::infinity();
        for (int k = 0; k < N_ANTS; ++k) {
            if (lengths[k] < iterationBest)
                iterationBest = lengths[k];
        }
        iterationMin[iter] = iterationBest;
        
        // Parallel pheromone evaporation
        #pragma omp parallel for
        for (int i = 0; i < numPoints * numPoints; i++) {
            pheromone[i] *= EVAP_RATE;
        }

        // Sequential pheromone updates
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
    auto elapsed = end - start; // in nanoseconds

    cout << "Execution time: " << elapsed.count() / 1e9 << " seconds\n";
    cout << "Best path length: " << bestLength << "\nPath:\n";
    for (int i = 0; i < numPoints - 1; i++) {
        int index = bestPath[i];
        cout << "[" << coordinates[index].first << ", " << coordinates[index].second << "], ";
    }
    cout << "[" << coordinates[bestPath[numPoints - 1]].first << ", " 
         << coordinates[bestPath[numPoints - 1]].second << "].\n";

    // Write the minimum path length for each iteration to a file.
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
