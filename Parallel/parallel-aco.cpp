#include <iostream>
#include <cmath>
#include <time.h>
#include <omp.h>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

constexpr int N_ANTS = 30;           // Number of ants
constexpr int N_ITER = 10000;       // Number of iterations
constexpr int ALPHA = 1;            // Pheromone importance
constexpr int BETA = 1;             // Distance importance
constexpr double EVAP_RATE = 0.5;   // Pheromone evaporation rate
constexpr double AMT_PHERO = 100.0; // Amount of pheromone deposited

vector<pair<int, int>> coordinates = {
    {565, 575}, {25, 185}, {345, 750}, {945, 685}, {845, 655}, {880, 660}, {25, 230}, {525, 1000}, 
    {580, 1175}, {650, 1130}, {1605, 620}, {1220, 580}, {1465, 200}, {1530, 5}, {845, 680}, {725, 370}, 
    {145, 665}, {415, 635}, {510, 875}, {560, 365}, {300, 465}, {520, 585}, {480, 415}, {835, 625}, 
    {975, 580}, {1215, 245}, {1320, 315}, {1250, 400}, {660, 180}, {410, 250}, {420, 555}, {575, 665}, 
    {1150, 1160}, {700, 580}, {685, 595}, {685, 610}, {770, 610}, {795, 645}, {720, 635}, {760, 650}, 
    {475, 960}, {95, 260}, {875, 920}, {700, 500}, {555, 815}, {830, 485}, {1170, 65}, {830, 610},  
    {605, 625}, {595, 360}, {1340, 725}, {1740, 245}
};

const int numPoints = coordinates.size();

double euclideanDistance(pair<int, int> a, pair<int, int> b) {
    return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

int selectNextCity(int current, vector<int>& visited, vector<double>& pheromone, const vector<pair<int, int>>& points) {
    vector<double> probabilities(numPoints, 0.0);
    double total = 0.0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Task parallelism for calculating probabilities
            for (int i = 0; i < numPoints; ++i) {
                #pragma omp task shared(probabilities, total)
                {
                    if (!visited[i] && (points[current].first != points[i].first || points[current].second != points[i].second)) {
                        double euclidDistance = euclideanDistance(points[current], points[i]);
                        double tau = pow(pheromone[current * numPoints + i], ALPHA);
                        double eta = pow(1.0 / euclidDistance, BETA);
                        probabilities[i] = tau * eta;
                        #pragma omp atomic
                        total += probabilities[i];
                    }
                }
            }
        }

        #pragma omp barrier

        #pragma omp single
        {
            // Normalize probabilities
            for (int i = 0; i < numPoints; ++i) {
                #pragma omp task shared(probabilities, total)
                probabilities[i] /= total;
            }
        }
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

    for (int i = 0; i < numPoints; i++) {
        if (!visited[i])
            return i;
    }

    return -1;
}

int main() {
    int numThreads = 8;
    omp_set_num_threads(numThreads);

    auto start = std::chrono::high_resolution_clock::now();

    srand(42);
    vector<double> pheromone(numPoints * numPoints, 1.0);
    vector<int> bestPath(numPoints);
    double bestLength = INFINITY;

    vector<vector<int>> paths(N_ANTS, vector<int>(numPoints));
    vector<double> lengths(N_ANTS);

    #pragma omp parallel
    {
        for (int iter = 0; iter < N_ITER; ++iter) {
            #pragma omp single
            {
                // Task parallelism for ant path construction
                for (int k = 0; k < N_ANTS; ++k) {
                    #pragma omp task shared(paths, lengths, bestLength, bestPath)
                    {
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

                        pathLength += euclideanDistance(coordinates[current], coordinates[paths[k][0]]);
                        lengths[k] = pathLength;

                        #pragma omp critical
                        {
                            if (pathLength < bestLength) {
                                bestLength = pathLength;
                                bestPath = paths[k];
                            }
                        }
                    }
                }
            }

            #pragma omp barrier

            #pragma omp single
            {
                // Task parallelism for pheromone evaporation
                for (int i = 0; i < numPoints * numPoints; i++) {
                    #pragma omp task shared(pheromone)
                    pheromone[i] *= EVAP_RATE;
                }
            }

            #pragma omp barrier

            // Sequential pheromone updates
            #pragma omp single
            {
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
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    cout << "Execution time: " << elapsed.count() << " seconds\n";

    // Output results
    cout << "Best path length: " << bestLength << "\nPath:\n";
    for (int i = 0; i < numPoints - 1; i++) {
        int index = bestPath[i];
        cout << "[" << coordinates[index].first << ", " << coordinates[index].second << "], ";
    }
    
    cout << "[" << coordinates[bestPath[numPoints - 1]].first << ", "
         << coordinates[bestPath[numPoints - 1]].second << "].\n";

    return 0;
}