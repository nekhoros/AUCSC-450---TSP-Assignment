#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define N_POINTS 52

typedef struct {
    int x;
    int y;
} Point;

int n_ants = 10; //number of ants
int n_iterations = 100; //iterations through the cities
int alpha = 1; // the relative importance of the trail
int beta = 1; // the relative importance of the visibility
double evaporation_rate = 0.5; // 1-p, p = trail persistance
double Q = 100.0; // the quantity of trail laid by ants

Point BerlinCities[N_POINTS] = {
    {565, 575},{25, 185},{345, 750},{945, 685},{845, 655},{880, 660},{25, 230},
    {525, 1000},{580, 1175},{650, 1130},{1605, 620},{1220, 580},{1465, 200},{1530, 5},
    {845, 680},{725, 370},{145, 665},{415, 635},{510, 875},{560, 365},{300, 465},
    {520, 585},{480, 415},{835, 625},{975, 580},{1215, 245},{1320, 315},{1250, 400},
    {660, 180},{410, 250},{420, 555},{575, 665},{1150, 1160},{700, 580},{685, 595},
    {685, 610},{770, 610},{795, 645},{720, 635},{760, 650},{475, 960},{95, 260},
    {875, 920},{700, 500},{555, 815},{830, 485},{1170, 65},{830, 610},{605, 625},
    {595, 360},{1340, 725},{1740, 245}
};

double euclidean_distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int find_unvisited(int *visited) {
    for (int i = 0; i < N_POINTS; i++) {
        if (!visited[i]) return 1;
    }
    return 0;
}

int select_next_city(int current, int *visited, double *pheromone, Point *points) {
    double probabilities[N_POINTS] = {0.0};
    double total = 0.0;

    for (int i = 0; i < N_POINTS; i++) {
        if (!visited[i]) {
            double d = euclidean_distance(points[current], points[i]);
            if (d == 0) d = 0.0001; // Avoid division by zero
            double tau = pow(pheromone[current * N_POINTS + i], alpha);
            double eta = pow(1.0 / d, beta);
            probabilities[i] = tau * eta;
            total += probabilities[i];
        }
    }

    // Normalize
    for (int i = 0; i < N_POINTS; i++) {
        probabilities[i] /= total;
    }

    // Roulette wheel selection
    double r = ((double) rand()) / RAND_MAX;
    double cumulative = 0.0;
    for (int i = 0; i < N_POINTS; i++) {
        if (!visited[i]) {
            cumulative += probabilities[i];
            if (r <= cumulative) {
                return i;
            }
        }
    }

    // Fallback
    for (int i = 0; i < N_POINTS; i++) {
        if (!visited[i]) return i;
    }

    return -1;
}

int main() {
    srand(time(NULL));
    double *pheromone = (double*) malloc(N_POINTS * N_POINTS * sizeof(double));
    for (int i = 0; i < N_POINTS * N_POINTS; i++) pheromone[i] = 1.0;

    int *best_path = malloc(N_POINTS * sizeof(int));
    double best_length = INFINITY;

    for (int iter = 0; iter < n_iterations; iter++) {
        int **paths = malloc(n_ants * sizeof(int*));
        double *lengths = malloc(n_ants * sizeof(double));
        for (int k = 0; k < n_ants; k++) {
            paths[k] = malloc(N_POINTS * sizeof(int));
            int visited[N_POINTS] = {0};

            int current = rand() % N_POINTS;
            visited[current] = 1;
            paths[k][0] = current;
            double path_length = 0.0;

            for (int step = 1; step < N_POINTS; step++) {
                int next = select_next_city(current, visited, pheromone, BerlinCities);
                path_length += euclidean_distance(BerlinCities[current], BerlinCities[next]);
                current = next;
                paths[k][step] = current;
                visited[current] = 1;
            }

            path_length += euclidean_distance(BerlinCities[current], BerlinCities[paths[k][0]]);
            lengths[k] = path_length;

            if (path_length < best_length) {
                best_length = path_length;
                memcpy(best_path, paths[k], N_POINTS * sizeof(int));
            }
        }

        // Evaporate pheromone
        for (int i = 0; i < N_POINTS * N_POINTS; i++) {
            pheromone[i] *= evaporation_rate;
        }

        // Update pheromone
        for (int k = 0; k < n_ants; k++) {
            for (int i = 0; i < N_POINTS - 1; i++) {
                int from = paths[k][i];
                int to = paths[k][i + 1];
                pheromone[from * N_POINTS + to] += Q / lengths[k];
                pheromone[to * N_POINTS + from] += Q / lengths[k];
            }
            // Complete loop
            pheromone[paths[k][N_POINTS - 1] * N_POINTS + paths[k][0]] += Q / lengths[k];
            pheromone[paths[k][0] * N_POINTS + paths[k][N_POINTS - 1]] += Q / lengths[k];
        }

        // Free current iteration paths
        for (int k = 0; k < n_ants; k++) free(paths[k]);
        free(paths);
        free(lengths);
    }

    printf("Best path length: %.2f\nPath:\n", best_length);
    for (int i = 0; i < N_POINTS; i++) {
        int index = best_path[i];
        printf("[%d, %d], ",  BerlinCities[index].x, BerlinCities[index].y);
    }

    free(best_path);
    free(pheromone);
    return 0;
}
