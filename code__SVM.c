#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100  // S? lu?ng di?m d? li?u
#define DIM 2  // S? chi?u c?a d? li?u
#define MAX_ITER 1000
#define LEARNING_RATE 0.01

double weights[DIM];
double bias;

void initialize() {
    for (int i = 0; i < DIM; i++) {
        weights[i] = (double)rand() / RAND_MAX;
    }
    bias = (double)rand() / RAND_MAX;
}

double dot_product(double *v1, double *v2) {
    double result = 0.0;
    for (int i = 0; i < DIM; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

int predict(double *x) {
    return dot_product(weights, x) + bias > 0 ? 1 : -1;
}

void train(double X[N][DIM], int y[N]) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < N; i++) {
            if (y[i] * (dot_product(weights, X[i]) + bias) < 1) {
                for (int j = 0; j < DIM; j++) {
                    weights[j] += LEARNING_RATE * (y[i] * X[i][j] - 2 * 0.01 * weights[j]);
                }
                bias += LEARNING_RATE * y[i];
            } else {
                for (int j = 0; j < DIM; j++) {
                    weights[j] -= LEARNING_RATE * 2 * 0.01 * weights[j];
                }
            }
        }
    }
}

int main() {
    double X[N][DIM];
    int y[N];

    // Kh?i t?o d? li?u và nhãn
    for (int i = 0; i < N; i++) {
        X[i][0] = (double)rand() / RAND_MAX;
        X[i][1] = (double)rand() / RAND_MAX;
        y[i] = (X[i][0] + X[i][1] > 1) ? 1 : -1;
    }

    initialize();
    train(X, y);

    // Ki?m tra d? doán
    double test[DIM] = {0.6, 0.7};
    printf("Prediction for test data: %d\n", predict(test));

    return 0;
}