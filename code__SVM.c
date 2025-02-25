#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 1000
#define LEARNING_RATE 0.01

typedef struct {
    double *weights;
    double bias;
} SVM;

double dot_product(double *a, double *b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

void train(SVM *svm, double **X, double *y, int n_samples, int n_features) {
    svm->weights = (double *)malloc(n_features * sizeof(double));
    for (int i = 0; i < n_features; i++) {
        svm->weights[i] = 0.0;
    }
    svm->bias = 0.0;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < n_samples; i++) {
            double condition = y[i] * (dot_product(svm->weights, X[i], n_features) + svm->bias);
            if (condition < 1) {
                for (int j = 0; j < n_features; j++) {
                    svm->weights[j] += LEARNING_RATE * (y[i] * X[i][j] - 2 * (1.0 / MAX_ITER) * svm->weights[j]);
                }
                svm->bias += LEARNING_RATE * y[i];
            } else {
                for (int j = 0; j < n_features; j++) {
                    svm->weights[j] += LEARNING_RATE * (-2 * (1.0 / MAX_ITER) * svm->weights[j]);
                }
            }
        }
    }
}

double predict(SVM *svm, double *x, int n_features) {
    double result = dot_product(svm->weights, x, n_features) + svm->bias;
    return result;
}

int main() {
    // Example usage
    int n_samples = 4;
    int n_features = 2;
    double *X[] = {
        (double[]){2, 3},
        (double[]){1, 1},
        (double[]){2, 1},
        (double[]){3, 2}
    };
    double y[] = {1, -1, -1, 1};

    SVM svm;
    train(&svm, X, y, n_samples, n_features);

    double test_sample[] = {2, 2};
    double prediction = predict(&svm, test_sample, n_features);
    printf("Prediction: %f\n", prediction);

    free(svm.weights);
    return 0;
}