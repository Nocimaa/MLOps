#include <stddef.h>

#include "exp_series.h"
#include "linear_regression.h"

float linear_regression_prediction(const float *features,
                                   const float *thetas,
                                   int n_parameters)
{
    if (n_parameters <= 0 || thetas == NULL || (n_parameters > 1 && features == NULL)) {
        return 0.0f;
    }

    float prediction = thetas[0];
    for (int i = 1; i < n_parameters; ++i) {
        prediction += thetas[i] * features[i - 1];
    }

    return prediction;
}

float logistic_regression(const float *features,
                          const float *thetas,
                          int n_parameters)
{
    const float linear_output = linear_regression_prediction(features, thetas, n_parameters);
    return sigmoid(linear_output);
}

int simple_tree(const float *features, int n_features)
{
    if (features == NULL || n_features < 2) {
        return -1;
    }

    if (features[0] > 0.0f) {
        return 0;
    }

    if (features[1] > 0.0f) {
        return 0;
    }

    return 1;
}
