#include "exp_series.h"

float exp_approx(float x, int n_term)
{
    if (n_term <= 0) {
        return 0.0f;
    }

    float sum = 1.0f;
    float term = 1.0f;
    for (int i = 1; i < n_term; ++i) {
        term *= x / i;
        sum += term;
    }

    return sum;
}

float sigmoid(float x)
{
    const float exp_pos = exp_approx(x, 10);
    const float exp_neg = exp_approx(-x, 10);
    return exp_pos / (exp_pos + exp_neg);
}
