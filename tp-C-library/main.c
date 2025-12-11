#include <math.h>
#include <stdio.h>

#include "exp_series.h"
#include "linear_regression.h"

static int test_main(void)
{
    const float features[] = {1.0f, 1.0f, 1.0f};
    const float thetas[] = {0.0f, 1.0f, 1.0f, 1.0f};
    const float expected = 3.0f;

    const float obtained = linear_regression_prediction(features, thetas, 4);
    if (obtained != expected) {
        printf("linear_regression_prediction() -> expected %.2f but got %.2f\n", expected, obtained);
        return 1;
    }

    printf("linear_regression_prediction() -> %.2f (expected %.2f)\n", obtained, expected);

    const float exp_input = 2.0f;
    const float exp_expected = 5.0f; // 1 + 2 + (2^2)/2 = 5
    const float exp_obtained = exp_approx(exp_input, 3);
    if (exp_obtained != exp_expected) {
        printf("exp_approx(%.2f) -> expected %.2f but got %.2f\n", exp_input, exp_expected, exp_obtained);
        return 1;
    }

    printf("exp_approx(%.2f) -> %.2f (expected %.2f)\n", exp_input, exp_obtained, exp_expected);

    const float sigmoid_input = 0.0f;
    const float sigmoid_expected = 0.5f;
    const float sigmoid_obtained = sigmoid(sigmoid_input);
    if (fabsf(sigmoid_obtained - sigmoid_expected) > 0.001f) {
        printf("sigmoid(%.2f) -> expected %.2f but got %.2f\n", sigmoid_input, sigmoid_expected, sigmoid_obtained);
        return 1;
    }

    printf("sigmoid(%.2f) -> %.2f (expected %.2f)\n", sigmoid_input, sigmoid_obtained, sigmoid_expected);

    const float logistic_features[] = {0.0f, 0.0f};
    const float logistic_thetas[] = {0.0f, 0.0f, 0.0f};
    const float logistic_expected = 0.5f;
    const float logistic_obtained = logistic_regression(logistic_features, logistic_thetas, 3);
    if (fabsf(logistic_obtained - logistic_expected) > 0.001f) {
        printf("logistic_regression() -> expected %.2f but got %.2f\n", logistic_expected, logistic_obtained);
        return 1;
    }

    printf("logistic_regression() -> %.2f (expected %.2f)\n", logistic_obtained, logistic_expected);
    return 0;
}

int main(void)
{
    return test_main();
}
