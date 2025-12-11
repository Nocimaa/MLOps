#include <stdio.h>

float prediction(float *features, int n_feature)
{
    if (n_feature != 2) {
        return 0.0f;
    }

    if (features[0] > 0.000000000000f) {
        return 0;
    } else {
        if (features[1] > 0.000000000000f) {
            return 0;
        } else {
            return 1;
        }
    }

    return 0.0f;
}

int main(void)
{
    float sample[] = {160.000000000000f, 2.000000000000f};
    float prediction_value = prediction(sample, 2);
    printf("prediction = %f\n", prediction_value);
    return 0;
}
