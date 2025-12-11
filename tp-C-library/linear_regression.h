#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

float linear_regression_prediction(const float *features,
                                   const float *thetas,
                                   int n_parameters);
float logistic_regression(const float *features,
                          const float *thetas,
                          int n_parameters);
int simple_tree(const float *features, int n_features);

#endif // LINEAR_REGRESSION_H
