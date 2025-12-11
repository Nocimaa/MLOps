import argparse
import math
import pickle
import subprocess
from pathlib import Path

MODEL_FILE = Path(__file__).with_name("linear_model.joblib")
OUTPUT_C = Path(__file__).with_name("generated_model.c")
EXECUTABLE = Path(__file__).with_name("generated_model")
DEFAULT_SAMPLE = "160,2,1"


def parse_args():
    parser = argparse.ArgumentParser(description="Transpile a serialized regression/tree model to C")
    parser.add_argument("--model-path", type=Path, default=MODEL_FILE,
                        help="Path to the serialized model file")
    parser.add_argument("--model-type", choices=("linear", "logistic", "tree"), default="linear",
                        help="Type of model stored in the file")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE,
                        help="Comma-separated feature sample to hard-code into the generated driver")
    parser.add_argument("--output-c", type=Path, default=OUTPUT_C,
                        help="Output C source file")
    parser.add_argument("--executable", type=Path, default=EXECUTABLE,
                        help="Generated executable path")
    return parser.parse_args()


def load_model(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def extract_parameters(model):
    if isinstance(model, dict):
        return model["coef_"], float(model["intercept_"])
    if hasattr(model, "coef_") and hasattr(model, "intercept_"):
        coefs = getattr(model, "coef_")
        if hasattr(coefs, "ravel"):
            coefs = coefs.ravel()
        intercept = getattr(model, "intercept_")
        if hasattr(intercept, "item"):
            intercept = intercept.item()
        return list(map(float, coefs)), float(intercept)
    raise TypeError("Cannot extract parameters from provided model")


def format_vector(values):
    return ", ".join(f"{value:.12f}f" for value in values)


def parse_sample(sample_arg):
    values = [float(v.strip()) for v in sample_arg.split(",") if v.strip()]
    return values


def generate_linear_code(coefs, intercept, sample):
    n_features = len(coefs)
    coef_literal = format_vector(coefs)
    sample_literal = format_vector(sample)
    intercept_literal = f"{intercept:.12f}f"
    return f"""#include <stdio.h>

static const float COEFFICIENTS[{n_features}] = {{{coef_literal}}};

float prediction(float *features, int n_feature)
{{
    if (n_feature != {n_features}) {{
        return 0.0f;
    }}

    float result = {intercept_literal};
    for (int i = 0; i < n_feature; ++i) {{
        result += features[i] * COEFFICIENTS[i];
    }}
    return result;
}}

int main(void)
{{
    float sample[] = {{{sample_literal}}};
    float prediction_value = prediction(sample, {n_features});
    printf("prediction = %f\\n", prediction_value);
    return 0;
}}
"""


def format_tree_value(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.12f}f"


def render_tree_node(node, indent=4):
    indent_str = " " * indent
    if "value" in node:
        return indent_str + f"return {format_tree_value(node['value'])};\n"

    feature = node["feature"]
    threshold = node["threshold"]
    code = indent_str + f"if (features[{feature}] > {threshold:.12f}f) {{\n"
    code += render_tree_node(node["gt"], indent + 4)
    code += indent_str + "} else {\n"
    code += render_tree_node(node["lt"], indent + 4)
    code += indent_str + "}\n"
    return code


def generate_logistic_code(coefs, intercept, sample):
    n_features = len(coefs)
    coef_literal = format_vector(coefs)
    sample_literal = format_vector(sample)
    intercept_literal = f"{intercept:.12f}f"
    return f"""#include <stdio.h>

static const float COEFFICIENTS[{n_features}] = {{{coef_literal}}};

static float exp_approx(float x)
{{
    float sum = 1.0f;
    float term = 1.0f;
    for (int i = 1; i < 10; ++i) {{
        term *= x / i;
        sum += term;
    }}
    return sum;
}}

static float sigmoid(float x)
{{
    if (x > 20.0f) {{
        return 1.0f;
    }}
    if (x < -20.0f) {{
        return 0.0f;
    }}

    float pos = exp_approx(x);
    float neg = exp_approx(-x);
    return pos / (pos + neg);
}}

float prediction(float *features, int n_feature)
{{
    if (n_feature != {n_features}) {{
        return 0.0f;
    }}

    float linear = {intercept_literal};
    for (int i = 0; i < n_feature; ++i) {{
        linear += features[i] * COEFFICIENTS[i];
    }}
    return sigmoid(linear);
}}

int main(void)
{{
    float sample[] = {{{sample_literal}}};
    float prediction_value = prediction(sample, {n_features});
    printf("prediction = %f\\n", prediction_value);
    return 0;
}}
"""


def generate_tree_code(tree, sample):
    sample_literal = format_vector(sample)
    code = f"""#include <stdio.h>

float prediction(float *features, int n_feature)
{{
    if (n_feature != {len(sample)}) {{
        return 0.0f;
    }}

{render_tree_node(tree, 4)}
    return 0.0f;
}}

int main(void)
{{
    float sample[] = {{{sample_literal}}};
    float prediction_value = prediction(sample, {len(sample)});
    printf("prediction = %f\\n", prediction_value);
    return 0;
}}
"""
    return code


def python_linear_prediction(coefs, intercept, sample):
    return intercept + sum(c * x for c, x in zip(coefs, sample))


def python_logistic_prediction(coefs, intercept, sample):
    linear = python_linear_prediction(coefs, intercept, sample)
    return 1.0 / (1.0 + math.exp(-linear))


def python_tree_prediction(tree, sample):
    node = tree
    while True:
        if "value" in node:
            return node["value"]

        feature = node["feature"]
        threshold = node["threshold"]
        if sample[feature] > threshold:
            node = node["gt"]
        else:
            node = node["lt"]


def compile_c(source_path: Path, output_path: Path):
    command = ["gcc", str(source_path), "-o", str(output_path), "-Wall", "-Wextra"]
    print("Compilation command:", " ".join(command))
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    model = load_model(args.model_path)
    sample = parse_sample(args.sample)

    if args.model_type in ("linear", "logistic"):
        coefs, intercept = extract_parameters(model)
        if len(sample) != len(coefs):
            raise ValueError("Sample length must match coefficient count")
    else:
        tree = model.get("tree")
        if tree is None:
            raise ValueError("Tree model must expose 'tree' key.")
        if "n_features" in model and len(sample) != model["n_features"]:
            raise ValueError("Sample length must match tree feature count")

    if args.model_type == "linear":
        python_prediction = python_linear_prediction(coefs, intercept, sample)
        code = generate_linear_code(coefs, intercept, sample)
    elif args.model_type == "logistic":
        python_prediction = python_logistic_prediction(coefs, intercept, sample)
        code = generate_logistic_code(coefs, intercept, sample)
    else:
        python_prediction = python_tree_prediction(tree, sample)
        code = generate_tree_code(tree, sample)

    print("Python prediction:", python_prediction)
    args.output_c.write_text(code)
    print(f"Generated C driver at {args.output_c}")

    compile_c(args.output_c, args.executable)
    print("Running generated binary...")
    subprocess.run([str(args.executable)], check=True)


if __name__ == "__main__":
    main()
