import pandas as pd
import yaml
import sys
import os

params = yaml.safe_load(open('params.yaml'))['preprocess']


def preprocess(input_path, output_path):
    data = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = data.dropna()
    data = data.drop_duplicates()

    data.to_csv(output_path, header=None, index=False)
    print(f"Preprocessing done into {output_path}")


if __name__=="__main__":
    preprocess(params["input"], params["output"])