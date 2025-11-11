import numpy as np
import pandas as pd

coords = pd.read_csv("data/coords.csv").set_index("index")
coord_mapping = coords.to_dict(orient="index")

width, height = coords["x"].max() + 1, coords["y"].max() + 1

train = pd.read_csv("data/sat.train.csv", sep=";", header=None)


def arr2img(arr: np.ndarray) -> np.ndarray:
    assert len(arr.shape) <= 2
    assert len(arr) == len(train), "This function is only for train data"
    arr_index = np.arange(len(train))
    if len(arr.shape) == 1:
        channels = 1
    else:
        channels = arr.shape[1]
    coords_arr = [coord_mapping.get(i, None) for i in arr_index]
    canvas = np.zeros((height, width, channels))
    for i, coord in zip(arr_index, coords_arr):
        if coord is not None:
            canvas[coord["y"], coord["x"]] = arr[i]
    return canvas
