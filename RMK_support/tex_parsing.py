from math import isclose
import numpy as np


def numToScientificTex(num: float, removeUnity=False):
    """Convert a float to scientific LaTeX notation.

    Args:
        num (float): Float to convert
        removeUnity (bool, optional): If true and the number is sufficiently close to 1 or -1 the 1 will be dropped. Defaults to False.
    """
    a, b = f"{num:.2e}".split("e")
    while a[-1] == "0":
        a = a[:-1]
    if a[-1] == ".":
        a = a[:-1]
    scientificConst = "".join([a, "\\cdot 10^{" + str(int(b)) + "}"])

    if removeUnity:
        if isclose(num, 1.0, rel_tol=1e-3):
            return ""
        if isclose(num, -1.0, rel_tol=1e-3):
            return "-"
    if isclose(num, round(num), rel_tol=1e-2) and num < 1000:
        return f"{round(num)}"
    a = f"{num:.2f}"
    while a[-1] == "0":
        a = a[:-1]
    if a[-1] == ".":
        a = a[:-1]
    return a if (np.abs(num) > 0.01 and np.abs(num) < 999) else scientificConst
