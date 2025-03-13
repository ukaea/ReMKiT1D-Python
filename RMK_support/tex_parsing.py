from math import isclose
import numpy as np


def numToScientificTex(num: float, removeUnity=False, decimals: int = 2):
    """Convert a float to scientific LaTeX notation.

    Args:
        num (float): Float to convert
        removeUnity (bool, optional): If true and the number is sufficiently close to 1 or -1 the 1 will be dropped. Defaults to False.
        decimals (int, optional): Tolerance for truncation of decimal points. Defaults to 2 decimal places.
    """
    fmt = "{:." + str(decimals) + "e}"
    a, b = fmt.format(num).split("e")
    while a[-1] == "0":
        a = a[:-1]
    if a[-1] == ".":
        a = a[:-1]
    scientificConst = "".join([a, "\\cdot 10^{" + str(int(b)) + "}"])

    if removeUnity:
        if isclose(num, 1.0, rel_tol=10 ** (-decimals - 1)):
            return ""
        if isclose(num, -1.0, rel_tol=10 ** (-decimals - 1)):
            return "-"
    if isclose(num, round(num), rel_tol=10 ** (-decimals)) and num < 1000:
        return f"{round(num)}"
    fmt = "{:." + str(decimals) + "f}"
    a = fmt.format(num)
    while a[-1] == "0":
        a = a[:-1]
    if a[-1] == ".":
        a = a[:-1]
    return (
        a
        if (np.abs(num) > 10 ** (-decimals) and np.abs(num) < 999)
        else scientificConst
    )
