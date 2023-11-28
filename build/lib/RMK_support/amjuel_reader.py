import numpy as np
import re
from typing import List


def loadAMJUELReactionData(
    reaction: str, section: str, filename="../data/amjuel.tex"
) -> List[str]:
    """Loads and performs minor cleanup of individual AMJUEL reaction data

    Args:
        reaction (str): Reaction number in AMJUEL
        filename (str, optional): Name of AMJUEL tex file. Defaults to "amjuel.tex".
        instance (int, optional): Which instance of the reaction data to use (AMJUEL has non-unique data labelling). Defaults to 0, using the first instance.

    Returns:
        List[str]: Split lines corresponding to given reaction
    """

    amjuel = open(filename)
    sectionRegEx = r"\\section{" + section + r" (.*?)\\section"
    sectionsFound = re.findall(sectionRegEx, amjuel.read(), re.S)
    assert len(sectionsFound) > 0, "Unable to find section " + section

    reactionRegex = r"Reaction " + reaction + r" (.*?)\\end{small}"
    A = re.findall(reactionRegex, sectionsFound[0], re.S)

    assert len(A) > 0, "Unable to find reaction " + reaction

    B = re.findall(r"\\begin{verbatim}(.*?)\\end{verbatim}", str(A[0]), re.S)
    lines = re.findall(r"(.*?)\\n", str(B), re.S)
    lines = [i for i in lines if i != "\\" and i != ""]
    lines = [line.split() for line in lines]

    return lines


def read2DAMJUELFitCoeffs(lines: List[str], maxTIndex=8, maxEIndex=8) -> np.ndarray:
    """Reads 2D AMJUEL fit coefficients from loaded lines

    Args:
        lines (List[str]): Split lines corresponding to an AMJUEL reaction with 2D fit
        maxTIndex (int, optional): Greated expected T index. Defaults to 8.
        maxEIndex (int, optional): Greatest expected E index. Defaults to 8.

    Returns:
        np.array: Numpy array with fit coefficients for reaction data from passed lines in [T,E] shape
    """

    rate = np.zeros([maxTIndex + 1, maxEIndex + 1])
    TIndex = 0
    EIndices = [0]
    expectData = False
    for line in lines:
        if line[0] == "E-Index:":
            expectData = False
            EIndices = [int(i) for i in line[1:]]

        if expectData:
            TIndex = int(line[0])
            rate[TIndex, EIndices] = [np.double(x.replace("D", "e")) for x in line[1:]]

            if TIndex == maxTIndex and max(EIndices) == maxEIndex:
                break

        if line[0] == "T-Index:":
            expectData = True

    return rate


def read1DAMJUELFitCoeffs(lines: List[str], maxIndex=8, coefName="b") -> np.ndarray:
    """Reads 1D AMJUEL fit coefficients from loaded lines

    Args:
        lines (List[str]): Split lines corresponding to an AMJUEL reaction data with 1D fit coefficients
        maxIndex (int, optional): Greatest expected index. Defaults to 8.
        coefName (str, optional): Coefficient name/prefix. Defaults to "b".

    Returns:
        np.array: Numpy array with fit coefficients for reaction data from passed lines
    """

    rate = np.zeros(maxIndex + 1)
    expectData = False
    index = 0
    for line in lines:
        for entry in line:
            if expectData:
                rate[index] = np.double(entry.replace("D", "e"))
                expectData = False
                if index == maxIndex:
                    break
            if coefName in entry:
                expectData = True
                index = int(entry[1:])

    return rate
