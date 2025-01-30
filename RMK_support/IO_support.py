# %%
import json
import h5py  # type: ignore
from . import variable_container as vc
from .grid import Grid
import xarray as xr
import numpy as np
from os import path, listdir
from os.path import isfile, join
from typing import Union, List
from copy import copy


def writeDictToJSON(
    data: dict, filepath="./config.json", prefixKey: Union[str, None] = None
):
    """Writes a dictionary into json file. If the file is already present it is updated using the data passed

    Args:
        data (dict): Data to write
        filepath (str, optional): I/O JSON filepath. Defaults to "./config.json".
        prefixKey (Union[str,None], optional): If not None the data dictionary is added as {prefixKey: data}. Defaults to None
    """
    newFile = not path.exists(filepath)
    usedData = data
    if prefixKey != None:
        usedData = {prefixKey: data}

    if newFile:
        with open(filepath, "w") as file:
            json.dump(usedData, file, indent=4, sort_keys=True)
    else:
        with open(filepath, "r+") as file:
            fileData = json.load(file)
            fileData.update(usedData)
            file.seek(0)
            json.dump(fileData, file, indent=4)


def writeRMKHDF5(varCont: vc.VariableContainer, filepath="./ReMKiT1DVarInput.h5"):
    """Write a ReMKiT1D readable HDF5 file

    Args:
        varCont (vc.VariableContainer): Reference variable container containing data to be written
        filepath (str, optional): Output filepath. Defaults to "./ReMKiT1DVarInput.h5".
    """
    with h5py.File(filepath, "w") as f:
        for varName in list(varCont.dataset.data_vars.keys()):
            data = varCont.dataset.data_vars[varName].to_numpy().flatten()
            f.create_dataset(varName, data=data)


def loadVariableFromHDF5(
    var: vc.Variable, filepaths: List[str] = ["./ReMKiT1DVarInput.h5"]
) -> vc.Variable:
    """Load a single variable from a list of HDF5 files, assuming they are at different timesteps.

    Args:
        var (vc.Variable): Variable to be loaded (assumed present in the HDF5 files)
        filepaths (List[str], optional): List of filepaths of HDF5 files. Defaults to ["./ReMKiT1DVarInput.h5"].

    Returns:
        vc.Variable: Loaded variable
    """

    if var.isScalar:
        buffer = np.zeros(len(filepaths))
    else:
        buffer = np.zeros(tuple([len(filepaths)] + list(var.values.shape)))
    for i, filepath in enumerate(filepaths):
        with h5py.File(filepath, "r") as f:
            dset = f[var.name]
            data = dset[:]
            buffer[i, ...] = data.reshape(var.values.shape)

    result = copy(var)
    if len(filepaths) > 1:
        result.addTimeDim(buffer)
    else:
        if var.isScalar:
            result.values = buffer
        else:
            result.values = buffer[0, ...]
    return result


def loadDummyVarFromHDF5(
    grid: Grid, varName: str, filepaths: List[str] = ["./ReMKiT1DVarInput.h5"]
) -> vc.Variable:
    """Load a variable without knowing all of its properties. Will attempt to identify the variable dimensionality.

    Args:
        grid (Grid): Grid used to determine dimensionality
        varName (str): Variable name to search for in the HDF5 files
        filepaths (List[str], optional): List of filepaths of HDF5 files. Defaults to ["./ReMKiT1DVarInput.h5"].

    Returns:
        vc.Variable: Loaded dummy variable
    """
    buffers: List[np.ndarray] = []
    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:
            dset = f[varName]
            data = dset[:]
            if len(data) > grid.numX:
                buffers.append(data.reshape((grid.numX, grid.numH, grid.numV)))
            else:
                buffers.append(data)

    if len(filepaths) == 1:
        buffer = buffers[0]
    else:
        buffer = np.zeros(tuple([len(filepaths)] + list(buffers[0].shape)))

        for i, _ in enumerate(filepaths):
            buffer[i, ...] = buffers[i]

    isScalar = buffers[0].shape == (1,) if grid.numX > 1 else False
    isDistribution = len(buffers[0].shape) == 3
    result = vc.Variable(
        varName,
        grid,
        data=buffer,
        timeDimSize=0 if len(buffers) == 1 else len(buffers),
        isScalar=isScalar,
        isDistribution=isDistribution,
    )

    return result


def loadVarContFromHDF5(
    *args: vc.Variable, filepaths: List[str] = ["./ReMKiT1DVarInput.h5"]
) -> vc.VariableContainer:
    """Load any number of variables from a list of HDF5 files and put them in a VariableContainer object. Assumes all variables are present.

    Args:
        filepaths (List[str], optional): List of filepaths of HDF5 files. Defaults to ["./ReMKiT1DVarInput.h5"].

    Returns:
        vc.VariableContainer: Variable container object with requested variables.
    """
    assert len(args), "loadVarContFromHDF5 called with no variable arguments"

    time: List[float] = []
    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:

            if "time" in f:
                dset = f["time"]
                time.append(dset[0])
            else:
                time.append(len(time) + 1)

    varCont = vc.VariableContainer(
        args[0].grid, timestamps=np.array(time), autoAddDuals=False
    )

    for var in args:
        varCont.add(loadVariableFromHDF5(var, filepaths))

    return varCont


def loadFromHDF5(
    grid: Grid, varNames: List[str], filepaths: List[str] = ["./ReMKiT1DVarInput.h5"]
) -> vc.VariableContainer:
    """Load a list of variables by name from a list of HDF5 files. Assumes no knowledge of variable dimensionality. Assumes all variable names are present in the loaded files.

    Args:
        grid (Grid): Grid used to determine dimensionality
        varNames (List[str]): List of variable names to search for in the HDF5 files
        filepaths (List[str], optional): List of filepaths of HDF5 files. Defaults to ["./ReMKiT1DVarInput.h5"].

    Returns:
        vc.VariableContainer: Variable container object with requested variables.
    """
    vars = (loadDummyVarFromHDF5(grid, name, filepaths) for name in varNames)

    time: List[float] = []
    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:

            if "time" in f:
                dset = f["time"]
                time.append(dset[0])
            else:
                time.append(len(time) + 1)

    varCont = vc.VariableContainer(grid, timestamps=np.array(time), autoAddDuals=False)

    varCont.add(*vars)
    return varCont


def getOutputFilenames(hdf5Dir: str) -> List[str]:
    """Search a given directory for all valid ReMKiT1D VarOutput files and return them as a sorted list

    Args:
        hdf5Dir (str): Directory to search for HDF5 files

    Returns:
        List[str]: Sorted list of output file names in given directory
    """
    onlyfiles = [
        f for f in listdir(hdf5Dir) if isfile(join(hdf5Dir, f)) if "VarOutput" in f
    ]
    onlyfiles.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return onlyfiles
