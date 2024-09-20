import json
import h5py  # type: ignore
from . import variable_container as vc
import xarray as xr
import numpy as np
from os import path
from typing import Union, List


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


def loadFromHDF5(
    baseVarCont: vc.VariableContainer,
    filepaths=["./ReMKiT1DVarInput.h5"],
    varsToIgnore=[],
    isXinMeters=False,
) -> xr.Dataset:
    """Loads ReMKiT1D variable data from a given list of files into an xarray.Dataset object. The files are assumed to be different
    time slices.

    Args:
        baseVarCont (vc.VariableContainer): Variable container whose dataset should be extended into the time dimension
        filepaths (list, optional): List of files to load from. Defaults to ["./ReMKiT1DVarInput.h5"].
        varsToIgnore (list, optional): List of variable names to ignore when loading.
        isXinMeters (bool, optional): True if the x grid is in units of meters. Defaults to false.

    Returns:
        xr.Dataset: Output dataset
    """
    loadDataset = baseVarCont.dataset
    if "time" in list(baseVarCont.dataset.data_vars.keys()):
        loadDataset = loadDataset.drop("time")
    loadDataset = loadDataset.expand_dims({"time": len(filepaths)})
    buffer = {}

    for varName in list(loadDataset.data_vars.keys()):
        buffer[varName] = loadDataset[varName].copy()

    time: List[float] = []
    for filepath in filepaths:
        with h5py.File(filepath, "r") as f:
            for varName in list(loadDataset.data_vars.keys()):
                if not varName in varsToIgnore:
                    dset = f[varName]
                    data = dset[:]
                    buffer[varName][len(time), ...] = data.reshape(
                        buffer[varName].shape[1:]
                    )

            if "time" in list(baseVarCont.dataset.data_vars.keys()):
                dset = f["time"]
                time.append(dset[0])
            else:
                time.append(len(time) + 1)

    for varName in list(loadDataset.data_vars.keys()):
        if varName != "time" and varName not in varsToIgnore:
            loadDataset[varName] = buffer[varName]

    loadDataset.coords["time"] = np.array(time)
    loadDataset.coords["x"].attrs["units"] = "$x_0$"
    if isXinMeters:
        loadDataset.coords["x"].attrs["units"] = "$m$"
    loadDataset.coords["v"].attrs["units"] = "$v_{th}$"
    loadDataset.coords["time"].attrs["standard_name"] = "t"
    loadDataset.coords["time"].attrs["units"] = "$t_0$"

    return loadDataset
