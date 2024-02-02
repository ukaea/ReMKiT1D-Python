# %%
from typing import Union, List, cast, Dict, Callable
import copy
from itertools import accumulate
import numpy as np
from scipy import special  # type: ignore


class Node:
    """Calculation tree node class. Should only ever be invoked for leaf nodes representing variables."""

    def __init__(self, leafVar: str) -> None:
        self.__leafVar__ = leafVar
        self.__additiveMode__ = False
        self.__constant__: Union[None, float] = None
        self.__unaryTransform__: Union[None, UnaryTransform] = None
        self.children: List[Node] = []

    @property
    def leafVar(self):
        return self.__leafVar__

    @property
    def additiveMode(self):
        return self.__additiveMode__

    @additiveMode.setter
    def additiveMode(self, isAdditive: bool):
        self.__additiveMode__ = isAdditive

    @property
    def constant(self):
        return self.__constant__

    @constant.setter
    def constant(self, const: float):
        self.__constant__ = const

    @property
    def unaryTransform(self):
        return self.__unaryTransform__

    @unaryTransform.setter
    def unaryTransform(self, transform):
        self.__unaryTransform__ = transform

    def dict(self) -> dict:
        """Return ReMKiT1D format dictionary representing this node's kernel

        Returns:
            dict: Node kernel dictionary
        """
        node = {"isAdditiveNode": self.additiveMode}

        if self.constant is not None:
            node.update({"constant": self.constant})
        if self.leafVar is not None:
            node.update({"leafVariable": self.leafVar})
        if self.unaryTransform is not None:
            node.update(self.unaryTransform.dict())

        return node

    def __add__(self, rhs):
        if isinstance(rhs, Node):
            newNode = Node("none")
            newNode.additiveMode = True
            newNode.children = [copy.deepcopy(self), copy.deepcopy(rhs)]
        if isinstance(rhs, int) or isinstance(rhs, float):
            if self.additiveMode and self.unaryTransform is None:
                newNode = copy.deepcopy(self)
                if newNode.constant is not None:
                    newNode.constant = newNode.constant + float(rhs)
                else:
                    newNode.constant = float(rhs)
            else:
                newNode = Node("none")
                newNode.additiveMode = True
                newNode.constant = float(rhs)
                newNode.children = [copy.deepcopy(self)]
        return newNode

    def __radd__(self, lhs):
        if isinstance(lhs, int) or isinstance(lhs, float):
            if self.additiveMode and self.unaryTransform is None:
                newNode = copy.deepcopy(self)
                if newNode.constant is not None:
                    newNode.constant = newNode.constant + float(lhs)
                else:
                    newNode.constant = float(lhs)
            else:
                newNode = Node("none")
                newNode.additiveMode = True
                newNode.constant = float(lhs)
                newNode.children = [copy.deepcopy(self)]
        return newNode

    def __mul__(self, rhs):
        if isinstance(rhs, Node):
            newNode = Node("none")
            newNode.children = [copy.deepcopy(self), copy.deepcopy(rhs)]
        if isinstance(rhs, int) or isinstance(rhs, float):
            if not self.additiveMode and self.unaryTransform is None:
                newNode = copy.deepcopy(self)
                if newNode.constant is not None:
                    newNode.constant = newNode.constant * float(rhs)
                else:
                    newNode.constant = float(rhs)
                return newNode
            else:
                newNode = Node("none")
                newNode.constant = float(rhs)
                newNode.children = [copy.deepcopy(self)]
        return newNode

    def __rmul__(self, lhs):
        if isinstance(lhs, int) or isinstance(lhs, float):
            if not self.additiveMode and self.unaryTransform is None:
                newNode = copy.deepcopy(self)
                if newNode.constant is not None:
                    newNode.constant = newNode.constant * float(lhs)
                else:
                    newNode.constant = float(lhs)
            else:
                newNode = Node("none")
                newNode.constant = float(lhs)
                newNode.children = [copy.deepcopy(self)]
        return newNode

    def __truediv__(self, rhs):
        if isinstance(rhs, Node):
            if rhs.unaryTransform is not None:
                newNode = Node("none")
                newNode.unaryTransform = powUnary(-1)
                newNode.children = [copy.deepcopy(rhs)]

                topNode = Node("none")
                topNode.children = [copy.deepcopy(self), newNode]
                return topNode
            else:
                newNode = copy.deepcopy(rhs)
                newNode.unaryTransform = powUnary(-1)
                topNode = Node("none")
                topNode.children = [copy.deepcopy(self), newNode]
                return topNode
        if isinstance(rhs, int) or isinstance(rhs, float):
            if not self.additiveMode and self.unaryTransform is None:
                newNode = copy.deepcopy(self)
                if newNode.constant is not None:
                    newNode.constant = newNode.constant / float(rhs)
                else:
                    newNode.constant = 1 / float(rhs)
                return newNode
            else:
                newNode = Node("none")
                newNode.constant = 1 / float(rhs)
                newNode.children = [copy.deepcopy(self)]
                return newNode

    def __rtruediv__(self, lhs):
        if isinstance(lhs, int) or isinstance(lhs, float):
            if self.unaryTransform is not None:
                newNode = Node("none")
                newNode.constant = 1 / float(lhs)
                newNode.unaryTransform = powUnary(-1)
                newNode.children = [copy.deepcopy(self)]

                return newNode
            else:
                newNode = copy.deepcopy(self)
                newNode.unaryTransform = powUnary(-1)

                topNode = Node("none")
                topNode.children = [newNode]
                topNode.constant = float(lhs)
                return topNode

    def __sub__(self, rhs):
        if isinstance(rhs, Node):
            if rhs.additiveMode:
                newNode = Node("none")
                newNode.constant = -1.0
                newNode.children = [copy.deepcopy(rhs)]

                topNode = Node("none")
                topNode.additiveMode = True
                topNode.children = [copy.deepcopy(self), newNode]
                return topNode
            else:
                newNode = copy.deepcopy(rhs)
                if newNode.constant is not None:
                    newNode.constant = -newNode.constant
                else:
                    newNode.constant = -1.0
                topNode = Node("none")
                topNode.additiveMode = True
                topNode.children = [copy.deepcopy(self), newNode]
                return topNode
        if isinstance(rhs, int) or isinstance(rhs, float):
            if self.additiveMode and self.unaryTransform is None:
                newNode = copy.deepcopy(self)
                if newNode.constant is not None:
                    newNode.constant = newNode.constant - float(rhs)
                else:
                    newNode.constant = -float(rhs)
                return newNode
            else:
                newNode = Node("none")
                newNode.additiveMode = True
                newNode.constant = -float(rhs)
                newNode.children = [self]
                return newNode

    def __neg__(self):
        if not self.additiveMode and self.unaryTransform is None:
            newNode = copy.deepcopy(self)
            if newNode.constant is not None:
                newNode.constant = -newNode.constant
            else:
                newNode.constant = -1.0
            return newNode
        else:
            newNode = Node("none")
            newNode.constant = -1.0
            newNode.children = [self]
            return newNode

    def __rsub__(self, lhs):
        if isinstance(lhs, int) or isinstance(lhs, float):
            return lhs + (-self)

    def __pow__(self, rhs):
        if isinstance(rhs, int):
            if self.unaryTransform is not None:
                newNode = Node("none")
                newNode.unaryTransform = powUnary(rhs)
                newNode.children = [copy.deepcopy(self)]
                return newNode
            else:
                newNode = copy.deepcopy(self)
                newNode.unaryTransform = powUnary(rhs)
                return newNode
        if isinstance(rhs, float):
            if self.unaryTransform is not None:
                newNode = Node("none")
                newNode.unaryTransform = powUnary(rhs)
                newNode.children = [copy.deepcopy(self)]

                return newNode
            else:
                newNode = copy.deepcopy(self)
                newNode.unaryTransform = powUnary(rhs)
                return newNode

    def evaluate(self, varDict: Dict[str, np.ndarray]) -> np.ndarray:
        const = 1 if self.constant is None else self.constant
        if self.additiveMode:
            const = 0 if self.constant is None else self.constant

        if self.leafVar != "none":
            childrenResult = varDict[self.leafVar].copy()

            if self.additiveMode:
                childrenResult += const
            else:
                childrenResult *= const
        else:
            if self.additiveMode:
                childrenResult = (
                    list(
                        accumulate(
                            [child.evaluate(varDict) for child in self.children],
                            (lambda x, y: x + y),
                        )
                    )[-1]
                    + const
                )

            else:
                childrenResult = (
                    list(
                        accumulate(
                            [child.evaluate(varDict) for child in self.children],
                            (lambda x, y: x * y),
                        )
                    )[-1]
                    * const
                )

        if self.unaryTransform is None:
            return childrenResult

        return self.unaryTransform.evaluate(childrenResult)


class UnaryTransform:
    """Representation of ReMKiT1D's unary transformations. Should only ever be directly created for parameterized transformations."""

    def __init__(
        self,
        tag: str,
        realParams: List[float] = [],
        intParams: List[int] = [],
        logicalParams: List[bool] = [],
        unaryCallable: Union[
            Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray], None
        ] = None,
    ) -> None:
        self.__tag__ = tag
        self.__realParams__ = realParams
        self.__intParams__ = intParams
        self.__logicalParams__ = logicalParams
        self.__unaryCallable__ = unaryCallable

    def evaluate(self, array: np.ndarray) -> np.ndarray:
        """Evaluate the unary transform on given array if the callable is defined, otherwise throw error

        Args:
            array (np.ndarray): Array to transform

        Returns:
            np.ndarray: Result of applying the unary transformation on a numpy array
        """

        assert self.__unaryCallable__ is not None, (
            "Unary transform "
            + self.__tag__
            + " does not have a callable defined and cannot be evaluated"
        )

        return self.__unaryCallable__(
            self.__realParams__, self.__intParams__, self.__logicalParams__, array
        )

    def dict(self) -> dict:
        """Return unary transformation properties as ReMKiT1D dictionary

        Returns:
            dict: Transformation dictionary readable by ReMKiT1D
        """
        return {
            "unaryTransform": self.__tag__,
            "unaryRealParams": self.__realParams__,
            "unaryIntegerParams": self.__intParams__,
            "unaryLogicalParams": self.__logicalParams__,
        }

    def __call__(self, node: Node) -> Node:
        """Applies unary transformation to tree with given root node

        Args:
            node (Node): Root node of tree to apply the transformation to.

        Returns:
            Node: Root node of tree with the transformation applied.
        """
        if node.unaryTransform is not None:
            newNode = Node("none")
            newNode.unaryTransform = self
            newNode.children = [copy.deepcopy(node)]
            return newNode
        else:
            newNode = copy.deepcopy(node)
            newNode.unaryTransform = self
            return newNode


class NodeIterator:
    """Helper class for iterating over a calculation tree"""

    def __init__(self) -> None:
        self.__nodeCounter__ = 0

    def traverse(self, node: Node, fun, counter, parentsCounter, *args, **kwargs):
        fun(node, counter, parentsCounter, *args, **kwargs)
        thisCounter = copy.copy(self.__nodeCounter__)
        self.__nodeCounter__ += 1
        for child in node.children:
            self.traverse(
                child, fun, self.__nodeCounter__, thisCounter, *args, **kwargs
            )

        if thisCounter == 0:
            self.__nodeCounter__ = 0


def packTree(self, nodeCounter, parentsCounter, container, parents):
    container.append(copy.deepcopy(self))
    parents.append(parentsCounter)


def flattenTree(node: Node):
    container: List[Node] = []
    parents: List[int] = []
    nodeIter = NodeIterator()
    nodeIter.traverse(node, packTree, 0, -1, container, parents)
    children = [
        [i + 1 for i, x in enumerate(parents) if x == j] for j in range(len(parents))
    ]

    return container, parents, children


def addLeafVar(self: Node, nodeCounter, parentsCounter, container):
    if self.leafVar != "none" and self.leafVar not in container:
        container.append(self.leafVar)


def getLeafVars(node: Node):
    nodeIter = NodeIterator()
    container: List[str] = []
    nodeIter.traverse(node, addLeafVar, 0, -1, container)

    return container


def treeDerivation(rootNode: Node) -> dict:
    """Return derivation which performs its calculation based on

    Args:
        rootNode (Node): Root node (containing the entire tree)

    Returns:
        dict: Derivation property dictionary
    """
    nodes, parents, children = flattenTree(rootNode)
    numNodes = len(nodes)
    treeDict = {
        "type": "calculationTreeDerivation",
        "numNodes": numNodes,
        "nodes": dict(),
    }

    for i in range(numNodes):
        cast(Dict[str, dict], treeDict["nodes"])["index" + str(i + 1)] = {}
        cast(Dict[str, dict], treeDict["nodes"])["index" + str(i + 1)].update(
            nodes[i].dict()
        )
        cast(Dict[str, dict], treeDict["nodes"])["index" + str(i + 1)].update(
            {"parent": parents[i] + 1, "children": children[i]}
        )

    return treeDict


def powUnary(power: Union[int, float]) -> UnaryTransform:
    if isinstance(power, int):
        func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
            lambda floats, ints, bools, arg: arg ** ints[0]
        )
        transform = UnaryTransform("ipow", intParams=[power], unaryCallable=func)
    else:
        func = lambda floats, ints, bools, arg: arg ** floats[0]
        transform = UnaryTransform("rpow", realParams=[power], unaryCallable=func)
    return transform


def log(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.log(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("log", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("log", unaryCallable=func)
        return newNode


def exp(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.exp(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("exp", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("exp", unaryCallable=func)
        return newNode


def sin(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.sin(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("sin", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("sin", unaryCallable=func)
        return newNode


def cos(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.cos(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("cos", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("cos", unaryCallable=func)
        return newNode


def abs(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.abs(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("abs", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("abs", unaryCallable=func)
        return newNode


def sign(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.sign(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("sign", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("sign", unaryCallable=func)
        return newNode


def tan(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.tan(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("tan", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("tan", unaryCallable=func)
        return newNode


def atan(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.arctan(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("atan", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("atan", unaryCallable=func)
        return newNode


def asin(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.arcsin(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("asin", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("asin", unaryCallable=func)
        return newNode


def acos(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: np.arccos(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("acos", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("acos", unaryCallable=func)
        return newNode


def erf(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: special.erf(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("erf", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("erf", unaryCallable=func)
        return newNode


def erfc(node: Node) -> Node:
    func: Callable[[List[float], List[int], List[bool], np.ndarray], np.ndarray] = (
        lambda floats, ints, bools, arg: special.erfc(arg)
    )
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("erfc", unaryCallable=func)
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("erfc", unaryCallable=func)
        return newNode


# %%
