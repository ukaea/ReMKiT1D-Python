from typing import Union, List, cast, Dict
import copy


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
            if self.additiveMode:
                newNode = copy.deepcopy(self)
                if newNode.constant is not None:
                    newNode.constant = newNode.constant + float(rhs)
                else:
                    newNode.constant = float(rhs)
                newNode
            else:
                newNode = Node("none")
                newNode.additiveMode = True
                newNode.constant = float(rhs)
                newNode.children = [copy.deepcopy(self)]
        return newNode

    def __radd__(self, lhs):
        if isinstance(lhs, int) or isinstance(lhs, float):
            if self.additiveMode:
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
            if not self.additiveMode:
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
            if not self.additiveMode:
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
                newNode.unaryTransform = UnaryTransform("ipow", intParams=[-1])
                newNode.children = [copy.deepcopy(rhs)]

                topNode = Node("none")
                topNode.children = [copy.deepcopy(self), newNode]
                return topNode
            else:
                newNode = copy.deepcopy(rhs)
                newNode.unaryTransform = UnaryTransform("ipow", intParams=[-1])
                topNode = Node("none")
                topNode.children = [copy.deepcopy(self), newNode]
                return topNode
        if isinstance(rhs, int) or isinstance(rhs, float):
            if not self.additiveMode:
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
            if self.additiveMode:
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

    def __pow__(self, rhs):
        if isinstance(rhs, int):
            if self.unaryTransform is not None:
                newNode = Node("none")
                newNode.unaryTransform = UnaryTransform("ipow", intParams=[rhs])
                newNode.children = [copy.deepcopy(self)]
                return newNode
            else:
                newNode = copy.deepcopy(self)
                newNode.unaryTransform = UnaryTransform("ipow", intParams=[rhs])
                return newNode
        if isinstance(rhs, float):
            if self.unaryTransform is not None:
                newNode = Node("none")
                newNode.unaryTransform = UnaryTransform("rpow", realParams=[rhs])
                newNode.children = [copy.deepcopy(self)]

                return newNode
            else:
                newNode = copy.deepcopy(self)
                newNode.unaryTransform = UnaryTransform("rpow", realParams=[rhs])
                return newNode


class UnaryTransform:
    """Representation of ReMKiT1D's unary transformations. Should only ever be directly created for parameterized transformations."""

    def __init__(
        self,
        tag: str,
        realParams: List[float] = [],
        intParams: List[int] = [],
        logicalParams: List[bool] = [],
    ) -> None:
        self.__tag__ = tag
        self.__realParams__ = realParams
        self.__intParams__ = intParams
        self.__logicalParams__ = logicalParams

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


def log(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("log")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("log")
        return newNode


def exp(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("exp")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("exp")
        return newNode


def sin(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("sin")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("sin")
        return newNode


def cos(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("cos")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("cos")
        return newNode


def abs(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("abs")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("abs")
        return newNode


def sign(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("sign")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("sign")
        return newNode


def tan(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("tan")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("tan")
        return newNode


def atan(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("atan")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("atan")
        return newNode


def asin(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("asin")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("asin")
        return newNode


def acos(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("acos")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("acos")
        return newNode


def erf(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("erf")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("erf")
        return newNode


def erfc(node: Node) -> Node:
    if node.unaryTransform is not None:
        newNode = Node("none")
        newNode.unaryTransform = UnaryTransform("erfc")
        newNode.children = [copy.deepcopy(node)]
        return newNode
    else:
        newNode = copy.deepcopy(node)
        newNode.unaryTransform = UnaryTransform("erfc")
        return newNode
