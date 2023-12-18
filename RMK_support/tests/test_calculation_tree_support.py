import RMK_support.calculation_tree_support as ct
import pytest
import numpy as np
from scipy import special  # type: ignore


@pytest.fixture
def varDict():
    return {"a": np.ones(3), "b": -2 * np.ones(3)}


def test_init_node():
    newNode = ct.Node("var")

    assert not newNode.additiveMode
    assert newNode.constant is None
    assert newNode.unaryTransform is None
    assert newNode.leafVar == "var"

    newNode.additiveMode = True
    newNode.constant = 1.0
    newNode.unaryTransform = ct.UnaryTransform("transform")

    assert newNode.additiveMode
    assert newNode.constant == 1.0
    assert newNode.unaryTransform.dict() == ct.UnaryTransform("transform").dict()

    assert newNode.dict() == {
        "isAdditiveNode": True,
        "constant": 1.0,
        "leafVariable": "var",
        "unaryTransform": "transform",
        "unaryRealParams": [],
        "unaryIntegerParams": [],
        "unaryLogicalParams": [],
    }


def test_addition(varDict):
    a = ct.Node("a")
    a.additiveMode = True
    b = ct.Node("b")

    a = a + 2
    b = b + 2
    c = 2 + (a + b)
    c = c + 2
    c = 3 + c

    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 4
    assert nodes[0].dict() == c.dict()
    assert parents == [-1, 0, 0, 2]
    assert children == [[2, 3], [], [4], []]

    assert nodes[0].constant == 7.0
    assert nodes[1].constant == 2.0
    assert nodes[2].constant == 2.0

    assert all(c.evaluate(varDict) == 10 * np.ones(3))


def test_multiplication(varDict):
    a = ct.Node("a")
    b = ct.Node("b")
    a = a * 2
    b.additiveMode = True
    c = 2 * (a * (a + b * 2))
    c = c * 2
    c = 3 * c
    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 6
    assert parents == [-1, 0, 0, 2, 2, 4]
    assert children == [[2, 3], [], [4, 5], [], [6], []]

    assert nodes[0].constant == 12.0
    assert nodes[1].constant == 2.0

    assert nodes[4].constant == 2.0

    assert all(c.evaluate(varDict) == -48 * np.ones(3))


def test_div(varDict):
    a = ct.Node("a")
    b = ct.Node("b")
    c = a / b
    c = c / 2
    c = c / 2

    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 3
    assert nodes[0].constant == 0.25
    assert nodes[2].unaryTransform.dict()["unaryTransform"] == "ipow"
    assert nodes[2].unaryTransform.dict()["unaryIntegerParams"] == [-1]

    assert all(c.evaluate(varDict) == -np.ones(3) / 8)


def test_rtruediv(varDict):
    a = ct.Node("a")
    b = ct.Node("b")
    c = 5 / a
    d = 1 / b

    nodesc, _, _ = ct.flattenTree(c)
    nodesd, _, _ = ct.flattenTree(d)

    assert len(nodesc) == 2
    assert nodesc[0].constant == 5
    assert len(nodesd) == 2
    assert nodesd[0].constant == 1

    assert all(c.evaluate(varDict) == 5 * np.ones(3))
    assert all(d.evaluate(varDict) == -0.5 * np.ones(3))


def test_sub(varDict):
    a = ct.Node("a")
    b = ct.Node("b")

    c = a - b
    c = c - 2

    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 3
    assert nodes[0].constant == -2.0

    assert all(c.evaluate(varDict) == np.ones(3))


def test_pow(varDict):
    a = ct.Node("a")
    b = ct.Node("b")

    c = (a**3.0) ** 2 + (b**2) ** 3.0

    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 5
    assert nodes[1].unaryTransform.dict()["unaryTransform"] == "ipow"
    assert nodes[1].unaryTransform.dict()["unaryIntegerParams"] == [2]

    assert nodes[2].unaryTransform.dict()["unaryTransform"] == "rpow"
    assert nodes[2].unaryTransform.dict()["unaryRealParams"] == [3.0]

    assert nodes[3].unaryTransform.dict()["unaryTransform"] == "rpow"
    assert nodes[3].unaryTransform.dict()["unaryRealParams"] == [3.0]

    assert nodes[4].unaryTransform.dict()["unaryTransform"] == "ipow"
    assert nodes[4].unaryTransform.dict()["unaryIntegerParams"] == [2]

    assert all(c.evaluate(varDict) == 65 * np.ones(3))


def test_unary_call():
    a = ct.Node("a")

    fun = ct.UnaryTransform("fun")

    b = fun(a)
    b = fun(b)

    nodes, parents, children = ct.flattenTree(b)

    assert len(nodes) == 2

    assert nodes[0].unaryTransform.dict()["unaryTransform"] == "fun"
    assert nodes[1].unaryTransform.dict()["unaryTransform"] == "fun"


def test_neg(varDict):
    a = ct.Node("a")

    b = -a
    b = -b

    nodes, parents, children = ct.flattenTree(b)

    assert len(nodes) == 1
    assert nodes[0].constant == 1.0

    assert all(b.evaluate(varDict) == np.ones(3))


def test_unaryTransformation_mul():
    a = ct.Node("a")
    fun = ct.UnaryTransform("fun")

    b = fun(a) * 2
    c = 2 * fun(a)

    nodes, parents, children = ct.flattenTree(b)

    assert len(nodes) == 2

    assert nodes[1].unaryTransform.dict()["unaryTransform"] == "fun"
    assert nodes[0].constant == 2.0

    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 2

    assert nodes[1].unaryTransform.dict()["unaryTransform"] == "fun"
    assert nodes[0].constant == 2.0


def test_unaryTransformation_add():
    a = ct.Node("a")
    a.additiveMode = True
    fun = ct.UnaryTransform("fun")

    b = fun(a) + 2
    c = 2 + fun(a)

    nodes, parents, children = ct.flattenTree(b)

    assert len(nodes) == 2

    assert nodes[1].unaryTransform.dict()["unaryTransform"] == "fun"
    assert nodes[0].constant == 2.0

    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 2

    assert nodes[1].unaryTransform.dict()["unaryTransform"] == "fun"
    assert nodes[0].constant == 2.0


def test_unaryTransformation_sub():
    a = ct.Node("a")
    a.additiveMode = True
    fun = ct.UnaryTransform("fun")

    b = fun(a) - 2
    c = 2 - fun(a)

    nodes, parents, children = ct.flattenTree(b)

    assert len(nodes) == 2

    assert nodes[1].unaryTransform.dict()["unaryTransform"] == "fun"
    assert nodes[0].constant == -2.0

    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 3

    assert nodes[2].unaryTransform.dict()["unaryTransform"] == "fun"
    assert nodes[1].constant == -1.0
    assert nodes[0].constant == 2.0


def test_funs(varDict):
    funs = [
        ct.log,
        ct.exp,
        ct.abs,
        ct.sin,
        ct.cos,
        ct.acos,
        ct.sign,
        ct.asin,
        ct.tan,
        ct.atan,
        ct.erf,
        ct.erfc,
    ]

    funTags = [
        "log",
        "exp",
        "abs",
        "sin",
        "cos",
        "acos",
        "sign",
        "asin",
        "tan",
        "atan",
        "erf",
        "erfc",
    ]

    numFuns = [
        np.log,
        np.exp,
        np.abs,
        np.sin,
        np.cos,
        np.arccos,
        np.sign,
        np.arcsin,
        np.tan,
        np.arctan,
        special.erf,
        special.erfc,
    ]

    for i, fun in enumerate(funs):
        a = ct.Node("a")
        b = fun(a)
        assert all(b.evaluate(varDict) == numFuns[i](np.ones(3)))

        b = fun(b)

        nodes, parents, children = ct.flattenTree(b)

        assert len(nodes) == 2

        assert nodes[0].unaryTransform.dict()["unaryTransform"] == funTags[i]
        assert nodes[1].unaryTransform.dict()["unaryTransform"] == funTags[i]


def test_radd_multNode(varDict):
    a = ct.Node("a")

    a = 2 + a

    nodes, parents, children = ct.flattenTree(a)

    assert len(nodes) == 2

    assert nodes[0].constant == 2.0

    assert all(a.evaluate(varDict) == 3 * np.ones(3))


def test_rmul_addNode(varDict):
    a = ct.Node("a")
    a.additiveMode = True
    a = 2 * a

    nodes, parents, children = ct.flattenTree(a)

    assert len(nodes) == 2

    assert nodes[0].constant == 2.0

    assert all(a.evaluate(varDict) == 2 * np.ones(3))


def test_div_addNode(varDict):
    a = ct.Node("a")
    a.additiveMode = True
    a = a / 2

    nodes, parents, children = ct.flattenTree(a)

    assert len(nodes) == 2

    assert nodes[0].constant == 0.5

    assert all(a.evaluate(varDict) == 0.5 * np.ones(3))


def test_div_unaryNode(varDict):
    a = ct.Node("a")
    b = ct.Node("b")
    a = ct.exp(a)
    c = b / a

    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 4

    assert nodes[2].unaryTransform.dict()["unaryTransform"] == "ipow"
    assert nodes[2].unaryTransform.dict()["unaryIntegerParams"] == [-1]

    assert nodes[3].unaryTransform.dict()["unaryTransform"] == "exp"

    assert all(c.evaluate(varDict) == -2 * np.ones(3) / np.exp(1))


def test_sub_addNode(varDict):
    a = ct.Node("a")
    b = ct.Node("b")

    b.additiveMode = True

    b = b - 2

    b = b - 2

    c = a - b
    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 4
    assert not nodes[2].additiveMode
    assert nodes[3].constant == -4.0

    assert all(c.evaluate(varDict) == 7 * np.ones(3))


def test_sub_multConstNode(varDict):
    a = ct.Node("a")
    b = ct.Node("b")

    b.constant = 2.0

    c = a - b
    nodes, parents, children = ct.flattenTree(c)

    assert len(nodes) == 3
    assert nodes[2].constant == -2.0

    assert all(c.evaluate(varDict) == 5 * np.ones(3))


def test_calc_deriv():
    a = ct.Node("a")
    b = ct.Node("b")

    c = a + ct.exp(b)

    assert ct.treeDerivation(c) == {
        "type": "calculationTreeDerivation",
        "numNodes": 3,
        "nodes": {
            "index1": {
                "isAdditiveNode": True,
                "leafVariable": "none",
                "parent": 0,
                "children": [2, 3],
            },
            "index2": {
                "isAdditiveNode": False,
                "leafVariable": "a",
                "parent": 1,
                "children": [],
            },
            "index3": {
                "isAdditiveNode": False,
                "leafVariable": "b",
                "unaryTransform": "exp",
                "unaryRealParams": [],
                "unaryIntegerParams": [],
                "unaryLogicalParams": [],
                "parent": 1,
                "children": [],
            },
        },
    }


def testAddLeafVars():
    a = ct.Node("a")
    b = ct.Node("b")

    c = a + a * b

    leafVars = ct.getLeafVars(c)

    assert leafVars == ["a", "b"]
