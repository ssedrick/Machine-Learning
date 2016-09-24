from math import sqrt


class KDNode(object):
    __slots__ = ('dom_elt', 'split', 'left', 'right')

    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right


class Orthotope(object):
    __slots__ = ('min', 'max')

    def __init__(self, min, max):
        self.min = min
        self.max = max


class KDTree(object):
    __slots__ = ("n", "bounds")

    def __init__(self):
        def nk2(split, exset):
            if not exset:
                return None

