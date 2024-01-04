import timeit


class Poly:
    def __init__(self, id):
        self.previous = None
        self.next = None
        self.twin = None
        self.source = None
        self.target = None
        self.face = None
        self.id = id


def loop_halfedges(halfedges):
    halfedge = 1
    for step in range(100):
        halfedge = halfedges[halfedge].next.id


h1 = Poly(1)
h2 = Poly(2)
h3 = Poly(3)
h4 = Poly(4)

h1.next = h2
h2.next = h3
h3.next = h4
h4.next = h1

halfedges = {1: h1, 2: h2, 3: h3, 4: h4}

setup = """
class Poly:
    def __init__(self, id):
        self.previous = None
        self.next = None
        self.twin = None
        self.source = None
        self.target = None
        self.face = None
        self.id = id


def loop_halfedges(halfedges):
    halfedge = 1
    for step in range(100):
        halfedge = halfedges[halfedge].next.id


h1 = Poly(1)
h2 = Poly(2)
h3 = Poly(3)
h4 = Poly(4)

h1.next = h2
h2.next = h3
h3.next = h4
h4.next = h1

halfedges = {1: h1, 2: h2, 3: h3, 4: h4}
"""

timeit.timeit("loop_halfedges(halfedges)", setup=setup, number=10000)
