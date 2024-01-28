from firedrake import *
from netgen.occ import *
from naca import naca
n = 140
profile = "2412"
xNACA = naca(profile, n, False, False)[0]
yNACA = naca(profile, n, False, False)[1]
pnts = [Pnt(xNACA[i], yNACA[i], 0) for i in range(len(xNACA))]
spline = SplineApproximation(pnts)
airfoil = Face(Wire(spline)).Move((0.3,0.5,0)).Rotate(Axis((0.3,0.5,0), Z), -10)
circle = Circle(Pnt(0.37,0.5),0.07).Face()
shape = (Rectangle(4, 1).Face()-airfoil-circle)
shape.edges.name="wall"
shape.edges.Min(X).name="inlet"
shape.edges.Max(X).name="outlet"
geo = OCCGeometry(shape, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(Mesh(ngmesh).curve_field(4))
File("output/nacaMesh.pvd").write(mesh)