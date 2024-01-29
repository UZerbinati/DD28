from firedrake import *
from netgen.occ import *
from naca import naca
from solvers import paramsLU, eps
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
ngmesh.SplitAlfeld()
mesh = Mesh(ngmesh)
File("output/nacaMesh.pvd").write(mesh)


V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "DG", 1)
Z = V * W
u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
nu = Constant(1e-3)
a = (nu*inner(eps(u), eps(v)) - p * div(v) - div(u) * q)*dx
L = inner(Constant((0, 0)), v) * dx


sol0 = Function(Z)
x, y = SpatialCoordinate(mesh)
inflowoutflow = Function(V).interpolate(as_vector([-1, 0]))
labelsInlet = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["outlet"]]
labelsWall = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["wall","cyl"]]
bcs = [DirichletBC(Z.sub(0), inflowoutflow, labelsInlet),
       DirichletBC(Z.sub(0), zero(2), labelsWall)]
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
solve(a == L, sol0, bcs=bcs, solver_parameters=paramsLU)

#Solving Stokes for initial data

u0, p0 = sol0.split()
u0.rename("velocity")
p0.rename("pressure")
divergence = abs(assemble(div(u0)*div(u0)*dx))

print(GREEN % ("Solved Stokes flow with nu: {}, div: {}".format(float(nu), divergence)))
File("output/scottVogelius.pvd").write(u0)