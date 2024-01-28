from firedrake import *
from netgen.occ import *
from naca import naca
from solvers import paramsFullMG
n = 140
profile = "2412"
xNACA = naca(profile, n, False, False)[0]
yNACA = naca(profile, n, False, False)[1]
pnts = [Pnt(xNACA[i], yNACA[i], 0) for i in range(len(xNACA))]
spline = SplineApproximation(pnts)
circle = Circle(Pnt(0.37,0.5),0.07).Face()
airfoil = Face(Wire(spline)).Move((0.3,0.5,0)).Rotate(Axis((0.3,0.5,0), Z), -10)
shape = (Rectangle(4, 1).Face()-airfoil-circle)
shape.edges.name="wall"
shape.edges.Min(X).name="inlet"
shape.edges.Max(X).name="outlet"
geo = OCCGeometry(shape, dim=2)
ngmesh = geo.GenerateMesh(maxh=0.1)
mesh = Mesh(ngmesh)
from ngsPETSc import NetgenHierarchy
mh = NetgenHierarchy(ngmesh,2, 2)
mesh = mh[-1]

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
nu = Constant(1e-3)
a = (nu*inner(grad(u), grad(v)) - p * div(v) + div(u) * q)*dx
L = inner(Constant((0, 0)), v) * dx


sol0 = Function(Z)
x, y = SpatialCoordinate(mesh)
inflowoutflow = Function(V).interpolate(as_vector([-1, 0]))
labelsInlet = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["outlet"]]
labelsWall = [i+1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["wall","cyl"]]
bcs = [DirichletBC(Z.sub(0), inflowoutflow, labelsInlet),
       DirichletBC(Z.sub(0), zero(2), labelsWall)]
nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

class Mass(AuxiliaryOperatorPC): 
    def form(self, pc, test, trial):
        a=1/nu*inner(test, trial)*dx
        bcs = None
        return (a, bcs)
paramsFullMG["ksp_monitor"] = None
solve(a == L, sol0, bcs=bcs, solver_parameters=paramsFullMG)

#Solving Stokes for initial data

u0, p0 = sol0.split()
u0.rename("velocity")
p0.rename("pressure")
divergence = abs(assemble(div(u0)*div(u0)*dx))

print(GREEN % ("Solved Stokes flow with nu: {}, div: {}".format(float(nu), divergence)))
File("output/multigrid.pvd").write(u0)