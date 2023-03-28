import numpy as np
import ufl
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         extract_function_spaces, form,
                         locate_dofs_topological, locate_dofs_geometrical)
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
from ufl import div, dx, grad, inner
from dolfinx import fem, la
from dolfinx.io import gmshio
from math import *
import sys 
import gmsh

def gmsh_stokes_tutorial(par):
    # par is the data class containing model parameters.
    w  = par.w
    h1 = par.h1
    h2 = par.h2
    dxx= par.dxx
    ra = par.ra
    rb = par.rb
    rot= par.rot
    xe = par.xe
    ye = par.ye
    dxx_curv = par.dxx_curv
    # the function uses Gmsh Python API to create mesh. 
    gmsh.initialize()
    gmsh.model.add("stokes")

    gmsh.model.occ.addPoint(0., 0., 0., dxx, 1)
    gmsh.model.occ.addPoint(w,  0., 0., dxx, 2)
    gmsh.model.occ.addPoint(w,  h1, 0., dxx, 3)
    gmsh.model.occ.addPoint(0., h1, 0., dxx, 4)
    gmsh.model.occ.addPoint(0., h1+h2,  0., dxx, 5)
    gmsh.model.occ.addPoint(w,  h1+h2,  0., dxx, 6)

    gmsh.model.occ.addLine(1,2,1) # bottom
    gmsh.model.occ.addLine(2,3,2) # right down
    gmsh.model.occ.addLine(3,4,3) # top middle
    gmsh.model.occ.addLine(4,1,4) # left down
    gmsh.model.occ.addLine(4,5,5) # left up
    gmsh.model.occ.addLine(5,6,6) # top
    gmsh.model.occ.addLine(6,3,7) # right up

    #gmsh.model.occ.addCircle(h/2,l/2,0,r, 5, angle1=0., angle2=2*pi)
    # create the elliptical inclusion
    gmsh.model.occ.addEllipse(xe, ye, 0., ra, rb, 8, angle1=0., angle2=2.*pi) 
    gmsh.model.occ.rotate([(1,8)], xe, ye, 0., 0, 0, 1, rot) 
    # [(1,8)] stands for 1 - dimension of the entitiy, here it is an ellipse loop. 
    # 8 stands for the tag of that entity.

    gmsh.model.occ.addCurveLoop([3,7,6,5],1) # create the loop for the top rectangle
    gmsh.model.occ.addCurveLoop([1,2,3,4],2) # create the loop for the bottom rectangle 
    gmsh.model.occ.addCurveLoop([8],3)       # create the loop for the inclusion

    gmsh.model.occ.addPlaneSurface([1],1)    # create the plane for the top rectangle
    gmsh.model.occ.addPlaneSurface([2,3],2)  # create the plane for the (bottom rectangle - inclusion)
    gmsh.model.occ.addPlaneSurface([3],3)    # create the plane for the inclusion        

    gmsh.model.occ.synchronize()

    #surfs = gmsh.model.occ.getEntities(dim=2)
    #print(surfs)
    lines = gmsh.model.occ.getEntities(dim=1)
    #print(lines)
    left_marker   = 12
    right_marker  = 14
    top_marker    = 15
    bottom_marker = 13

    # ascribe two subdomains.
    crust_marker  = 9
    mantle_marker = 10
    inc_marker    = 11

    for surf in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surf[0], surf[1])
        #print('surf', surf)
        #print('com ', com)
        # the structure of surf is [(2,1),(2,2)] in this case.
        # the first column 2 indicates the dimension of the entities. So, 2 means surfaces.
        # the second column is the index.
        if surf[1] == 1:
            crust_matrix = surf[1]
        elif surf[1] == 2:
            mantle_matrix= surf[1]
        elif surf[1] == 3:
            inc        = surf[1]

    gmsh.model.addPhysicalGroup(2, [crust_matrix],  crust_marker,  "crust")
    gmsh.model.addPhysicalGroup(2, [mantle_matrix], mantle_marker, "mantle")
    gmsh.model.addPhysicalGroup(2, [inc],           inc_marker,    "anom")

    # ascribe line boundary conditions.
    left   = []
    bottom = []
    top    = []
    right  = []
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[0],0):
            left.append(line[1])
        if np.isclose(com[1],0):
            bottom.append(line[1])
        if np.isclose(com[0],w):
            right.append(line[1])
        if np.isclose(com[1],h1+h2):
            top.append(line[1])

    gmsh.model.addPhysicalGroup(1, bottom, bottom_marker, "bottom")
    gmsh.model.addPhysicalGroup(1, right, right_marker, "right")
    gmsh.model.addPhysicalGroup(1, top, top_marker, "top")
    gmsh.model.addPhysicalGroup(1, left, left_marker, "left")

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", dxx_curv)
    gmsh.model.mesh.generate(2)

    gmsh.write('stokes.msh')
    gmsh.finalize()

    return crust_marker, mantle_marker, inc_marker, \
            left_marker, right_marker, top_marker, bottom_marker

def fem_main(crust_marker, mantle_marker, inc_marker, 
             left_marker, right_marker, top_marker, bottom_marker, 
             par):
    # par is the data class containing model parameters.
    eta_crust  = par.eta_crust
    eta_mantle = par.eta_mantle
    eta_inc    = par.eta_inc
    bcs_type   = par.bcs_type
    dgrav      = par.dgrav
    
    # use dolfinx.gmshio to load Gmsh meshes. 
    msh, ct, ft = gmshio.read_from_msh("stokes.msh", MPI.COMM_WORLD, gdim=2)
    # msh is the mesh;
    # ct is cell markers;
    # ft is facet markers.

    # create DG0 space for constants.
    Const = FunctionSpace(msh, ("DG",0)) # create DG0 function space for constants. 
    eta   = Function(Const)
    dgx   = Function(Const)
    dgy   = Function(Const)

    #print(dgx, dgy)

    # locate the cells for different domains (up, down, anomaly)
    crust_cells     = ct.find(crust_marker)
    mantle_cells    = ct.find(mantle_marker)
    inc_cells       = ct.find(inc_marker)

    # assign viscosity nu based on domains.
    eta.x.array[crust_cells]  = np.full_like(crust_cells,  eta_crust, dtype=PETSc.ScalarType)
    eta.x.array[mantle_cells] = np.full_like(mantle_cells, eta_mantle,  dtype=PETSc.ScalarType)
    eta.x.array[inc_cells]    = np.full_like(inc_cells,    eta_inc,  dtype=PETSc.ScalarType)

    # assign body force x component based on domains. Should be all zeros. 
    dgx.x.array[crust_cells]  = np.full_like(crust_cells,  0.0, dtype=PETSc.ScalarType)
    dgx.x.array[mantle_cells] = np.full_like(mantle_cells, 0.0, dtype=PETSc.ScalarType)
    dgx.x.array[inc_cells]    = np.full_like(inc_cells,    0.0, dtype=PETSc.ScalarType)

    # assign body force y component based on domains. Should be non-zero in the anomaly ellipse.
    dgy.x.array[crust_cells]  = np.full_like(crust_cells,  0.0, dtype=PETSc.ScalarType)
    dgy.x.array[mantle_cells] = np.full_like(mantle_cells, 0.0, dtype=PETSc.ScalarType)
    dgy.x.array[inc_cells]    = np.full_like(inc_cells,    dgrav,  dtype=PETSc.ScalarType)

    # locate facets for different boundaries based on markers. 
    left_facets  = ft.find(left_marker)
    right_facets = ft.find(right_marker)
    top_facets   = ft.find(top_marker)
    bottom_facets= ft.find(bottom_marker)
    #print('left_facets = ', left_facets)
    #print('right_facets= ', right_facets)
    #print('top_facets  = ', top_facets)
    #print('bottom_facets = ', bottom_facets)

    # create mesh domain with dolfinx's meshing capability, which is quite limitied.
    #domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, -R]), np.array([length, R])], 
    #                               [nx, ny], mesh.CellType.triangle)
    
    # create function spaces for velocity (P2) and pressure (P1)
    P2 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
    V, Q = FunctionSpace(msh, P2), FunctionSpace(msh, P1)
    
    # assigning boundary conditions.
    # component-wise bcs https://jsdokken.com/dolfinx-tutorial/chapter3/component_bc.html

    # no slip bc
    noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)

    bottom_dofs   = locate_dofs_topological(V, msh.topology.dim-1, bottom_facets)
    bottom_y_dofs = locate_dofs_topological(V.sub(1), msh.topology.dim-1, bottom_facets)
    top_x_dofs    = locate_dofs_topological(V.sub(0), msh.topology.dim-1, top_facets)
    top_y_dofs    = locate_dofs_topological(V.sub(1), msh.topology.dim-1, top_facets)
    left_x_dofs   = locate_dofs_topological(V.sub(0), msh.topology.dim-1, left_facets)
    right_x_dofs  = locate_dofs_topological(V.sub(0), msh.topology.dim-1, right_facets)
    left_y_dofs   = locate_dofs_topological(V.sub(1), msh.topology.dim-1, left_facets)
    right_y_dofs  = locate_dofs_topological(V.sub(1), msh.topology.dim-1, right_facets)

    # Lid velocity
    def lid_velocity_expression(x):
        return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))
    # Driving (lid) velocity condition on top boundary (y = 1)
    lid_velocity = Function(V)
    lid_velocity.interpolate(lid_velocity_expression)
    #facets = locate_entities_boundary(msh, 1, lid)
    bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, 1, top_facets))

    # Collect Dirichlet boundary conditions
    if bcs_type == 1:
        bcs = [dirichletbc(PETSc.ScalarType(0), bottom_y_dofs, V.sub(1)),
               dirichletbc(PETSc.ScalarType(0), top_y_dofs,    V.sub(1)),
               dirichletbc(PETSc.ScalarType(0), left_x_dofs,   V.sub(0)),
               dirichletbc(PETSc.ScalarType(0), right_x_dofs,  V.sub(0))]
    elif bcs_type == 2:
        bcs = [dirichletbc(PETSc.ScalarType(0), bottom_y_dofs, V.sub(1)),
               dirichletbc(PETSc.ScalarType(0), left_x_dofs,   V.sub(0)),
               dirichletbc(PETSc.ScalarType(0), right_x_dofs,  V.sub(0))]
    elif bcs_type == 3:
        bcs = [dirichletbc(noslip, bottom_dofs, V),
               dirichletbc(PETSc.ScalarType(1), top_x_dofs,    V.sub(0)),
               dirichletbc(PETSc.ScalarType(0), top_y_dofs,    V.sub(1)),
               dirichletbc(PETSc.ScalarType(0), left_y_dofs,   V.sub(1)),
               dirichletbc(PETSc.ScalarType(0), right_y_dofs,  V.sub(1))]
    
    # Define variational problem
    (u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    (v, q) = ufl.TestFunction(V),  ufl.TestFunction(Q)
    #f      = Constant(msh, (PETSc.ScalarType(0), dg))

    # define function grav_vel to calculate the inner product of body force inner(f,v) = inner([dgx,dgy],[v[0],v[1]])
    def grav_vel(dgx, dgy, v):
        return dgx*v[0] + dgy*v[1]

    # the weak formulation in UFL syntax.
    a     = form([[inner(grad(u), eta * grad(v)) * dx, inner(p, div(v)) * dx],
              [inner(div(u), q) * dx, None]])
    #L = form([inner(f, v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])
    L     = form([grav_vel(dgx,dgy,v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])
    a_p11 = form(inner(p, q) * dx)
    a_p   = [[a[0][0], None],
             [None, a_p11]]
    
    return a, a_p, L, bcs, V, Q, msh

def nested_iterative_solver():
    """Solve the Stokes problem using nest matrices and an iterative solver."""

    # Assemble nested matrix operators
    A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
    A.assemble()

    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P11 = fem.petsc.assemble_matrix(a_p11, [])
    P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
    P.assemble()

    # Assemble right-hand side vector
    b = fem.petsc.assemble_vector_nest(L)

    # Modify ('lift') the RHS for Dirichlet boundary conditions
    fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

    # Sum contributions for vector entries that are share across
    # parallel processes
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
    fem.petsc.set_bc_nest(b, bcs0)

    # The pressure field is determined only up to a constant. We supply
    # a vector that spans the nullspace to the solver, and any component
    # of the solution in this direction will be eliminated during the
    # solution process.
    null_vec = fem.petsc.create_vector_nest(L)

    # Set velocity part to zero and the pressure part to a non-zero
    # constant
    null_vecs = null_vec.getNestSubVecs()
    null_vecs[0].set(0.0), null_vecs[1].set(1.0)

    # Normalize the vector that spans the nullspace, create a nullspace
    # object, and attach it to the matrix
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    #assert nsp.test(A)
    A.setNullSpace(nsp)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # Define the matrix blocks in the preconditioner with the velocity
    # and pressure matrix index sets
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block. For the top-left
    # Laplace-type operator we use algebraic multigrid. For the
    # lower-right block we use a Jacobi preconditioner. By default, GAMG
    # will infer the correct near-nullspace from the matrix block size.
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Create finite element {py:class}`Function <dolfinx.fem.Function>`s
    # for the velocity (on the space `V`) and for the pressure (on the
    # space `Q`). The vectors for `u` and `p` are combined to form a
    # nested vector and the system is solved.
    u, p = Function(V), Function(Q)
    x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(u.x),
                                la.create_petsc_vector_wrap(p.x)])
    ksp.solve(b, x)

    # Save solution to file in XDMF format for visualization, e.g. with
    # ParaView. Before writing to file, ghost values are updated using
    # `scatter_forward`.
    with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as ufile_xdmf:
        u.x.scatter_forward()
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u)

    with XDMFFile(MPI.COMM_WORLD, "pressure.xdmf", "w") as pfile_xdmf:
        p.x.scatter_forward()
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p)

    # Compute norms of the solution vectors
    norm_u = u.x.norm()
    norm_p = p.x.norm()
    if MPI.COMM_WORLD.rank == 0:
        print(f"(A) Norm of velocity coefficient vector (blocked, iterative): {norm_u}")
        print(f"(A) Norm of pressure coefficient vector (blocked, iterative): {norm_p}")

    return norm_u, norm_p

def block_operators(a, a_p, L, bcs, V, Q):
    """Return block operators and block RHS vector for the Stokes
    problem"""

    # Assembler matrix operator, preconditioner and RHS vector into
    # single objects but preserving block structure
    A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
    A.assemble()
    P = fem.petsc.assemble_matrix_block(a_p, bcs=bcs)
    P.assemble()
    b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

    # Set the nullspace for pressure (since pressure is determined only
    # up to a constant)
    null_vec = A.createVecLeft()
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    null_vec.array[offset:] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    #assert nsp.test(A)
    A.setNullSpace(nsp)

    return A, P, b

def block_iterative_solver(a, a_p, L, bcs, V, Q, msh):
    """Solve the Stokes problem using blocked matrices and an iterative
    solver."""

    # Assembler the operators and RHS vector
    A, P, b = block_operators(a, a_p, L, bcs, V, Q)

    # Build PETSc index sets for each field (global dof indices for each
    # field)
    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
    offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
    is_u = PETSc.IS().createStride(V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF)
    is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setTolerances(rtol=1e-9)
    ksp.setType("minres")
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
    ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    # Configure velocity and pressure sub-solvers
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # The matrix A combined the vector velocity and scalar pressure
    # parts, hence has a block size of 1. Unlike the MatNest case, GAMG
    # cannot infer the correct near-nullspace from the matrix block
    # size. Therefore, we set block size on the top-left block of the
    # preconditioner so that GAMG can infer the appropriate near
    # nullspace.
    ksp.getPC().setUp()
    Pu, _ = ksp_u.getPC().getOperators()
    Pu.setBlockSize(msh.topology.dim)

    # Create a block vector (x) to store the full solution and solve
    x = A.createVecRight()
    ksp.solve(b, x)

    # Create Functions to split u and p
    u, p = Function(V), Function(Q)
    u.name = "Velocity" 
    # https://discourse.paraview.org/t/paraview-crashes-when-opening-xdmf-files-created-with-fenicsx/9453/4
    offset = V_map.size_local * V.dofmap.index_map_bs
    u.x.array[:offset] = x.array_r[:offset]
    p.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]

    # Save solution to file in XDMF format for visualization, e.g. with
    # ParaView. Before writing to file, ghost values are updated using
    # `scatter_forward`.
    with XDMFFile(MPI.COMM_WORLD, "velocity.xdmf", "w") as ufile_xdmf:
        #ufile_xdmf.parameters["flush_output"] = True
        u.x.scatter_forward()
        ufile_xdmf.write_mesh(msh)     
        ufile_xdmf.write_function(u)
        
    with XDMFFile(MPI.COMM_WORLD, "pressure.xdmf", "w") as pfile_xdmf:
        p.x.scatter_forward()
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p)
        
    # Compute the $L^2$ norms of the solution vectors
    norm_u, norm_p = u.x.norm(), p.x.norm()
    if MPI.COMM_WORLD.rank == 0:
        print(f"(B) Norm of velocity coefficient vector (blocked, iterative): {norm_u}")
        print(f"(B) Norm of pressure coefficient vector (blocked, iterative): {norm_p}")

    return norm_u, norm_p, u, p