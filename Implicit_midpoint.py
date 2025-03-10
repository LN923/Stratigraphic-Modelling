from firedrake import *
import matplotlib.pyplot as plt

# Create mesh and define function space

dt = 4*1e-5
Dt = Constant(dt)

n = 100
mesh = IntervalMesh(n, 1)
h_const = Constant(1/n)

V = FunctionSpace(mesh, "CG", 2)
W = MixedFunctionSpace((V, V))

w0 = Function(W)
s0, u0 = w0.subfunctions

x, = SpatialCoordinate(mesh)

u0.interpolate(0)
s0.interpolate(0)

b = Function(V)

def D(d): 
  return (0.2/sqrt(2*pi))*exp(-((d-5)/10)**2/2)

def l(d):
  r = conditional(d>0, exp(-d/10)/(1 + exp(-50*d)), exp(50*d - d/10)/(exp(50*d) + 1))
  return 2000*r

def D_tilde(d, u, s, h):
   return sqrt((D(d)**2) + 0.1*(h**5)*(((s - div(D(d)*grad(u)) - l(d))**2)))


p, q = TestFunctions(W)
w1 = Function(W)
w1.assign(w0)
s1, u1 = split(w1)
s0, u0 = split(w0)

sh = 0.5*(s1 + s0)
uh = 0.5*(u1 + u0)
b.interpolate(100*tanh(5*x-5/2))
time = Constant(0.0)

#L = (
#(q*(u1 - u0) + Dt*(D(sin(2*pi*time) - b - uh)*uh.dx(0)*q.dx(0) - q*l(sin(2*pi*time) - b - uh)))*dx +
##(p*(u1 - u0) - Dt*sh*p)*dx
#)

L = (
(q*(u1 - u0) + Dt*(D_tilde(sin(2*pi*time) - b - uh, uh, sh, h_const)*uh.dx(0)*q.dx(0) - q*l(sin(2*pi*time) - b - uh)))*dx +
(p*(u1 - u0) - Dt*sh*p)*dx
)
parameter_safe ={'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'}

parameter = {'mat_type': 'aij',
    'ksp_type': 'gmres',
    'ksp_atol': 1e-50,
    'ksp_rtol': 1e-6,
    'snes_stol': 1e-50,
    'snes_atol' :1e-6,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type':'additive',
    'fieldsplit_0_pc_type' :'lu',
    'fieldsplit_1_pc_type' :'lu',
    'ksp_converged_reason': None,
    'snes_converged_reason': None,}

parameter_patch = {                                                                  
    #"snes_monitor": None,                                                       
    #"snes_converged_reason": None,                                              
    "snes_atol": 1e-50,                                                         
    "snes_stol": 1e-50,                                                         
    #"ksp_monitor": None,                                                        
    #"ksp_converged_rate": None,                                                 
    "ksp_type": "gmres",                                                        
    "ksp_rtol": 1e-6,                                                           
    "ksp_max_it": 40,                                                           
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",                                      
    "patch_pc_patch_save_operators": True,                                      
    "patch_pc_patch_partition_of_unity": True,                                  
    "patch_pc_patch_sub_mat_type": "seqdense",                                  
    "patch_pc_patch_construct_dim": 0,                                          
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_local_type": "additive",                                    
    "patch_pc_patch_precompute_element_tensors": True,                          
    "patch_pc_patch_symmetrise_sweep": False,                                   
    "patch_sub_ksp_type": "preonly",                                            
    "patch_sub_pc_type": "lu",                                                  
    "patch_sub_pc_factor_shift_type": "nonzero"                                 
}
uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters= parameter_patch
   )

# Next we use the other form of :attr:`~.Function.subfunctions`, ``w0.subfunctions``,
# which is the way to split up a Function in order to access its data
# e.g. for output. ::

m0, u0 = w0.subfunctions
m1, u1 = w1.subfunctions

# We choose a final time, and initialise a :class:`~.vtk_output.VTKFile`
# object for storing ``u``. as well as an array for storing the function
# to be visualised::

T = 1.0
ufile = VTKFile('u_dewiggled_15alpha.pvd')
t =0.0
ufile.write(u1, time=t)
all_us = []

# We also initialise a dump counter so we only dump every 10 timesteps. ::
ndump = 1
dumpn = 0

# Now we enter the timeloop. ::
nsteps = 100
dt = T/nsteps
Dt.assign(dt)
for step in ProgressBar("timestep").iter(range(nsteps)):
   time.assign(t + 0.5*dt)
   t += dt

# The energy can be computed and checked. ::


#
   usolver.solve()
   w0.assign(w1)

# Finally, we check if it is time to dump the data. The function will be appended
# to the array of functions to be plotted later::


#
   dumpn += 1
   if dumpn == ndump:
      dumpn -= ndump
      ufile.write(u1, time=t)



