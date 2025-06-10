import asQ
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from utils.timing import SolverTimer
from utils.serial import SerialMiniApp
import os
from argparse import ArgumentParser
from utils import diagnostics
# Create mesh and define function space

parser = ArgumentParser(
    description='ParaDiag timestepping 2D Stratigraphic Model.',
    epilog="""\
Optional PETSc command line arguments:

   -circulant_alpha :float: The circulant parameter to use in the preconditioner. Default 1e-4'"""
)

parser.add_argument('--nx', type=int, default=8, help='Number of cells along each side of the square for the coarsest mesh in multigrid.')
parser.add_argument('--a' , type=float, default=0.1, help='Residual Added Diffusion coefficient.')
parser.add_argument('--beta', type=float, default=5, help='Residual Added Diffusion coefficient for h')
parser.add_argument('--dt', type=float, default=4*1e-5, help='Degree of the scalar space.')
parser.add_argument('--nt', type=int, default=16000, help='Number of timesteps to solve.')
parser.add_argument('--show_args', action='store_true', help='Print all the arguments when the script starts.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

dt = args.dt
Dt = Constant(dt)

n = 8
mesh = UnitSquareMesh(n, n, comm=ensemble.comm)
n_ref = 4
hierarchy = MeshHierarchy(mesh, n_ref)
mesh = hierarchy[-1]
n *= 2**n_ref
h_const = Constant(1/n)

V = FunctionSpace(mesh, "CG", 2)
W = MixedFunctionSpace((V, V))

w0 = Function(W)
u0, s0 = w0.subfunctions

x, y = SpatialCoordinate(mesh)

u0.interpolate(0)
s0.interpolate(0)

b = Function(V)

def D(d): 
  return (0.2/sqrt(2*pi))*exp(-((d-5)/10)**2/2)

def l(d):
  r = conditional(d>0, exp(-d/10)/(1 + exp(-50*d)), exp(50*d - d/10)/(exp(50*d) + 1))
  return 2000*r

def D_tilde(d, u, s, h):
   return sqrt((D(d)**2) + args.a*(h**args.beta)*(((s - div(D(d)*grad(u)) - l(d))**2)))


p, q = TestFunctions(W)
w1 = Function(W)
w1.assign(w0)


b.interpolate(100*tanh(5*x-5/2))


def form_mass(u, s, q, p):
    return q*u*dx + p*s*dx

def form_function(u, s, q, p, t):
   return (D_tilde(sin(2*pi*t) - b - u, u, s, h_const)*inner(grad(u),grad(q)) - q*l(sin(2*pi*t) - b - u))*dx(degree = 2) - s*p*dx


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

block_parameter = {                                                      
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
    "patch_sub_pc_factor_shift_type": "nonzero"}

block_parameters = {
    'mat_type': 'matfree',
    'ksp_type': 'gmres',
    'ksp': {
        'rtol': 1e-6,
        'max_it': 40,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'levels': {
            'ksp_type': 'richardson',
            'ksp_richardson_scale': 0.9,
            'ksp_max_it': 3,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
                'pc_patch': {
                    'save_operators': True,
                    'partition_of_unity': True,
                    'sub_mat_type': 'seqdense',
                    'construct_dim': 0,
                    'construct_type': 'star',
                    'local_type': 'additive',
                    'precompute_element_tensors': True,
                    'symmetrise_sweep': False
                },
                'sub': {
                    'ksp_type': 'preonly',
                    'pc_type': 'lu',
                    'pc_factor_shift_type': 'nonzero',
                }
            }
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled': {'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
            },
        },
    }
}

parameter_serial = {                                                                  
    #"snes_monitor": None,                                                       
    "snes_converged_reason": None,                                              
    "snes_atol": 1e-50,                                                         
    "snes_stol": 1e-50,
    "snes_ksp_ew": None,  
    "snes_linesearch_type": "basic",
    "snes_rtolmax": 0.7,                                           
    #"ksp_monitor": None,                                                      
    "ksp_converged_reason": None,    
    "ksp_type": "fgmres",                                                        
    "ksp_rtol": 1e-6,                                                         
    "ksp_max_it": 40                                                            
}
parameter_serial.update(block_parameters)

# Create a file to write the solution to
#try:
    #ufile = VTKFile('1d_paradiag_u200.pvd', ensemble.comm)
#except FileExistsError:
    # the dir already exists
    # put code handing this case here
    #pass

miniapp = SerialMiniApp(dt, 0.5, w0,
                        form_mass, form_function,
                        parameter_serial)


# create a timer to profile the calculations
timer = SolverTimer()

PETSc.Sys.Print('### === --- Timestepping loop --- === ###')
PETSc.Sys.Print('')
linear_its = 0
nonlinear_its = 0


# This function will be called before solving each timestep. We can use
# this to make the output a bit easier to read, and to time the calculation
def preproc(app, step, t):
    PETSc.Sys.Print(f'### === --- Calculating timestep {step} --- === ###')
    PETSc.Sys.Print('')
    # for now we are interested in timing only the solve, this
    # makes sure we don't time any synchronisation after prints.
    with PETSc.Log.Event("timestep_preproc.Coll_Barrier"):
        mesh.comm.Barrier()
    timer.start_timing()


# This function will be called after solving each timestep. We can use
# this to finish the timestep calculation timing and print the result,
# and to record the number of iterations.
def postproc(app, step, t):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Timestep solution time: {round(timer.times[-1], 5)}')
    PETSc.Sys.Print('')

    global linear_its
    global nonlinear_its
    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()


# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')
PETSc.Sys.Print('')
with PETSc.Log.Event("warmup_solve"):
    miniapp.solve(1)
PETSc.Sys.Print('')

# reset solution
miniapp.w0.assign(w0)
miniapp.w1.assign(w0)

PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

# Solve nt timesteps
with PETSc.Log.Event("timed_solves"):
    miniapp.solve(args.nt,
                  preproc=preproc,
                  postproc=postproc)

# # # === --- Solver diagnostics --- === # # #

PETSc.Sys.Print('### === --- Iteration and timing results --- === ###')
PETSc.Sys.Print('')

# parallelism
PETSc.Sys.Print(f'DoFs per timestep: {V.dim()}')
PETSc.Sys.Print(f'Number of MPI ranks per timestep: {mesh.comm.size}')
PETSc.Sys.Print(f'DoFs/rank: {V.dim()/mesh.comm.size}')
PETSc.Sys.Print('')

# Number of nonlinear iterations will be 1 per timestep for linear problems
PETSc.Sys.Print(f'Nonlinear iterations: {str(nonlinear_its).rjust(5)}  |  Iterations per window: {str(nonlinear_its/args.nt).rjust(5)}')

# Number of linear iterations of the all-at-once system, total and per window.
PETSc.Sys.Print(f'Linear iterations:    {str(linear_its).rjust(5)}  |  Iterations per window: {str(linear_its/args.nt).rjust(5)}')
PETSc.Sys.Print('')

# Timing measurements
PETSc.Sys.Print(timer.string(timesteps_per_solve=1,
                             total_iterations=linear_its, ndigits=5))
PETSc.Sys.Print('')
