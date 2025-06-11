import asQ
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from utils.timing import SolverTimer
import os
from argparse import ArgumentParser

parser = ArgumentParser(description='ParaDiag timestepping for Stratigraphic Model.'
                       epilog="""\
Optional PETSc command line arguments: circulant_alpha :float: The circulant parameter to use in the preconditioner. Default 1e-4.)

parser.add_argument('--nx', type=int, default=400, help='Number of cells along each side of the square for the coarsest mesh.')
parser.add_argument('--dt', type=float, default=4*1e-5, help='Degree of the scalar space.')
parser.add_argument('--nwindows', type=int, default=10, help='Number of time-windows to solve.')
parser.add_argument('--a' , type=float, default=0.1, help='Residual Added Diffusion coefficient.')
parser.add_argument('--beta', type=float, default=5, help='Residual Added Diffusion coefficient for h')
parser.add_argument('--nslices', type=int, default=4, help='Number of time-slices in the all-at-once system. Must divide the number of MPI ranks exactly.')
parser.add_argument('--slice_length', type=int, default=1, help='Number of timesteps per time-slice. Total number of timesteps in the all-at-once system is nslices*slice_length.')
parser.add_argument('--show_args', action='store_true', help='Print all the arguments when the script starts.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

time_partition = tuple(args.slice_length for _ in range(args.nslices))
ensemble = asQ.create_ensemble(time_partition)
window_length = sum(time_partition)
nwindows = args.nwindows
nsteps = nwindows*window_length

dt = args.dt
Dt = Constant(dt)

n = args.nx
mesh = UnitSquareMesh(n, n, comm=ensemble.comm)
n_ref = 4
hierarchy = MeshHierarchy(mesh, n_ref)
mesh = hierarchy[-1]
n *= 2**n_ref
h_const = Constant(1/n)

V = FunctionSpace(mesh, "CG", 2)
W = MixedFunctionSpace((V, V))

w0 = Function(W)

s0, u0 = w0.subfunctions

x, y= SpatialCoordinate(mesh)

u0.interpolate(0)
s0.interpolate(0)

b = Function(V)
a = Constant(args.a)
beta = Constant(args.beta)

def D(d): 
  return (0.2/sqrt(2*pi))*exp(-((d-5)/10)**2/2)

def l(d):
  r = conditional(d>0, exp(-d/10)/(1 + exp(-50*d)), exp(50*d - d/10)/(exp(50*d) + 1))
  return 2000*r

def D_tilde(d, u, s, h):
   return sqrt((D(d)**2) + a*(h**beta)*(((s - div(D(d)*grad(u)) - l(d))**2)))


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

block_parameter = {"ksp_type": "gmres",                                                        
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
    'ksp_type': 'fgmres',
    'ksp': {
        'rtol': 1e-3,
        'max_it': 30,
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

parameter_paradiag = {                                                                  
    #"snes_monitor": None,                                                       
    "snes_converged_reason": None,                                              
    "snes_atol": 1e-50,                                                         
    "snes_stol": 1e-50,                                                         
    #"ksp_monitor": None,                                                      
    "ksp_converged_reason": None,                                                 
    "ksp_type": "fgmres",                                                        
    "ksp_rtol": 1e-6,                                                           
    "ksp_max_it": 40,                                                           
    "pc_type": "python",
    "pc_python_type":'asQ.CirculantPC',
    'circulant_alpha': 1e-4,
    'circulant_block': block_parameter               
}


paradiag = asQ.Paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass,
                   ics=w0, dt=dt, theta=0.5,
                   time_partition=time_partition,
                   solver_parameters=parameter_paradiag)


#ufile = VTKFile('u_pdg.pvd')

# This function will be called before paradiag solves each time-window. We can use
# this to make the output a bit easier to read, and to time the window calculation
def window_preproc(paradiag, wndw, rhs):
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')
    # for now we are interested in timing only the solve, this
    # makes sure we don't time any synchronisation after prints.
    with PETSc.Log.Event("window_preproc.Coll_Barrier"):
        paradiag.ensemble.ensemble_comm.Barrier()
    timer.start_timing()


# This function will be called after paradiag solves each time-window. We can use
# this to finish the window calculation timing and print the result.
def window_postproc(paradiag, wndw, rhs):
    timer.stop_timing()
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'Window solution time: {round(timer.times[-1], 5)}')
    PETSc.Sys.Print('')


# Setup all solver objects. The firedrake DM and options management
# makes it difficult to setup some preconditioners without actually
# calling `solve`, so we just run once to set everything up.
PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up solver and prefactoring --- === ###')
PETSc.Sys.Print('')
with PETSc.Log.Event("warmup_solve"):
    paradiag.solve(1)
PETSc.Sys.Print('')

# reset solution and iteration counts for timed solved
paradiag.reset_diagnostics()
aaofunc = paradiag.solver.aaofunc
aaofunc.bcast_field(-1, aaofunc.initial_condition)
aaofunc.assign(aaofunc.initial_condition)

PETSc.Sys.Print('### === --- Solving timeseries --- === ###')
PETSc.Sys.Print('')

# Solve nwindows of the all-at-once system
with PETSc.Log.Event("timed_solves"):
    paradiag.solve(nwindows,
                   preproc=window_preproc,
                   postproc=window_postproc)
    

PETSc.Sys.Print('### === --- Iteration and timing results --- === ###')
PETSc.Sys.Print('')

# Write out the convergence statistics to a file
os.makedirs('2d_paradiag_convergestats/meshsize{n}'.format(n=n), exist_ok=True)
asQ.write_paradiag_metrics(paradiag, directory='2d_paradiag_convergestats/meshsize{n}'.format(n=n))

nw = paradiag.total_windows
nt = paradiag.total_timesteps
PETSc.Sys.Print(f'Total windows: {nw}')
PETSc.Sys.Print(f'Total timesteps: {nt}')
PETSc.Sys.Print('')


# paradiag collects a few iteration counts for us
lits = paradiag.linear_iterations
nlits = paradiag.nonlinear_iterations
blits = paradiag.block_iterations.data()

# Number of nonlinear iterations will be 1 per window for linear problems
PETSc.Sys.Print(f'Nonlinear iterations: {str(nlits).rjust(5)}  |  Iterations per window: {str(nlits/nw).rjust(5)}')

# Number of linear iterations of the all-at-once system, total and per window.
PETSc.Sys.Print(f'Linear iterations:    {str(lits).rjust(5)}  |  Iterations per window: {str(lits/nw).rjust(5)}')

# Number of iterations needed for each block in step-(b), total and per block solve
PETSc.Sys.Print(f'Total block linear iterations: {blits}')
PETSc.Sys.Print(f'Iterations per block solve: {blits/lits}')
TSc.Sys.Print(f'Maximum Iterations per timestep: {max(blits)/nt}')
PETSc.Sys.Print(f'Minimum block iterations per solve: {min(blits)/lits}')
PETSc.Sys.Print(f'Maximum block iterations per solve (kp): {max(blits)/lits}')
PETSc.Sys.Print('')

# Timing measurements
PETSc.Sys.Print(timer.string(timesteps_per_solve=window_length,
                             total_iterations=paradiag.linear_iterations, ndigits=5))
PETSc.Sys.Print('')
