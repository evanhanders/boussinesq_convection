"""
Dedalus script for Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

Usage:
    rayleigh_benard.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e4]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nz=<nz>                  Vertical resolution [default: 32]
    --nx=<nx>                  Horizontal resolution [default: 64]
    --ny=<nx>                  Horizontal resolution [default: 64]
    --aspect=<aspect>          Aspect ratio of problem [default: 2]

    --fixed_f                  Fixed flux boundary conditions top/bottom
    --fixed_t                  Fixed temperature boundary conditions top/bottom
    --stress_free              Stress free boundary conditions top/bottom

    --3D                       Run in 3D
    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --run_time_wall=<time>     Run time, in hours [default: 23.5]
    --run_time_buoy=<time>     Run time, in buoyancy times
    --run_time_therm=<time_>   Run time, in thermal times [default: 1]

    --restart=<restart_file>   Restart from checkpoint
    --overwrite                If flagged, force file mode to overwrite
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --label=<label>            Optional additional case name label
    --verbose                  Do verbose output (e.g., sparsity patterns of arrays)
    --output_dt=<num>          Simulation time between outputs [default: 0.2]
    --no_coeffs                If flagged, coeffs will not be output   
    --no_volumes               If flagged, volumes will not be output   
    --no_join                  If flagged, don't join files at end of run
    --root_dir=<dir>           Root directory for output [default: ./]

    --stat_wait_time=<t>       Time to wait before averaging Nu, T [default: 0]
    --stat_window=<t_w>        Time to take Nu, T averages over [default: 100]

    --ae                       Do accelerated evolution

"""
import sys
import logging
logger = logging.getLogger(__name__)

import numpy as np
from mpi4py import MPI
import time

from docopt import docopt
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

from logic.output import initialize_output
from logic.checkpointing import Checkpoint
from logic.ae_tools import BoussinesqAESolver

args = docopt(__doc__)

checkpoint_min = 30
RA_CRIT = 1295.78


def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.

    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)


def global_noise(domain, seed=42, **kwargs):
    """
    Create a field fielled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)
    return noise_field


fixed_f = args['--fixed_f']
fixed_t = args['--fixed_t']
if not (fixed_f or fixed_t):
    mixed_BCs = True

stress_free = args['--stress_free']
if not stress_free:
    no_slip = True

# save data in directory named after script
data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]

threeD = args['--3D']
if threeD:
    data_dir += '_3D'
else:
    data_dir += '_2D'

if fixed_f:
    data_dir += '_fixedF'
elif fixed_t:
    data_dir += '_fixedT'
else:
    data_dir += '_mixedFT'

if stress_free:
    data_dir += '_stressFree'
else:
    data_dir += '_noSlip'

ra = float(args['--Rayleigh'])
pr = float(args['--Prandtl'])
aspect = float(args['--aspect'])

data_dir += "_Ra{}_Pr{}_a{}".format(args['--Rayleigh'], args['--Prandtl'], args['--aspect'])
if args['--label'] is not None:
    data_dir += "_{}".format(args['--label'])
data_dir += '/'
logger.info("saving run in: {}".format(data_dir))



run_time_buoy = args['--run_time_buoy']
run_time_therm = args['--run_time_therm']
run_time_wall = float(args['--run_time_wall'])
if run_time_buoy is not None:
    run_time_buoy = float(run_time_buoy)
if run_time_therm is not None:
    run_time_therm = float(run_time_therm)

mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]

    
import os
from dedalus.tools.config import config

config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'

import mpi4py.MPI
if mpi4py.MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.makedirs('{:s}/'.format(data_dir))
    logdir = os.path.join(data_dir,'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
logger = logging.getLogger(__name__)
logger.info("saving run in: {}".format(data_dir))

import time
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

# input parameters
logger.info("Ra = {}, Pr = {}".format(ra, pr))


# Parameters
nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])

P = (ra*pr)**(-1./2)
R = (ra/pr)**(-1./2)


x_basis = de.Fourier( 'x', nx, interval = [-aspect/2, aspect/2], dealias=3/2)
if threeD : x_basis = de.Fourier( 'y', ny, interval = [-aspect/2, aspect/2], dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval = [-1./2, 1./2], dealias=3/2)

if threeD:  bases = [x_basis, y_basis, z_basis]
else:       bases = [x_basis, z_basis]

domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)


variables = ['T1', 'T1_z', 'p', 'u', 'v', 'w', 'Ox', 'Oy', 'Oz']
if not threeD:
    variables.remove('v')
    variables.remove('Ox')
    variables.remove('Oz')
problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

problem.parameters['P'] = P
problem.parameters['R'] = R
problem.parameters['Lx'] = problem.parameters['Ly'] = aspect
problem.parameters['Lz'] = 1
problem.substitutions['T0']   = '-z'
problem.substitutions['T0_z'] = '-1'
problem.substitutions['Lap(A, A_z)']=       '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
problem.substitutions['UdotGrad(A, A_z)'] = '(u*dx(A) + v*dy(A) + w*A_z)'

if not threeD:
    problem.substitutions['dy(A)'] = '0'
    problem.substitutions['Ox'] = '0'
    problem.substitutions['Oz'] = '0'
    problem.substitutions['v'] = '0'
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
else:
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'
problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
problem.substitutions['enstrophy'] = '(Ox**2 + Oy**2 + Oz**2)'

problem.substitutions['enth_flux'] = '(w*(T1+T0))'
problem.substitutions['cond_flux'] = '(-P*(T1_z+T0_z))'
problem.substitutions['tot_flux'] = '(cond_flux+enth_flux)'
problem.substitutions['momentum_rhs_z'] = '(u*Oy - v*Ox)'
problem.substitutions['Nu'] = '((enth_flux + cond_flux)/vol_avg(cond_flux))'
problem.substitutions['delta_T1'] = '(left(T1)-right(T1))'
problem.substitutions['vel_rms'] = 'sqrt(u**2 + v**2 + w**2)'

problem.substitutions['Re'] = '(vel_rms / R)'
problem.substitutions['Pe'] = '(vel_rms / P)'



problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z           = -UdotGrad(T1, T1_z)")
problem.add_equation("dt(u)  + R*(dy(Oz) - dz(Oy))  + dx(p)       =  v*Oz - w*Oy ")
if threeD: problem.add_equation("dt(v)  + R*(dz(Ox) - dx(Oz))  + dy(p)       =  w*Ox - u*Oz ")
problem.add_equation("dt(w)  + R*(dx(Oy) - dy(Ox))  + dz(p) - T1  =  u*Oy - v*Ox ")
problem.add_equation("T1_z - dz(T1) = 0")
if threeD: problem.add_equation("Ox - dy(w) + dz(v) = 0")
problem.add_equation("Oy - dz(u) + dx(w) = 0")
if threeD: problem.add_equation("Oz - dx(v) + dy(u) = 0")


if args['--fixed_f']:
    logger.info("Thermal BC: fixed flux (full form)")
    problem.add_bc( "left(T1_z) = 0")
    problem.add_bc("right(T1_z) = 0")
    dirichlet_set.append('T1_z')
elif args['--fixed_t']:
    logger.info("Thermal BC: fixed temperature (T1)")
    problem.add_bc( "left(T1) = 0")
    problem.add_bc("right(T1) = 0")
    dirichlet_set.append('T1')
else:
    logger.info("Thermal BC: fixed flux/fixed temperature")
    problem.add_bc("left(T1_z) = 0")
    problem.add_bc("right(T1)  = 0")

if args['--stress_free']:
    logger.info("Horizontal velocity BC: stress free")
    problem.add_bc("left(Oy) = 0")
    problem.add_bc("right(Oy) = 0")
    if threeD:
        problem.add_bc("left(Ox) = 0")
        problem.add_bc("right(Ox) = 0")
else:
    logger.info("Horizontal velocity BC: no slip")
    problem.add_bc( "left(u) = 0")
    problem.add_bc("right(u) = 0")
    if threeD:
        problem.add_bc("left(v) = 0")
        problem.add_bc("right(v) = 0")

# vertical velocity boundary conditions
logger.info("Vertical velocity BC: impenetrable")
problem.add_bc( "left(w) = 0")
if threeD:
    problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_bc("right(w) = 0", condition="(nx != 0) or  (ny != 0)")
else:
    problem.add_bc("right(p) = 0", condition="(nx == 0)")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")

# Build solver
ts = de.timesteppers.RK222
cfl_safety = 0.5
#cfl_safety = 0.8

solver = problem.build_solver(ts)
logger.info('Solver built')

checkpoint = Checkpoint(data_dir)
restart = args['--restart']
if isinstance(restart, type(None)):
    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    T1.set_scales(domain.dealias)
    noise = global_noise(domain, int(args['--seed']))
    z_de = domain.grid(-1, scales=domain.dealias)
    T1['g'] = 1e-6*P*np.sin(np.pi*z_de)*noise['g']*(-z_de)
    T1.differentiate('z', out=T1_z)

    dt = None
    mode = 'overwrite'
else:
    logger.info("restarting from {}".format(restart))
    checkpoint.restart(restart, solver)
    if overwrite:
        mode = 'overwrite'
    else:
        mode = 'append'
checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
    
# Integration parameters
if run_time_buoy is not None:    solver.stop_sim_time = run_time_buoy
elif run_time_therm is not None: solver.stop_sim_time = run_time_therm/P
else:                            solver.stop_sim_time = 1/P
solver.stop_wall_time = run_time_wall*3600.
Hermitian_cadence = 100

# Analysis
max_dt    = 0.1
analysis_tasks = initialize_output(solver, data_dir, aspect, threeD=threeD)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
if threeD:
    CFL.add_velocities(('u', 'v', 'w'))
else:
    CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re", name='Re')
flow.add_property("Nu", name='Nu')
flow.add_property("T0+T1", name='T')

rank = domain.dist.comm_cart.rank
if rank == 0:
    nu_vals    = np.zeros(5*int(args['--stat_window']))
    temp_vals  = np.zeros(5*int(args['--stat_window']))
    dt_vals    = np.zeros(5*int(args['--stat_window']))
    writes     = 0


if args['--ae']:
    kwargs = { 'first_ae_wait_time' : 0,
               'first_ae_avg_time' : 2,
               'first_ae_avg_thresh' : 1e0 }
    ae_solver = BoussinesqAESolver(nz, solver, domain.dist, ['tot_flux', 'enth_flux', 'momentum_rhs_z'], ['T1', 'p', 'delta_T1'], P, R,
                **kwargs)

first_step = True
# Main loop
try:
    count = 0
    logger.info('Starting loop')
    Re_avg = 0
    not_corrected_times = True
    init_time = last_time = solver.sim_time
    start_iter = solver.iteration
    while (solver.ok and np.isfinite(Re_avg)):
        dt = CFL.compute_dt()
        solver.step(dt) #, trim=True)


        # Solve for blow-up over long timescales in 3D due to hermitian-ness
        effective_iter = solver.iteration - start_iter
        if threeD and effective_iter % Hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()
        
        if Re_avg > 1:
            if not_corrected_times:
                if run_time_buoy is not None:
                    solver.stop_sim_time  = run_time_buoy + solver.sim_time
                elif run_time_therm is not None:
                    solver.stop_sim_time = run_time_therm/P + solver.sim_time
                not_corrected_times = False
            
            if last_time == init_time:
                last_time = solver.sim_time + float(args['--stat_wait_time'])
            if solver.sim_time - last_time >= 0.2:
                avg_Nu, avg_T = flow.grid_average('Nu'), flow.grid_average('T')
                if domain.dist.comm_cart.rank == 0:
                    if writes != dt_vals.shape[0]:
                        dt_vals[writes] = solver.sim_time - last_time
                        nu_vals[writes] = avg_Nu
                        temp_vals[writes] = avg_T
                        writes += 1
                    else:
                        dt_vals[:-1] = dt_vals[1:]
                        nu_vals[:-1] = nu_vals[1:]
                        temp_vals[:-1] = temp_vals[1:]
                        dt_vals[-1] = solver.sim_time - last_time
                        nu_vals[-1] = avg_Nu
                        temp_vals[-1] = avg_T

                    avg_nu   = np.sum((dt_vals*nu_vals)[:writes])/np.sum(dt_vals[:writes])
                    avg_temp = np.sum((dt_vals*temp_vals)[:writes])/np.sum(dt_vals[:writes])
        else:
            avg_nu = avg_temp = 0

        if args['--ae']:
            ae_solver.loop_tasks()
                    

        if effective_iter % 10 == 0:
            Re_avg = flow.grid_average('Re')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e} ({:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time, solver.sim_time*P,  dt)
            log_string += 'Re: {:8.3e}/{:8.3e}, '.format(Re_avg, flow.max('Re'))
            log_string += 'Nu: {:8.3e} (av: {:8.3e}), '.format(flow.grid_average('Nu'), avg_nu)
            log_string += 'T: {:8.3e} (av: {:8.3e})'.format(flow.grid_average('T'), avg_temp)
            logger.info(log_string)

       
        if first_step:
            if args['--verbose']:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.spy(solver.pencils[0].L, markersize=1, markeredgewidth=0.0)
                fig.savefig(data_dir+"sparsity_pattern.png", dpi=1200)
                
                import scipy.sparse.linalg as sla
                LU = sla.splu(solver.pencils[0].LHS.tocsc(), permc_spec='NATURAL')
                fig = plt.figure()
                ax = fig.add_subplot(1,2,1)
                ax.spy(LU.L.A, markersize=1, markeredgewidth=0.0)
                ax = fig.add_subplot(1,2,2)
                ax.spy(LU.U.A, markersize=1, markeredgewidth=0.0)
                fig.savefig(data_dir+"sparsity_pattern_LU.png", dpi=1200)
                
                logger.info("{} nonzero entries in LU".format(LU.nnz))
                logger.info("{} nonzero entries in LHS".format(solver.pencils[0].LHS.tocsc().nnz))
                logger.info("{} fill in factor".format(LU.nnz/solver.pencils[0].LHS.tocsc().nnz))
            first_step=False
            start_time = time.time()
except:
    raise
    logger.error('Exception raised, triggering end of main loop.')
finally:
    end_time = time.time()
    main_loop_time = end_time-start_time
    n_iter_loop = solver.iteration-1
    logger.info('Iterations: {:d}'.format(n_iter_loop))
    logger.info('Sim end time: {:f}'.format(solver.sim_time))
    logger.info('Run time: {:f} sec'.format(main_loop_time))
    logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
    logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
    try:
        final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
        final_checkpoint.set_checkpoint(solver, wall_dt=1, mode=mode)
        solver.step(dt) #clean this up in the future...works for now.
        post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
    except:
        raise
        print('cannot save final checkpoint')
    finally:
        if not args['--no_join']:
            logger.info('beginning join operation')
            post.merge_analysis(data_dir+'checkpoints')

            for key, task in analysis_tasks.items():
                logger.info(task.base_path)
                post.merge_analysis(task.base_path)

        logger.info(40*"=")
        logger.info('Iterations: {:d}'.format(n_iter_loop))
        logger.info('Sim end time: {:f}'.format(solver.sim_time))
        logger.info('Run time: {:f} sec'.format(main_loop_time))
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
        logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
