"""
Dedalus script for Rayleigh-Benard convection.

This script uses a Fourier basis in the horizontal direction(s) with periodic boundary
conditions. The vertical direction is represented as Chebyshev coefficients.
The equations are scaled in units of the buoyancy time (Fr = 1).

By default, the boundary conditions are:
    Velocity: Impenetrable, no-slip at both the top and bottom
    Thermal:  Fixed flux (bottom), fixed temp (top)

Usage:
    rayleigh_benard.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e5]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nr=<nz>                  Radial resolution [default: 32]
    --nphi=<nx>                Azimuthal resolution [default: 64]

    --FS                       Free-slip/stress free boundary conditions (default No-slip, NS)

    --3D                       Run in 3D
    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --run_time_wall=<time>     Run time, in hours [default: 23.5]
    --run_time_buoy=<time>     Run time, in buoyancy times
    --run_time_therm=<time_>   Run time, in thermal times [default: 1]

    --restart=<file>           Restart from checkpoint file
    --overwrite                If flagged, force file mode to overwrite
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --label=<label>            Optional additional case name label
    --verbose                  Do verbose output (e.g., sparsity patterns of arrays)
    --no_join                  If flagged, don't join files at end of run
    --root_dir=<dir>           Root directory for output [default: ./]
    --safety=<s>               CFL safety factor [default: 0.5]
    --RK443                    Use RK443 instead of RK222


    --stat_wait_time=<t>       Time to wait before taking rolling averages of quantities like Nu [default: 20]
    --stat_window=<t_w>        Max time to take rolling averages over [default: 100]

    --ae                       Do accelerated evolution

"""
import logging
import os
import sys
import time

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post
from dedalus.tools.config import config

from logic.output import initialize_output_annular
from logic.checkpointing import Checkpoint
from logic.ae_tools import BoussinesqAESolver
from logic.extras import global_noise

GLOBAL_NU = None

logger = logging.getLogger(__name__)
args = docopt(__doc__)

### 1. Read in command-line args, set up data directory

FS = args['--FS']
if not FS:
    NS = True
else:
    NS = False

data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]

threeD = args['--3D']
if threeD:
    data_dir += '_3D'
else:
    data_dir += '_2D'

data_dir += '_TT'

if FS:
    data_dir += '_FS'
else:
    data_dir += '_NS'

data_dir += "_Ra{}_Pr{}".format(args['--Rayleigh'], args['--Prandtl'])
if args['--label'] is not None:
    data_dir += "_{}".format(args['--label'])
data_dir += '/'
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.makedirs('{:s}/'.format(data_dir))
    logdir = os.path.join(data_dir,'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
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




### 2. Simulation parameters
ra = float(args['--Rayleigh'])
pr = float(args['--Prandtl'])
P = (ra*pr)**(-1./2)
R = (ra/pr)**(-1./2)

nr = int(args['--nr'])
nφ = int(args['--nphi'])
r_inner = 0.3

C2 = -r_inner
C1 = C2*(r_inner+1)

logger.info("Ra = {:.3e}, Pr = {:2g}, resolution = {}x{}".format(ra, pr, nr, nφ))

### 3. Setup Dedalus domain, problem, and substitutions/parameters
φ_basis = de.Fourier(  'φ', nφ, interval = [0,  2*np.pi], dealias=3/2)
r_basis = de.Chebyshev('r', nr, interval = [r_inner,  r_inner+1], dealias=3/2)

bases = [φ_basis, r_basis]
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)

variables = ['T1', 'T1_r', 'p', 'ur', 'uφ', 'ω']
problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

problem.parameters['P'] = P
problem.parameters['R'] = R
problem.parameters['Lr'] = 1
problem.parameters['pi'] = np.pi
problem.parameters['C1'] = C1
problem.parameters['C2'] = C2

problem.substitutions['T0']   = '(-C1/r + C2)'
problem.substitutions['T0_r'] = '(C1/r**2)'
problem.substitutions['Lap(A, A_r)']      = '( (1/r**2)*dr(r**2*A_r) + (1/r**2)*dφ(dφ(A)) )'
problem.substitutions['UdotGrad(A, A_r)'] = '( ur*A_r + (uφ/r)*dφ(A) )'

problem.substitutions['plane_avg(A)'] = 'integ(r*A, "φ")/(2*pi)'
problem.substitutions['vol_avg(A)']   = 'integ(r*A)/(2*pi*Lr)'
problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'

problem.substitutions['enstrophy']    = 'ω**2'

problem.substitutions['enth_flux'] = '(ur*(T1+T0))'
problem.substitutions['cond_flux'] = '(-P*(T1_r+T0_r))'
problem.substitutions['tot_flux'] = '(cond_flux+enth_flux)'
problem.substitutions['Nu'] = '((enth_flux + cond_flux)/vol_avg(cond_flux))'

problem.substitutions['delta_T1'] = '(left(T1)-right(T1))'
problem.substitutions['vel_rms'] = 'sqrt(ur**2 + uφ**2)'

problem.substitutions['Re'] = '(vel_rms / R)'
problem.substitutions['Pe'] = '(vel_rms / P)'


### 4.Setup equations and Boundary Conditions
problem.add_equation("r*( (1/r**2)*dr(r**2*ur) + (1/r)*dφ(uφ) ) = 0")
problem.add_equation("r**2*( dt(T1) - P*Lap(T1, T1_r) + ur*T0_r )  = -r**2*UdotGrad(T1, T1_r)")
problem.add_equation("r*( dt(ur)  - (R/r)*dφ(ω)   + dr(p)            )  =  -r*uφ*ω ")
problem.add_equation("r*( dt(uφ)  + (R/r)*dr(r*ω) + (1/r)*dφ(p) - T1 )  =   r*ur*ω ")
problem.add_equation("T1_r - dr(T1) = 0")
problem.add_equation("r*( ω - ( (1/r)*( dφ(ur) - dr(r*uφ) ) ) ) = 0")

logger.info("Thermal BC: fixed temperature (T1)")
problem.add_bc( "left(T1) = 0")
problem.add_bc("right(T1) = 0")

if FS:
    logger.info("Horizontal velocity BC: free-slip/stress free")
    problem.add_bc("left(ω) = 0")
    problem.add_bc("right(ω) = 0")
else:
    logger.info("Horizontal velocity BC: no slip")
    problem.add_bc( "left(uφ) = 0")
    problem.add_bc("right(uφ) = 0")

logger.info("Vertical velocity BC: impenetrable")
problem.add_bc( "left(ur) = 0")
problem.add_bc("right(p) = 0",  condition="(nφ == 0)")
problem.add_bc("right(ur) = 0", condition="(nφ != 0)")

### 5. Build solver
# Note: SBDF2 timestepper does not currently work with AE.
#ts = de.timesteppers.SBDF2
if args['--RK443']:
    ts = de.timesteppers.RK443
else:
    ts = de.timesteppers.RK222
cfl_safety = float(args['--safety'])
solver = problem.build_solver(ts)
logger.info('Solver built')


### 6. Set initial conditions: noise or loaded checkpoint
checkpoint = Checkpoint(data_dir)
checkpoint_min = 30
restart = args['--restart']
not_corrected_times = True
true_t_ff = 1
if restart is None:
    T1 = solver.state['T1']
    T1_r = solver.state['T1_r']
    T1.set_scales(domain.dealias)
    T1_r.set_scales(domain.dealias)
    r_de = domain.grid(0, scales=domain.dealias)

    A0 = 1e-6

    #Add noise kick
    noise = global_noise(domain, int(args['--seed']))
    T1['g'] += A0*P*np.cos(np.pi*r_de)*noise['g']
    T1.differentiate('r', out=T1_r)


    dt = None
    mode = 'overwrite'
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
    mode = 'append'
    not_corrected_times = False
checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
   

### 7. Set simulation stop parameters, output, and CFL
if run_time_buoy is not None:    solver.stop_sim_time = run_time_buoy + solver.sim_time
elif run_time_therm is not None: solver.stop_sim_time = run_time_therm/P + solver.sim_time
else:                            solver.stop_sim_time = 1/P + solver.sim_time
solver.stop_wall_time = run_time_wall*3600.

max_dt    = 0.25
if dt is None: dt = max_dt
analysis_tasks = initialize_output_annular(solver, data_dir, threeD=threeD, output_dt=0.1, slice_output_dt=1, vol_output_dt=10, mode=mode, volumes=True)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
CFL.add_velocities(('ur', 'uφ/r'))


### 8. Setup flow tracking for terminal output, including rolling averages
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re", name='Re')
flow.add_property("vel_rms**2/2", name='KE')
flow.add_property("Nu", name='Nu')
flow.add_property("T0+T1", name='T')

first_step = True
# Main loop
try:
    count = Re_avg = 0
    logger.info('Starting loop')
    init_time = last_time = solver.sim_time
    start_iter = solver.iteration
    start_time = time.time()
    avg_nu = avg_temp = avg_tz = 0
    while (solver.ok and np.isfinite(Re_avg)) or first_step:
        if first_step: first_step = False
        if Re_avg > 1:
            # Run times specified at command line are for convection, not for pre-transient.
            if not_corrected_times:
                if run_time_buoy is not None:
                    solver.stop_sim_time  = true_t_ff*run_time_buoy + solver.sim_time
                elif run_time_therm is not None:
                    solver.stop_sim_time = run_time_therm/P + solver.sim_time
                not_corrected_times = False

        effective_iter = solver.iteration - start_iter
                

        dt = CFL.compute_dt()
        solver.step(dt) #, trim=True)

        if effective_iter % 1 == 0:
            Re_avg = flow.grid_average('Re')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e} ({:8.3e} true_ff / {:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time, solver.sim_time/true_t_ff, solver.sim_time*P,  dt)
            log_string += 'Re: {:8.3e}/{:8.3e}, '.format(Re_avg, flow.max('Re'))
            log_string += 'KE: {:8.3e}/{:8.3e}, '.format(flow.grid_average('KE'), flow.max('KE'))
            log_string += 'Nu: {:8.3e} (av: {:8.3e}), '.format(flow.grid_average('Nu'), avg_nu)
            log_string += 'T: {:8.3e} (av: {:8.3e}), '.format(flow.grid_average('T'), avg_temp)
            logger.info(log_string)
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
            post.merge_analysis(data_dir+'checkpoint')

            for key, task in analysis_tasks.items():
                logger.info(task.base_path)
                post.merge_analysis(task.base_path)

        logger.info(40*"=")
        logger.info('Iterations: {:d}'.format(n_iter_loop))
        logger.info('Sim end time: {:f}'.format(solver.sim_time))
        logger.info('Run time: {:f} sec'.format(main_loop_time))
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
        logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
