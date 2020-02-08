"""
Dedalus script for 3D rotating Rayleigh-Benard convection.

This script uses a Fourier basis in the horizontal direction(s) with periodic boundary
conditions. The vertical direction is represented as Chebyshev coefficients.
The equations are scaled in units of the buoyancy time (Fr = 1).

By default, the boundary conditions are:
    Velocity: Impenetrable, stress-free at both the top and bottom
    Thermal:  Fixed flux (bottom), fixed temp (top)

Usage:
    rotating_rayleigh_benard.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e5]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --Ekman=<Ekman>            Ekman number [default: 1e-2]
    --nz=<nz>                  Vertical resolution [default: 32]
    --nx=<nx>                  Horizontal resolution [default: 64]
    --ny=<nx>                  Horizontal resolution [default: 64]
    --aspect=<aspect>          Aspect ratio of problem [default: 2]

    --fixed_f                  Fixed flux boundary conditions top/bottom (FF)
    --fixed_t                  Fixed temperature boundary conditions top/bottom (TT)
    --no_slip                  Stress free boundary conditions top/bottom

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

from logic.output import initialize_rotating_output
from logic.checkpointing import Checkpoint
from logic.ae_tools import BoussinesqAESolver
from logic.extras import global_noise

logger = logging.getLogger(__name__)
args = docopt(__doc__)

### 1. Read in command-line args, set up data directory
fixed_f = args['--fixed_f']
fixed_t = args['--fixed_t']
if not (fixed_f or fixed_t):
    mixed_BCs = True

no_slip = args['--no_slip']
if not no_slip:
    stress_free = True

data_dir = args['--root_dir'] + '/' + sys.argv[0].split('.py')[0]

if fixed_f:
    data_dir += '_fixedF'
elif fixed_t:
    data_dir += '_fixedT'
else:
    data_dir += '_mixedFT'

if args['--ae']:
    data_dir += '_AE'


if stress_free:
    data_dir += '_stressFree'
else:
    data_dir += '_noSlip'

data_dir += "_Ek{}_Ra{}_Pr{}_a{}".format(args['--Ekman'], args['--Rayleigh'], args['--Prandtl'], args['--aspect'])
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
ek = float(args['--Ekman'])
aspect = float(args['--aspect'])
P = (ra*pr)**(-1./2)
R = (ra/pr)**(-1./2)

nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])

logger.info("Ra = {:.3e}, Pr = {:2g}, resolution = {}x{}x{}".format(ra, pr, nx, ny, nz))

### 3. Setup Dedalus domain, problem, and substitutions/parameters
x_basis = de.Fourier( 'x', nx, interval = [-aspect/2, aspect/2], dealias=3/2)
y_basis = de.Fourier( 'y', ny, interval = [-aspect/2, aspect/2], dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval = [-1./2, 1./2], dealias=3/2)

bases = [x_basis, y_basis, z_basis]
domain = de.Domain(bases, grid_dtype=np.float64, mesh=mesh)

variables = ['T1', 'T1_z', 'p', 'u', 'v', 'w', 'Ox', 'Oy', 'Oz']
problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

problem.parameters['P'] = P
problem.parameters['R'] = R
problem.parameters['E'] = ek
problem.parameters['Lx'] = problem.parameters['Ly'] = aspect
problem.parameters['Lz'] = 1

problem.substitutions['T0']   = '(-z + 0.5)'
problem.substitutions['T0_z'] = '-1'
problem.substitutions['Lap(A, A_z)']=       '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
problem.substitutions['UdotGrad(A, A_z)'] = '(u*dx(A) + v*dy(A) + w*A_z)'

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
problem.substitutions['Ro'] = 'sqrt(enstrophy)/(R/E)'
problem.substitutions['true_Ro'] = 'sqrt((v*Oz-w*Oy)**2+(w*Ox-u*Oz)**2+(u*Oy-v*Ox)**2)/(R/E*sqrt(v**2+u**2))'


### 4.Setup equations and Boundary Conditions
problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z           = -UdotGrad(T1, T1_z)")
problem.add_equation("dt(u)  + R*(dy(Oz) - dz(Oy) - v/E)  + dx(p)       =  v*Oz - w*Oy ")
problem.add_equation("dt(v)  + R*(dz(Ox) - dx(Oz) + u/E)  + dy(p)       =  w*Ox - u*Oz ")
problem.add_equation("dt(w)  + R*(dx(Oy) - dy(Ox)      )  + dz(p) - T1  =  u*Oy - v*Ox ")
problem.add_equation("T1_z - dz(T1) = 0")
problem.add_equation("Ox - dy(w) + dz(v) = 0")
problem.add_equation("Oy - dz(u) + dx(w) = 0")
problem.add_equation("Oz - dx(v) + dy(u) = 0")


if fixed_f:
    logger.info("Thermal BC: fixed flux (full form)")
    problem.add_bc( "left(T1_z) = 0")
    problem.add_bc("right(T1_z) = 0")
elif fixed_t:
    logger.info("Thermal BC: fixed temperature (T1)")
    problem.add_bc( "left(T1) = 0")
    problem.add_bc("right(T1) = 0")
else:
    logger.info("Thermal BC: fixed flux/fixed temperature")
    problem.add_bc("left(T1_z) = 0")
    problem.add_bc("right(T1)  = 0")

if stress_free:
    logger.info("Horizontal velocity BC: stress free")
    problem.add_bc("left(Oy) = 0")
    problem.add_bc("right(Oy) = 0")
    problem.add_bc("left(Ox) = 0")
    problem.add_bc("right(Ox) = 0")
else:
    logger.info("Horizontal velocity BC: no slip")
    problem.add_bc( "left(u) = 0")
    problem.add_bc("right(u) = 0")
    problem.add_bc("left(v) = 0")
    problem.add_bc("right(v) = 0")

logger.info("Vertical velocity BC: impenetrable")
problem.add_bc( "left(w) = 0")
problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")
problem.add_bc("right(w) = 0", condition="(nx != 0) or  (ny != 0)")


### 5. Build solver
# Note: SBDF2 timestepper does not currently work with AE.
#ts = de.timesteppers.SBDF2
ts = de.timesteppers.RK222
cfl_safety = float(args['--safety'])
solver = problem.build_solver(ts)
logger.info('Solver built')


### 6. Set initial conditions: noise or loaded checkpoint
checkpoint = Checkpoint(data_dir)
checkpoint_min = 30
restart = args['--restart']
if restart is None:
    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    T1.set_scales(domain.dealias)
    noise = global_noise(domain, int(args['--seed']))
    z_de = domain.grid(-1, scales=domain.dealias)
    T1['g'] = 1e-6*P*np.cos(np.pi*z_de)*noise['g']*(0.5 - z_de)
    T1.differentiate('z', out=T1_z)

    dt = None
    mode = 'overwrite'
else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
    mode = 'append'
checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
   

### 7. Set simulation stop parameters, output, and CFL
if run_time_buoy is not None:    solver.stop_sim_time = run_time_buoy
elif run_time_therm is not None: solver.stop_sim_time = run_time_therm/P
else:                            solver.stop_sim_time = 1/P
solver.stop_wall_time = run_time_wall*3600.

max_dt    = 0.25
if dt is None: dt = max_dt
analysis_tasks = initialize_rotating_output(solver, data_dir, aspect, mode=mode, slice_output_dt=0.25)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
CFL.add_velocities(('u', 'v', 'w'))


### 8. Setup flow tracking for terminal output, including rolling averages
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re", name='Re')
flow.add_property("Ro", name='Ro')
flow.add_property("true_Ro", name='true_Ro')
flow.add_property("vel_rms**2/2", name='KE')
flow.add_property("Nu", name='Nu')
flow.add_property("-1 + (left(T1_z) + right(T1_z) ) / 2", name='Tz_excess')
flow.add_property("T0+T1", name='T')

rank = domain.dist.comm_cart.rank
if rank == 0:
    nu_vals    = np.zeros(5*int(args['--stat_window']))
    ro_vals    = np.zeros(5*int(args['--stat_window']))
    Tz_excess  = np.zeros(5*int(args['--stat_window']))
    temp_vals  = np.zeros(5*int(args['--stat_window']))
    dt_vals    = np.zeros(5*int(args['--stat_window']))
    writes     = 0


### 9. Initialize Accelerated Evolution, if appropriate
if args['--ae']:
    kwargs = { 'first_ae_wait_time' : 30,
               'first_ae_avg_time' : 20,
               'first_ae_avg_thresh' : 1e-2 }
    ae_solver = BoussinesqAESolver( nz, solver, domain.dist, 
                                    ['tot_flux', 'enth_flux', 'momentum_rhs_z'], 
                                    ['T1', 'p', 'delta_T1'], P, R,
                                    **kwargs)

Hermitian_cadence = 100
first_step = True
# Main loop
try:
    count = Re_avg = 0
    logger.info('Starting loop')
    not_corrected_times = True
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
                    solver.stop_sim_time  = run_time_buoy + solver.sim_time
                elif run_time_therm is not None:
                    solver.stop_sim_time = run_time_therm/P + solver.sim_time
                not_corrected_times = False
                



        dt = CFL.compute_dt()
        solver.step(dt) #, trim=True)


        # Solve for blow-up over long timescales in 3D due to hermitian-ness
        effective_iter = solver.iteration - start_iter
        if effective_iter % Hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()
        
        if Re_avg > 1:
            # Rolling average logic 
            if last_time == init_time:
                last_time = solver.sim_time + float(args['--stat_wait_time'])
            if solver.sim_time - last_time >= 0.2:
                avg_Nu, avg_Ro, avg_T, Tz = flow.grid_average('Nu'), flow.grid_average('Ro'), flow.grid_average('T'), flow.grid_average('Tz_excess')
                if domain.dist.comm_cart.rank == 0:
                    if writes != dt_vals.shape[0]:
                        dt_vals[writes] = solver.sim_time - last_time
                        nu_vals[writes] = avg_Nu
                        ro_vals[writes] = Tz
                        temp_vals[writes] = avg_T
                        Tz_excess[writes] = Tz
                        writes += 1
                    else:
                        dt_vals[:-1] = dt_vals[1:]
                        nu_vals[:-1] = nu_vals[1:]
                        ro_vals[:-1] = ro_vals[1:]
                        temp_vals[:-1] = temp_vals[1:]
                        Tz_excess[:-1] = Tz_excess[1:]
                        dt_vals[-1] = solver.sim_time - last_time
                        nu_vals[-1] = avg_Nu
                        ro_vals[-1] = avg_Ro
                        temp_vals[-1] = avg_T
                        Tz_excess[-1] = Tz

        
                    wait_time = 10
                    if np.sum(dt_vals) > wait_time:
                        avg_nu   = np.sum((dt_vals*nu_vals)[:writes])/np.sum(dt_vals[:writes])
                        avg_tz   = np.sum((dt_vals*Tz_excess)[:writes])/np.sum(dt_vals[:writes])
                        avg_temp = np.sum((dt_vals*temp_vals)[:writes])/np.sum(dt_vals[:writes])
                last_time = solver.sim_time

        if args['--ae']:
            ae_solver.loop_tasks()
                    
        if effective_iter % 10 == 0:
            Re_avg = flow.grid_average('Re')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e} ({:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time, solver.sim_time*P,  dt)
            log_string += 'Re: {:8.3e}/{:8.3e}, '.format(Re_avg, flow.max('Re'))
            log_string += 'Ro: {:8.3e}/{:8.3e}, '.format(flow.grid_average('Ro'), flow.max('Ro'))
            log_string += 'true_Ro: {:8.3e}/{:8.3e}, '.format(flow.grid_average('true_Ro'), flow.max('true_Ro'))
            log_string += 'KE: {:8.3e}/{:8.3e}, '.format(flow.grid_average('KE'), flow.max('KE'))
            log_string += 'Nu: {:8.3e} (av: {:8.3e}), '.format(flow.grid_average('Nu'), avg_nu)
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
