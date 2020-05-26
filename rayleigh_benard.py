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
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e4]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nz=<nz>                  Vertical resolution [default: 32]
    --nx=<nx>                  Horizontal resolution [default: 64]
    --ny=<nx>                  Horizontal resolution [default: 64]
    --aspect=<aspect>          Aspect ratio of problem [default: 2]

    --FF                       Fixed flux boundary conditions top/bottom (default FT)
    --TT                       Fixed temperature boundary conditions top/bottom (default FT)
    --FS                       Free-slip/stress free boundary conditions (default No-slip, NS)
    --smart_ICs                Use smarter static initial conditions for flux boundaries

    --3D                       Run in 3D
    --mesh=<mesh>              Processor mesh if distributing 3D run in 2D 
    
    --run_time_wall=<time>     Run time, in hours [default: 23.5]
    --run_time_buoy=<time>     Run time, in buoyancy times
    --run_time_therm=<time_>   Run time, in thermal times [default: 1]

    --restart=<file>           Restart from checkpoint file
    --restart_Nu=<Nu>          Nusselt number of run that is being restarted, for adjusting t_ff [default: 1]
    --TT_to_FT=<file>          Restart from checkpoint file, going from TT to FT BCs 
    --TT_to_FT_Nu=<Nu>         Nusselt number of fixed-T run being restarted from [default: 1]
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

from logic.output import initialize_output
from logic.checkpointing import Checkpoint
from logic.ae_tools import BoussinesqAESolver
from logic.extras import global_noise

GLOBAL_NU = None

logger = logging.getLogger(__name__)
args = docopt(__doc__)

### 1. Read in command-line args, set up data directory
FF = args['--FF']
TT = args['--TT']
if not (FF or TT):
    FT = True
else:
    FT = False

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

if FF:
    data_dir += '_FF'
elif TT:
    data_dir += '_TT'
else:
    data_dir += '_FT'

if args['--smart_ICs']:
    data_dir += '_smart'

if args['--ae']:
    data_dir += '_AE'

if args['--TT_to_FT'] is not None:
    data_dir += '_TTtoFT'

if FS:
    data_dir += '_FS'
else:
    data_dir += '_NS'

data_dir += "_Ra{}_Pr{}_a{}".format(args['--Rayleigh'], args['--Prandtl'], args['--aspect'])
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
aspect = float(args['--aspect'])
P = (ra*pr)**(-1./2)
R = (ra/pr)**(-1./2)

nx = int(args['--nx'])
ny = int(args['--ny'])
nz = int(args['--nz'])

logger.info("Ra = {:.3e}, Pr = {:2g}, resolution = {}x{}x{}".format(ra, pr, nx, ny, nz))

### 3. Setup Dedalus domain, problem, and substitutions/parameters
x_basis = de.Fourier( 'x', nx, interval = [-aspect/2, aspect/2], dealias=3/2)
if threeD : y_basis = de.Fourier( 'y', ny, interval = [-aspect/2, aspect/2], dealias=3/2)
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

problem.substitutions['T0']   = '(-z + 0.5)'
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


### 4.Setup equations and Boundary Conditions
problem.add_equation("dx(u) + dy(v) + dz(w) = 0")
problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z           = -UdotGrad(T1, T1_z)")
problem.add_equation("dt(u)  + R*(dy(Oz) - dz(Oy))  + dx(p)       =  v*Oz - w*Oy ")
if threeD: problem.add_equation("dt(v)  + R*(dz(Ox) - dx(Oz))  + dy(p)       =  w*Ox - u*Oz ")
problem.add_equation("dt(w)  + R*(dx(Oy) - dy(Ox))  + dz(p) - T1  =  u*Oy - v*Ox ")
problem.add_equation("T1_z - dz(T1) = 0")
if threeD: problem.add_equation("Ox - dy(w) + dz(v) = 0")
problem.add_equation("Oy - dz(u) + dx(w) = 0")
if threeD: problem.add_equation("Oz - dx(v) + dy(u) = 0")


if FF:
    logger.info("Thermal BC: fixed flux (full form)")
    problem.add_bc( "left(T1_z) = 0")
    problem.add_bc("right(T1_z) = 0")
elif TT:
    logger.info("Thermal BC: fixed temperature (T1)")
    problem.add_bc( "left(T1) = 0")
    problem.add_bc("right(T1) = 0")
else:
    logger.info("Thermal BC: fixed flux/fixed temperature")
    problem.add_bc("left(T1_z) = 0")
    problem.add_bc("right(T1)  = 0")

if FS:
    logger.info("Horizontal velocity BC: free-slip/stress free")
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

logger.info("Vertical velocity BC: impenetrable")
problem.add_bc( "left(w) = 0")
if threeD:
    problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")
    problem.add_bc("right(w) = 0", condition="(nx != 0) or  (ny != 0)")
else:
    problem.add_bc("right(p) = 0", condition="(nx == 0)")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")

# Ra crit literature values from Goluskin 2016 RBC & IH convection, table 2.2
if   FF and FS:
    ra_crit = 120
elif FF and NS:
    ra_crit = 720
elif TT and FS:
    ra_crit = 657.5
elif TT and NS:
    ra_crit = 1707.76
elif FT and FS:
    ra_crit = 384.693
else: #FT, NS
    ra_crit = 1295.78


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
TT_to_FT = args['--TT_to_FT']
not_corrected_times = True
true_t_ff = 1
if restart is None and TT_to_FT is None:
    p = solver.state['p']
    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    p.set_scales(domain.dealias)
    T1.set_scales(domain.dealias)
    T1_z.set_scales(domain.dealias)
    z_de = domain.grid(-1, scales=domain.dealias)

    A0 = 1e-6

    if args['--smart_ICs']:
        from scipy.special import erf
        def one_to_zero(z, z0, delta):
            return -(1/2)*(erf( (z - z0) / delta ) - 1)
        def zero_to_one(*args):
            return 1-one_to_zero(*args)

        # for some reason I run into filesystem errors when I just use T1 to antidifferentiate for HS for HSE
        # use a work field instead.
        work_field  = domain.new_field()
        work_field.set_scales(domain.dealias)

        if FT:
            #Solve out for estimated delta T / BL depth from Nu v Ra.
            Nu_law_const  = 0.138
            Nu_law_alpha  = 0.285
            Nu_estimate   = (Nu_law_const*ra**(Nu_law_alpha))**(1/(1+Nu_law_alpha))

            dT_evolved  = -1/(Nu_estimate)
            d_BL        = dT_evolved/(-2) #thermal BL depth
            true_t_ff   = np.sqrt(Nu_estimate)

            logger.info('Constructing smart ICs with Nu: {:.2e} / t_ff: {:.2e}'.format(Nu_estimate, true_t_ff))


            #Generate windowing function for boundary layers where dT/dz = -1
            window = one_to_zero(z_de, -0.5+2*d_BL, d_BL/2) + zero_to_one(z_de, 0.5-2*d_BL, d_BL/2) 
            T1_z['g'] = window
            w_integ   = np.mean(T1_z.integrate('z')['g'])

            # dT/dz = (grad T)_interior + window*(-1 - (grad T)_interior)
            # assuming (grad T)_interior is a constant, integ ( dT/dz ) = dT_evolved
            # Rearrange for (grad T)_interior
            grad_T_interior = (dT_evolved + w_integ) / (1 - w_integ)
            T1_z['g'] = grad_T_interior + window*(-1 - grad_T_interior) # Full T field
            T1_z['g'] -= (-1) #Subtract off T0z of constant coefficient.
            T1_z.antidifferentiate('z', ('right', 0), out=T1)

            #Hydrostatic equilibrium
            work_field['g'] = T1['g'] 
            work_field.antidifferentiate(  'z', ('right', 0), out=p) #hydrostatic equilibrium

            #Adjust magnitude of noise due to cos envelope & estimated magnitude of FT temperature fluctuations.
            A0 /= np.cos(np.pi*(-0.5+2*d_BL)) 
            A0 /= Nu_estimate
        else:
            logger.info("WARNING: Smart ICS not implemented for boundary condition choice.")
        

    #Add noise kick
    noise = global_noise(domain, int(args['--seed']))
    T1['g'] += A0*P*np.cos(np.pi*z_de)*noise['g']
    T1.differentiate('z', out=T1_z)


    dt = None
    mode = 'overwrite'
elif TT_to_FT is not None:
    logger.info("restarting from {} and swapping BCs".format(TT_to_FT))
    dt = checkpoint.restart(TT_to_FT, solver)
    mode = 'overwrite'

    Nu = float(args['--TT_to_FT_Nu'])
    true_t_ff   = np.sqrt(Nu)
            
    T1 = solver.state['T1']
    u = solver.state['u']
    w = solver.state['w']
    vels = [u, w]
    if threeD:
        v = solver.state['v']
        vels.append(v)

    #Adjust from fixed T to fixed flux
    z_de = domain.grid(-1, scales=domain.dealias)
    T1.set_scales(domain.dealias, keep_data=True)
    T1['g'] += (0.5 - z_de) #Add T0
    T1['g'] /= Nu        #Scale Temp flucs properly
    T1['g'] -= (0.5 - z_de) #Subtract T0
    for v in vels:
        v['g'] /= np.sqrt(Nu)
    not_corrected_times = False

else:
    logger.info("restarting from {}".format(restart))
    dt = checkpoint.restart(restart, solver)
    mode = 'append'
    not_corrected_times = False
    Nu = float(args['--restart_Nu'])
    true_t_ff = np.sqrt(Nu)
checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
   

### 7. Set simulation stop parameters, output, and CFL
if run_time_buoy is not None:    solver.stop_sim_time = run_time_buoy*true_t_ff + solver.sim_time
elif run_time_therm is not None: solver.stop_sim_time = run_time_therm/P + solver.sim_time
else:                            solver.stop_sim_time = 1/P + solver.sim_time
solver.stop_wall_time = run_time_wall*3600.

max_dt    = np.min((0.1*true_t_ff, 1))
if dt is None: dt = max_dt
analysis_tasks = initialize_output(solver, data_dir, aspect, threeD=threeD, output_dt=0.1*true_t_ff, slice_output_dt=1*true_t_ff, vol_output_dt=10*true_t_ff, mode=mode, volumes=True)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
if threeD:
    CFL.add_velocities(('u', 'v', 'w'))
else:
    CFL.add_velocities(('u', 'w'))


### 8. Setup flow tracking for terminal output, including rolling averages
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("Re", name='Re')
flow.add_property("vel_rms**2/2", name='KE')
flow.add_property("Nu", name='Nu')
flow.add_property("T0+T1", name='T')

rank = domain.dist.comm_cart.rank
if rank == 0:
    nu_vals    = np.zeros(5*int(args['--stat_window']))
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
    init_time = last_time = solver.sim_time
    start_iter = solver.iteration
    start_time = time.time()
    avg_nu = avg_temp = avg_tz = 0
    wait_time = float(args['--stat_wait_time'])*true_t_ff
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
                



        dt = CFL.compute_dt()
        solver.step(dt) #, trim=True)


        # Solve for blow-up over long timescales in 3D due to hermitian-ness
        effective_iter = solver.iteration - start_iter
        if threeD and effective_iter % Hermitian_cadence == 0:
            for field in solver.state.fields:
                field.require_grid_space()
        
        if Re_avg > 1:
            # Rolling average logic 
            if last_time == init_time:
                last_time = solver.sim_time + wait_time 
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

        
                    if np.sum(dt_vals) > wait_time:
                        GLOBAL_NU = avg_nu   = np.sum((dt_vals*nu_vals)[:writes])/np.sum(dt_vals[:writes])
                        avg_temp = np.sum((dt_vals*temp_vals)[:writes])/np.sum(dt_vals[:writes])
                last_time = solver.sim_time
                GLOBAL_NU = domain.dist.comm_cart.bcast(GLOBAL_NU, root=0)

        if args['--ae']:
            ae_solver.loop_tasks()
                    
        if effective_iter % 10 == 0:
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
