"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. The simulation should take roughly 10
cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

Usage:
    rayleigh_benard_d3.py [options] 

Options:
    --Ra=<Rayleigh>            The Rayleigh number [default: 3e3]
    --Pr=<Prandtl>             The Prandtl number  [default: 1e-2]
    --a=<aspect>               The aspect ratio    [default: 4]

    --nz=<nz>                  Vertical resolution [default: 32]
    --nx=<nx>                  Horizontal resolution [default: 64]

    --run_time_buoy=<time>     Run time, in buoyancy times [default: 5e2]
    --wall_time=<hrs>          Run time, in hours [default: 23.5]

    --root_dir=<dir>           Root directory for output [default: ./]
    --label=<label>            Optional additional case name label
"""

from docopt import docopt
import numpy as np
import time
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

from logic.parsing import construct_out_dir

# TODO: maybe fix plotting to directly handle vectors
# TODO: optimize and match d2 resolution
# TODO: get unit vectors from coords?

args = docopt(__doc__)

# Parameters
Lx, Lz = float(args['--a']), 1
Nx, Nz = int(args['--nx']), int(args['--nz'])
Rayleigh = float(args['--Ra'])
Prandtl = Pr = float(args['--Pr'])
dealias = 3/2
stop_sim_time = float(args['--run_time_buoy'])
stop_wall_time = float(args['--wall_time'])*60*60
timestepper = d3.SBDF2
max_timestep = 0.25
dtype = np.float64


data_dir = construct_out_dir(args, {}, base_flags=['Ra', 'Pr', 'a'], label_flags=[], resolution_flags=['nx', 'nz'])
logger.info('saving outputs to {}'.format(data_dir))

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
x = xbasis.local_grid(1)
z = zbasis.local_grid(1)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau1b = dist.Field(name='tau1b', bases=xbasis)
tau2b = dist.Field(name='tau2b', bases=xbasis)
tau1u = dist.VectorField(coords, name='tau1u', bases=xbasis)
tau2u = dist.VectorField(coords, name='tau2u', bases=xbasis)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

ex = dist.VectorField(coords, name='ex')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ez['g'][1] = 1

lift_basis = zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.LiftTau(A, lift_basis, n)
grad_u = d3.grad(u) + ez*lift(tau1u,-1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau1b,-1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau1b, tau2b, tau1u, tau2u], namespace=locals())
problem.add_equation("trace(grad_u) = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau2b,-1) = - dot(u,grad(b))")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) + lift(tau2u,-1) - b*ez = - dot(u,grad(u))")
problem.add_equation("b(z=0) = Lz")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0", condition="nx != 0")
problem.add_equation("dot(ex,u)(z=Lz) = 0", condition="nx == 0")
problem.add_equation("p(z=Lz) = 0", condition="nx == 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = stop_wall_time

# Initial conditions
zb, zt = zbasis.bounds
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b['g'] += Lz - z # Add linear background

plane_avg = lambda A: d3.Integrate(A/Lx, coords['x'])
vol_avg = lambda A: d3.Integrate(d3.Integrate(A/Lx/Lz, coords['z']),coords['x'])
dx = lambda A: d3.Differentiate(A, coords['x'])
dz = lambda A: d3.Differentiate(A, coords['z'])
ω = dz(d3.dot(u, ex)) - dx(d3.dot(u, ez))
enstrophy = ω**2

Fconv = b*u
Fcond = -kappa*d3.grad(b)

# Analysis
snapshots = solver.evaluator.add_file_handler('{}/slices'.format(data_dir), sim_dt=0.5, max_writes=50)
snapshots.add_task(p)
snapshots.add_task(b)
snapshots.add_task(d3.dot(u,ex), name='ux')
snapshots.add_task(d3.dot(u,ez), name='uz')

profiles = solver.evaluator.add_file_handler('{}/profiles'.format(data_dir), sim_dt=0.5, max_writes=50)
profiles.add_task(plane_avg(b), name='b')
profiles.add_task(plane_avg(d3.dot(u,u)/2), name='KE')
profiles.add_task(plane_avg(enstrophy), name='enstrophy')
profiles.add_task(plane_avg(d3.dot(ez, Fcond)), name='Fcond')
profiles.add_task(plane_avg(d3.dot(ez, Fconv)), name='Fconv')
profiles.add_task(plane_avg(d3.dot(ez, Fconv+Fcond)), name='Ftot')
profiles.add_task(plane_avg(d3.dot(ez, Fconv+Fcond))/vol_avg(d3.dot(ez, Fcond)), name='Nu')

scalars = solver.evaluator.add_file_handler('{}/scalars'.format(data_dir), sim_dt=0.5, max_writes=50)
scalars.add_task(vol_avg(d3.dot(u,u)/2), name='KE')
scalars.add_task(np.sqrt(vol_avg(d3.dot(u,u)))/nu, name='Re')
scalars.add_task(np.sqrt(vol_avg(d3.dot(u,u)))/kappa, name='Pe')
scalars.add_task(vol_avg(enstrophy), name='enstrophy')
scalars.add_task(vol_avg(d3.dot(ez, Fcond)), name='Fcond')
scalars.add_task(vol_avg(d3.dot(ez, Fconv)), name='Fconv')
scalars.add_task(1 + vol_avg(d3.dot(ez, Fconv))/vol_avg(d3.dot(ez, Fcond)), name='Nu')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(d3.dot(u,u))/nu, name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)/Pe=%f/%f' %(solver.iteration, solver.sim_time, timestep, max_Re, max_Re*Pr))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*dist.comm.size))
