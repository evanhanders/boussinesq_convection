from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI
import numpy as np

from dedalus import public as de
from dedalus.extras.flow_tools import GlobalFlowProperty

class BoussinesqAESolver:
    """
    Logic for coupling into a Dedalus IVP solve of Boussinesq, Rayleigh-Benard Convection.
    Currently only implemented for mixed thermal boundary conditions (fixed-flux at bottom,
    fixed-temp at top).

    Attributes:
    -----------
    (first_/curr_)ae_wait_time: float
        Freefall times to wait before collecting averages on (first/current) AE solve
    (first_/curr_)ae_avg_time: float
        Minimum number of freefall times over which to take averages on (first/current) AE solve
    (first_/curr_)ae_avg_thresh: float
        Convergence criterion, in % diff, of profiles before (first/current) AE BVP solve triggers
    AE_basis : Dedalus Chebyshev
        1D z-basis for AE solve
    AE_domain : Dedalus Domain
        Domain object for AE solve
    ae_fields : list
        Strings of variables whose profiles must converge on AE solves
    avg_profiles : OrderedDict
        Contains time-averaged profiles
    dist_ivp : Dedalus Distributor
        MPI distributor from convection DNS
    doing_ae : bool
        If True, profiles are actively being averaged for an AE solve
    elapsed_avg_time : float
        Total sim time that has passed over the averaging window
    extra_fields : list
        Fields to track the instantaneous vertical profile of from the DNS.
    finished_ae : bool
        If True, AE has converged within ivp_convergence_thresh
    flow : Dedalus GlobalFlowProperty
        Tracks instantaneous profiles in the DNS
    ivp_convergence_thresh: float
        When AE changes the current simulation's temp profile by this % or lower, consider solution converged.
    local_l2 : OrderedDict
        Contains info about change in profile from previous to current average.
    measured_profiles : OrderedDict
        Contains info about the current and previous instantaneous profile values
    measured_times : NumPy Array
        Sim time at measurement of current and previous profile values
    nz : int
        Chebyshev coefficient resolution
    nz_per_proc : int
        z data points in grid space on the local processor
    P, R : floats
        The values of 1/sqrt(Rayleigh*Prandtl) and 1/sqrt(Rayleigh/Prandtl)
    pe_switch : bool
        True if the avg Pe in the sim grid is > 1
    solver_ivp : Dedalus InitialValueSolver
        Solver object from convection DNS
    z_slices : list
        z indices of the local processor
    """

    def __init__(self, nz, solver_ivp, dist_ivp, ae_fields, extra_fields, P, R,
                 first_ae_wait_time=30,     ae_wait_time=20,
                 first_ae_avg_time=20,      ae_avg_time=10,
                 first_ae_avg_thresh=1e-2,  ae_avg_thresh=1e-3,
                 ivp_convergence_thresh=1e-2):
        """
        Initialize the object by grabbing solver states and making room for profile averages
        
        Arguments: 
        ----------
        All arguments have identical descriptions to their description in the class level docstring.
        """
        self.ivp_convergence_thresh = ivp_convergence_thresh
        self.first_ae_wait_time  = self.curr_ae_wait_time  =  first_ae_wait_time
        self.first_ae_avg_time   = self.curr_ae_avg_time   = first_ae_avg_time
        self.first_ae_avg_thresh = self.curr_ae_avg_thresh = first_ae_avg_thresh
        self.ae_wait_time  = ae_wait_time
        self.ae_avg_time   = ae_avg_time
        self.ae_avg_thresh = ae_avg_thresh
        self.nz            = nz
        self.solver_ivp    = solver_ivp
        self.dist_ivp      = dist_ivp
        self.ae_fields     = ae_fields
        self.extra_fields  = extra_fields
        self.P, self.R     = P, R
        self.doing_ae, self.finished_ae, self.pe_switch = False, False, False

        #Set up profiles and simulation tracking
        self.flow             = GlobalFlowProperty(solver_ivp, cadence=1)
        self.z_slices         = self.dist_ivp.grid_layout.slices(scales=1)[-1]
        self.nz_per_proc      = self.dist_ivp.grid_layout.local_shape(scales=1)[-1]
        self.measured_times   = np.zeros(2)
        self.elapsed_avg_time = 0
        self.measured_profiles, self.avg_profiles, self.local_l2 = OrderedDict(), OrderedDict(), OrderedDict()
        for k in ae_fields:
            self.flow.add_property('plane_avg({})'.format(k), name='{}'.format(k))
            self.measured_profiles[k] = np.zeros((2, self.nz_per_proc))
            self.avg_profiles[k]     = np.zeros( self.nz_per_proc )
            self.local_l2[k]     = np.zeros( self.nz_per_proc )
        for k in extra_fields:
            self.flow.add_property('plane_avg({})'.format(k), name='{}'.format(k))
        self.flow.add_property('Pe', name='Pe')

        if self.dist_ivp.comm_cart.rank == 0:
            self.AE_basis = de.Chebyshev('z', self.nz, interval=[-1./2, 1./2], dealias=3./2)
            self.AE_domain = de.Domain([self.AE_basis,], grid_dtype=np.float64, comm=MPI.COMM_SELF)
        else:
            self.AE_basis = self.AE_domain = None

    def _broadcast_ae_solution(self, solver):
        """ Communicate AE solve info from process 0 to all processes """
        base_keys = ['T1', 'Xi', 'T1_z']
        ae_profiles = OrderedDict()
        for f in base_keys:    ae_profiles[f] = np.zeros(self.nz)
        for f in ['Xi_mean', 'delta_T1']: ae_profiles[f] = np.zeros(1)

        if self.dist_ivp.comm_cart.rank == 0:
            for f in base_keys:
                state = solver.state[f]
                state.set_scales(1, keep_data=True)
                ae_profiles[f] = np.copy(state['g'])
            ae_profiles['Xi_mean'] = np.mean(solver.state['Xi'].integrate()['g'])
            ae_profiles['delta_T1'] = np.mean(solver.state['delta_T1']['g'])

        for f in ae_profiles.keys():
            ae_profiles[f] = self.dist_ivp.comm_cart.bcast(ae_profiles[f], root=0)
        return ae_profiles

    def _check_convergence(self):
        """Check if each field in self.ae_fields is converged, return True if so."""
        if (self.solver_ivp.sim_time  - self.curr_ae_wait_time) > self.curr_ae_avg_time:
            maxs = list()
            for f in self.ae_fields:
                maxs.append(self._get_profile_max(self.local_l2[f]))

            logger.info('AE: Max abs L2 norm for convergence: {:.4e} / {:.4e}'.format(np.median(maxs), self.curr_ae_avg_thresh))
            if np.median(maxs) < self.curr_ae_avg_thresh:
                return True
            else:
                return False

    def _communicate_profile(self, profile):
        """
        Construct and return a global z-profile using local pieces.

        Arguments:
        ----------
            profile : NumPy array
                contains the local piece of the profile
        """
        loc, glob = [np.zeros(self.nz) for i in range(2)]
        if len(self.dist_ivp.mesh) == 0:
            loc[self.z_slices] = profile 
        elif self.dist_ivp.comm_cart.rank < self.dist_ivp.mesh[-1]:
            loc[self.z_slices] = profile
        self.dist_ivp.comm_cart.Allreduce(loc, glob, op=MPI.SUM)
        return glob

    def _get_local_profile(self, key):
        """
        Grab the specified profile from the DNS and return it as a 1D array.

        Arguments:
        ----------
            prof_name: string
                The name of the profile to grab.
        """
        this_field = self.flow.properties['{}'.format(key)]['g']
        if len(this_field.shape) == 3:
            profile = this_field[0,0,:]
        else:
            profile = this_field[0,:]
        return profile

    def _get_profile_max(self, profile):
        """
        Return a profile's global maximum; utilize local pieces and communication.

        Arguments:
        ----------
            profile : NumPy array
                contains the local piece of a profile
        """
        loc, glob = [np.zeros(1) for i in range(2)]
        if len(self.dist_ivp.mesh) == 0:
            loc[0] = np.max(profile)
        elif self.dist_ivp.comm_cart.rank < self.dist_ivp.mesh[-1]:
            loc[0] = np.max(profile)
        self.dist_ivp.comm_cart.Allreduce(loc, glob, op=MPI.MAX)
        return glob[0]
       
    def _reset_profiles(self):
        """ Reset all local fields after doing a BVP """
        for fd in self.ae_fields:
            self.avg_profiles[fd]  *= 0
            self.measured_profiles[fd]  *= 0
            self.local_l2[fd]  *= 0
            self.measured_times *= 0
        self.measured_times[1] = self.solver_ivp.sim_time
        self.elapsed_avg_time = 0

    def _set_AE_equations(self, problem):
        """ Set the horizontally-averaged boussinesq equations """
        problem.add_equation("Xi = (P/tot_flux)")
        problem.add_equation("delta_T1 = left(T1) - right(T1)")

        problem.add_equation("dz(T1) - T1_z = 0")
        problem.add_equation(("P*dz(T1_z) = dz(Xi*enth_flux)"))
        problem.add_equation(("dz(p) - T1 = Xi*momentum_rhs_z"))
        
        problem.add_bc( "left(T1_z) = 0")
        problem.add_bc( "right(T1) = 0")
        problem.add_bc('right(p) = 0')

    def _update_avg_profiles(self):
        """ Update the averages of tracked z-profiles. """
        first = False

        # times
        self.measured_times[0] = self.solver_ivp.sim_time
        this_dt = self.measured_times[0] - self.measured_times[1]
        if self.elapsed_avg_time == 0:
            first = True
        self.elapsed_avg_time += this_dt
        self.measured_times[1] = self.measured_times[0]
    
        # profiles
        for k in self.ae_fields:
            self.measured_profiles[k][0,:] = self._get_local_profile(k)
            if first:
                self.avg_profiles[k] *= 0
                self.local_l2[k] *= 0
            else:
                old_avg = self.avg_profiles[k]/(self.elapsed_avg_time - this_dt)
                self.avg_profiles[k] += (this_dt/2)*np.sum(self.measured_profiles[k], axis=0)
                new_avg = self.avg_profiles[k]/self.elapsed_avg_time
                self.local_l2[k] = np.abs((new_avg - old_avg)/new_avg)
            self.measured_profiles[k][1,:] = self.measured_profiles[k][0,:]
       
    def _update_simulation_state(self, ae_profiles, avg_fields):
        """ Update T1 and T1_z with AE profiles """
        u_scaling = ae_profiles['Xi_mean']**(1./3)
        thermo_scaling = u_scaling**(2)

        #Get instantaneous thermo profiles
        [self.flow.properties[f].set_scales(1, keep_data=True) for f in ('T1', 'delta_T1')]
        T1_prof = self.flow.properties['T1']['g']
        old_delta_T1 = np.mean(self.flow.properties['delta_T1']['g'])
        new_delta_T1 = ae_profiles['delta_T1']

        T1 = self.solver_ivp.state['T1']
        T1_z = self.solver_ivp.state['T1_z']

        #Adjust Temp
        T1.set_scales(1, keep_data=True)
        T1['g'] -= T1_prof
        T1.set_scales(1, keep_data=True)
        T1['g'] *= thermo_scaling
        T1.set_scales(1, keep_data=True)
        T1['g'] += ae_profiles['T1'][self.z_slices]
        T1.set_scales(1, keep_data=True)
        T1.differentiate('z', out=self.solver_ivp.state['T1_z'])

        #Adjust velocity
        vel_fields = ['u', 'w']
        if len(T1['g'].shape) == 3:
            vel_fields.append('v')
        for k in vel_fields:
            self.solver_ivp.state[k].set_scales(1, keep_data=True)
            self.solver_ivp.state[k]['g'] *= u_scaling

        #See how much delta T over domain has changed.
        diff = np.mean(np.abs(1 - new_delta_T1/old_delta_T1))
        return diff

    def loop_tasks(self, tolerance=1e-10):
        """ 
        Perform AE tasks every loop iteration.

        Arguments:
        ---------
        tolerance : float
            convergence tolerance for AE NLBVP
        """
        # Don't do anything AE related if Pe < 1
        if self.flow.grid_average('Pe') < 1 and not self.pe_switch:
            return 
        elif not self.pe_switch:
            self.curr_ae_wait_time += self.solver_ivp.sim_time
            self.pe_switch = True

        #If first averaging iteration, reset stuff properly 
        first = False
        if not self.doing_ae and not self.finished_ae and self.solver_ivp.sim_time >= self.curr_ae_wait_time:
            self._reset_profiles() #set time data properly
            self.doing_ae = True
            first = True

        if self.doing_ae:
            self._update_avg_profiles()
            if first: return 

            do_AE = self._check_convergence()
            if do_AE:
                #Get averages from global domain
                avg_fields = OrderedDict()
                for k, prof in self.avg_profiles.items():
                    avg_fields[k] = self._communicate_profile(prof/self.elapsed_avg_time)

                #Solve BVP
                if self.dist_ivp.comm_cart.rank == 0:
                    problem = de.NLBVP(self.AE_domain, variables=['T1', 'T1_z', 'p', 'delta_T1', 'Xi'])
                    for k, p in (['P', self.P], ['R', self.R]):
                        problem.parameters[k] = p
                    for k, p in avg_fields.items():
                        f = self.AE_domain.new_field()
                        f['g'] = p
                        problem.parameters[k] = f
                    self._set_AE_equations(problem)
                    solver = problem.build_solver()
                    pert = solver.perturbations.data
                    pert.fill(1+tolerance)
                    while np.sum(np.abs(pert)) > tolerance:
                        solver.newton_iteration()
                        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
                else:
                    solver = None

                # Update fields appropriately
                ae_structure = self._broadcast_ae_solution(solver)
                diff = self._update_simulation_state(ae_structure, avg_fields)
                
                #communicate diff
                if diff < self.ivp_convergence_thresh: self.finished_ae = True
                logger.info('Diff: {:.4e}, finished_ae? {}'.format(diff, self.finished_ae))
                self.doing_ae = False
                self.curr_ae_wait_time = self.solver_ivp.sim_time + self.ae_wait_time
                self.curr_ae_avg_time = self.ae_avg_time
                self.curr_ae_avg_thresh = self.ae_avg_thresh
