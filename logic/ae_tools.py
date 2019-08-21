from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI
import numpy as np

from dedalus import public as de
from dedalus.extras.flow_tools import GlobalFlowProperty

class BoussinesqAESolver:

    def __init__(self, nz, solver_IVP, dist_IVP, ae_fields, extra_fields, P, R,
                 first_ae_wait_time=30,     ae_wait_time=20,
                 first_ae_avg_time=20,      ae_avg_time=10,
                 first_ae_avg_thresh=1e-2,  ae_avg_thresh=1e-3,
                 ivp_convergence_thresh=1e-2):
        """
        Initializes the object; grabs solver states and makes room for profile averages
        
        Arguments:
        nz                  - the (z) resolution of the sims
        solver_IVP          - The IVP solver
        dist_IVP            - The dedalus distributor of the IVP
        num_bvps            - Maximum number of BVPs to solve
        ae_fields           - fields used in AE convergence
        extra_fields        - extra fields tracked
        P, R                - combinations of Ra and Pr in the equations (propto sqrt(Ra))

        first_ae_wait_time  - Sim time to wait after Re > 1 to start AE averaging.
        first_ae_avg_time   - Min sim time to average over for first AE solve
        first_ae_avg_thresh - Averages must be below this threshold to start first AE solve
        ae_wait_time        - Sim time to wait after an AE solve to start AE averaging.
        ae_avg_time         - Min sim time to average over for all but first AE solve
        ae_avg_thresh       - Averages must be below this threshold to start subsequent AE solves
        ivp_convergence_thresh - Change in atmospheric delta_T to achieve to get simulation convergence.
        """
        #Get info about IVP
        self.nz         = nz
        self.solver_IVP = solver_IVP
        self.dist_IVP   = dist_IVP
        self.ae_fields  = ae_fields
        self.extra_fields = extra_fields
        self.doing_ae, self.finished_ae, self.Pe_switch = False, False, False
        self.P, self.R = P, R
        self.AE_basis = de.Chebyshev('z', self.nz, interval=[-1./2, 1./2], dealias=3./2)
        self.AE_domain = de.Domain([self.AE_basis,], grid_dtype=np.float64, comm=MPI.COMM_SELF)

        #Specify how BVPs work
        self.first_ae_wait_time  = self.sim_time_start =  first_ae_wait_time
        self.first_ae_avg_time   = self.min_bvp_time = first_ae_avg_time
        self.first_ae_avg_thresh = self.bvp_threshold = first_ae_avg_thresh

        self.ae_wait_time  = ae_wait_time
        self.ae_avg_time   = ae_avg_time
        self.ae_avg_thresh = ae_avg_thresh

        self.ivp_convergence_thresh = ivp_convergence_thresh

        self.flow = GlobalFlowProperty(solver_IVP, cadence=1)
        self.z_slices    = self.dist_IVP.grid_layout.slices(scales=1)[-1]
        self.nz_per_proc = self.dist_IVP.grid_layout.local_shape(scales=1)[-1]
        self.measured_profiles, self.avg_profiles, self.local_l2 = OrderedDict(), OrderedDict(), OrderedDict()
        for k in ae_fields:
            self.flow.add_property('plane_avg({})'.format(k), name='{}'.format(k))
            self.measured_profiles[k] = np.zeros((2, self.nz_per_proc))
            self.avg_profiles[k]     = np.zeros( self.nz_per_proc )
            self.local_l2[k]     = np.zeros( self.nz_per_proc )
        for k in extra_fields:
            self.flow.add_property('plane_avg({})'.format(k), name='{}'.format(k))
        self.flow.add_property('Pe', name='Pe')

        self.avg_times        = np.zeros(2)
        self.elapsed_avg_time = 0

    def loop_tasks(self, tolerance=1e-10, ncc_cutoff=1e-10):
        """
        Logic for AE performed every loop iteration
        """
        # Don't do anything AE related if Pe < 1
        if self.flow.grid_average('Pe') < 1 and not self.Pe_switch:
            return 
        elif not self.Pe_switch:
            self.sim_time_start += self.solver_IVP.sim_time
            self.Pe_switch = True

        #If first averaging iteration, reset stuff properly 
        first = False
        if not self.doing_ae and not self.finished_ae and self.solver_IVP.sim_time >= self.sim_time_start:
            self.reset_fields() #set time data properly
            self.doing_ae = True
            first = True

        if self.doing_ae:
            self.update_avgs()
            if first: return 

            do_AE = self.check_averager_convergence()
            if do_AE:
                #Get averages from global domain
                avg_fields = OrderedDict()
                for k, prof in self.avg_profiles.items():
                    avg_fields[k] = self.local_to_global_average(prof/self.elapsed_avg_time)

                #Solve BVP
                if self.dist_IVP.comm_cart.rank == 0:
                    problem = de.NLBVP(self.AE_domain, variables=['T1', 'T1_z', 'p', 'delta_T1', 'Xi'], ncc_cutoff=ncc_cutoff)
                    for k, p in (['P', self.P], ['R', self.R]):
                        problem.parameters[k] = p
                    for k, p in avg_fields.items():
                        f = self.AE_domain.new_field()
                        f['g'] = p
                        problem.parameters[k] = f
                    self.set_equations(problem)
                    self.set_BCs(problem)
                    solver = problem.build_solver()
                    pert = solver.perturbations.data
                    pert.fill(1+tolerance)
                    while np.sum(np.abs(pert)) > tolerance:
                        solver.newton_iteration()
                        logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
                else:
                    solver = None

                # Update fields appropriately
                ae_structure = self.local_to_global_ae(solver)
                diff = self.update_simulation_fields(ae_structure, avg_fields)
                
                #communicate diff
                if diff < self.ivp_convergence_thresh: self.finished_ae = True
                logger.info('Diff: {:.4e}, finished_ae? {}'.format(diff, self.finished_ae))
                self.doing_ae = False
                self.sim_time_start = self.solver_IVP.sim_time + self.ae_wait_time
                self.min_bvp_time = self.ae_avg_time
                self.bvp_threshold = self.ae_avg_thresh


    def get_local_profile(self, prof_name):
        """
        Grabs one of the local flow tracker's profiles. Assumes no horizontal variation of profile.

        Arguments:
        ----------
            prof_name: string
                The name of the profile to grab.
        """
        this_field = self.flow.properties['{}'.format(prof_name)]['g']
        if len(this_field.shape) == 3:
            profile = this_field[0,0,:]
        else:
            profile = this_field[0,:]
        return profile

    def local_to_global_average(self, profile):
        """
        Given the local piece of a dedalus z-profile, find the global z-profile.

        Arguments:
        ----------
            profile : NumPy array
                contains the local piece of the profile
        """
        loc, glob = [np.zeros(self.nz) for i in range(2)]
        if len(self.dist_IVP.mesh) == 0:
            loc[self.z_slices] = profile 
        elif self.dist_IVP.comm_cart.rank < self.dist_IVP.mesh[-1]:
            loc[self.z_slices] = profile
        self.dist_IVP.comm_cart.Allreduce(loc, glob, op=MPI.SUM)
        return glob

    def find_global_max(self, profile):
        """
        Given a local piece of a global profile, find the global maximum.

        Arguments:
        ----------
            profile : NumPy array
                contains the local piece of a profile
        """
        loc, glob = [np.zeros(1) for i in range(2)]
        if len(self.dist_IVP.mesh) == 0:
            loc[0] = np.max(profile)
        elif self.dist_IVP.comm_cart.rank < self.dist_IVP.mesh[-1]:
            loc[0] = np.max(profile)
        self.dist_IVP.comm_cart.Allreduce(loc, glob, op=MPI.MAX)
        return glob[0]
            
    def update_avgs(self):
        """
        Updates the averages of z-profiles. To be called every timestep.
        """
        first = False

        #Times
        self.avg_times[0] = self.solver_IVP.sim_time
        this_dt = self.avg_times[0] - self.avg_times[1]
        if self.elapsed_avg_time == 0:
            first = True
        self.elapsed_avg_time += this_dt
        self.avg_times[1] = self.avg_times[0]
    
        #Profiles
        for k in self.ae_fields:
            self.measured_profiles[k][0,:] = self.get_local_profile(k)
            if first:
                self.avg_profiles[k] *= 0
                self.local_l2[k] *= 0
            else:
                old_avg = self.avg_profiles[k]/(self.elapsed_avg_time - this_dt)
                self.avg_profiles[k] += (this_dt/2)*np.sum(self.measured_profiles[k], axis=0)
                new_avg = self.avg_profiles[k]/self.elapsed_avg_time
                self.local_l2[k] = np.abs((new_avg - old_avg)/new_avg)
            self.measured_profiles[k][1,:] = self.measured_profiles[k][0,:]
        
    def reset_fields(self):
        """ Reset all local fields after doing a BVP """
        for fd in self.ae_fields:
            self.avg_profiles[fd]  *= 0
            self.measured_profiles[fd]  *= 0
            self.local_l2[fd]  *= 0
            self.avg_times *= 0
        self.avg_times[1] = self.solver_IVP.sim_time
        self.elapsed_avg_time = 0

    def check_averager_convergence(self):
        """
        For each averager in self.averagers which is being tracked for convergence criterion,
        check if its fields have converged below the bvp threshold for AE.
        """
        if (self.solver_IVP.sim_time  - self.sim_time_start) > self.min_bvp_time:
            maxs = list()
            for f in self.ae_fields:
                maxs.append(self.find_global_max(self.local_l2[f]))

            logger.info('AE: Max abs L2 norm for convergence: {:.4e} / {:.4e}'.format(np.median(maxs), self.bvp_threshold))
            if np.median(maxs) < self.bvp_threshold:
                return True
            else:
                return False
        
    def local_to_global_ae(self, solver):
        """ Communicates AE solve info from process 0 to all processes """
        ae_profiles = OrderedDict()
        ae_profiles['T1'] = np.zeros(self.nz)
        ae_profiles['Xi'] = np.zeros(self.nz)
        ae_profiles['T1_z'] = np.zeros(self.nz)
        ae_profiles['Xi_mean'] = np.zeros(1)
        ae_profiles['delta_T1'] = np.zeros(1)
        full_scalar = np.zeros(1)
        full = np.zeros(self.nz)

        if self.dist_IVP.comm_cart.rank == 0:
            T1       = solver.state['T1']
            Xi       = solver.state['Xi']
            T1_z     = solver.state['T1_z']
            delta_T1 = solver.state['delta_T1']
            T1.set_scales(1, keep_data=True)
            Xi.set_scales(1, keep_data=True)
            T1_z.set_scales(1, keep_data=True)
            ae_profiles['T1'] = np.copy(T1['g'])
            ae_profiles['Xi'] = np.copy(Xi['g'])
            ae_profiles['T1_z'] = np.copy(T1_z['g'])
            ae_profiles['Xi_mean'] = np.mean(Xi.integrate()['g'])
            ae_profiles['delta_T1'] = np.mean(delta_T1['g'])

        self.dist_IVP.comm_cart.Allreduce(ae_profiles['T1'], full, op=MPI.SUM)
        ae_profiles['T1'][:] = full*1.
        full *= 0
        self.dist_IVP.comm_cart.Allreduce(ae_profiles['Xi'], full, op=MPI.SUM)
        ae_profiles['Xi'][:] = full*1.
        full *= 0
        self.dist_IVP.comm_cart.Allreduce(ae_profiles['T1_z'], full, op=MPI.SUM)
        ae_profiles['T1_z'][:] = full*1.
        self.dist_IVP.comm_cart.Allreduce(ae_profiles['Xi_mean'], full_scalar, op=MPI.SUM)
        ae_profiles['Xi_mean'] = full_scalar*1.
        full_scalar *= 0
        self.dist_IVP.comm_cart.Allreduce(ae_profiles['delta_T1'], full_scalar, op=MPI.SUM)
        ae_profiles['delta_T1'] = full_scalar*1.
        full_scalar *= 0
        return ae_profiles
        
    def update_simulation_fields(self, ae_profiles, avg_fields):
        """ Updates T1, T1_z with AE profiles """
        u_scaling = ae_profiles['Xi_mean']**(1./3)
        thermo_scaling = u_scaling**(2)

        #Calculate instantaneous thermo profiles
        [self.flow.properties[f].set_scales(1, keep_data=True) for f in ('T1', 'delta_T1')]
        T1_prof = self.flow.properties['T1']['g']
        old_delta_T1 = np.mean(self.flow.properties['delta_T1']['g'])
        new_delta_T1 = ae_profiles['delta_T1']

        T1 = self.solver_IVP.state['T1']
        T1_z = self.solver_IVP.state['T1_z']

        #Adjust Temp
        T1.set_scales(1, keep_data=True)
        T1['g'] -= T1_prof
        T1.set_scales(1, keep_data=True)
        T1['g'] *= thermo_scaling
        T1.set_scales(1, keep_data=True)
        T1['g'] += ae_profiles['T1'][self.z_slices]
        T1.set_scales(1, keep_data=True)
        T1.differentiate('z', out=self.solver_IVP.state['T1_z'])

        #Adjust velocity
        vel_fields = ['u', 'w']
        if len(T1['g'].shape) == 3:
            vel_fields.append('v')
        for k in vel_fields:
            self.solver_IVP.state[k].set_scales(1, keep_data=True)
            self.solver_IVP.state[k]['g'] *= u_scaling

        #See how much delta S over domain has changed.
        diff = np.mean(np.abs(1 - new_delta_T1/old_delta_T1))
        return diff
        

    def set_equations(self, problem):
        """ Sets the horizontally-averaged boussinesq equations """
        problem.add_equation("Xi = (P/tot_flux)")
        problem.add_equation("delta_T1 = left(T1) - right(T1)")

        problem.add_equation("dz(T1) - T1_z = 0")
        problem.add_equation(("P*dz(T1_z) = dz(Xi*enth_flux)"))
        problem.add_equation(("dz(p) - T1 = Xi*momentum_rhs_z"))
        
    def set_BCs(self, problem):
        """ Sets standard thermal BCs, and also enforces the m = 0 pressure constraint """
        problem.add_bc( "left(T1_z) = 0")
        problem.add_bc( "right(T1) = 0")
        problem.add_bc('right(p) = 0')

    def _update_profiles_dict(self, bc_kwargs, atmosphere, vel_adjust_factor):
        """
        Update the enthalpy flux profile such that the BVP solve gets us the right answer.
        """

        #Get the atmospheric z-points (on the right size grid)
        z = atmosphere._new_field()
        z['g'] = atmosphere.z
        z.set_scales(self.nz/atmosphere.nz, keep_data=True)
        z = z['g']

        #Keep track of initial flux profiles, then make a new enthalpy flux profile
        init_kappa_flux = 1*self.profiles_dict['tot_flux_IVP'] - self.profiles_dict['enth_flux_IVP']
        init_enth_flux = 1*self.profiles_dict['enth_flux_IVP']

        atmosphere.T0_z.set_scales(self.nz/atmosphere.nz, keep_data=True)
        flux_through_system = -atmosphere.P * atmosphere.T0_z['g']
        flux_scaling = flux_through_system / self.profiles_dict['tot_flux_IVP']

        #Scale flux appropriately.
        self.profiles_dict['enth_flux_IVP'] *= flux_scaling #flux_through_system/self.profiles_dict['tot_flux_IVP']
        self.profiles_dict['momentum_rhs_z'] *= flux_scaling #flux_through_system/self.profiles_dict['tot_flux_IVP']

        return flux_scaling

