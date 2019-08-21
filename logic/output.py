"""
    This file is a partial driving script for boussinesq dynamics.  Here,
    formulations of the boussinesq equations are handled in a clean way using
    classes.
"""
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__.split('.')[-1])

def initialize_output(solver, data_dir, aspect, threeD=False, volumes=False,
                      max_writes=20, output_dt=0.1, slice_output_dt=1, vol_output_dt=10,
                      mode="overwrite", **kwargs):
    """
    Sets up output from runs.
    """

    Ly = Lx = aspect
    Lz = 1

    # Analysis
    analysis_tasks = OrderedDict()
    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=max_writes, mode=mode)
    profiles.add_task("plane_avg(T1+T0)", name="T")
    profiles.add_task("plane_avg(dz(T1+T0))", name="Tz")
    profiles.add_task("plane_avg(T1)", name="T1")
    profiles.add_task("plane_avg(u)", name="u")
    profiles.add_task("plane_avg(w)", name="w")
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    profiles.add_task("plane_avg(Nu)", name="Nu")
    profiles.add_task("plane_avg(Re)", name="Re")
    profiles.add_task("plane_avg(Pe)", name="Pe")
    profiles.add_task("plane_avg(enth_flux)", name="enth_flux")
    profiles.add_task("plane_avg(cond_flux)", name="kappa_flux")
    profiles.add_task("plane_avg(cond_flux + enth_flux)", name="tot_flux")

    analysis_tasks['profiles'] = profiles

    scalar = solver.evaluator.add_file_handler(data_dir+'scalar', sim_dt=output_dt, max_writes=max_writes, mode=mode)
    scalar.add_task("vol_avg(T1)", name="IE")
    scalar.add_task("vol_avg(0.5*vel_rms**2)", name="KE")
    scalar.add_task("vol_avg(T1) + vol_avg(0.5*vel_rms**2)", name="TE")
    scalar.add_task("vol_avg(Nu)", name="Nu")
    scalar.add_task("vol_avg(Re)", name="Re")
    scalar.add_task("vol_avg(Pe)", name="Pe")
    scalar.add_task("vol_avg(left(T0+T1) - right(T0+T1))", name="delta_T")
    analysis_tasks['scalar'] = scalar

    if threeD:
        #Analysis
        slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=slice_output_dt, max_writes=max_writes, mode=mode)
        slices.add_task("interp(T1 + T0,         y={})".format(Ly/2), name='T')
        slices.add_task("interp(T1 + T0,         z={})".format(0.95*Lz), name='T near top')
        slices.add_task("interp(T1 + T0,         z={})".format(Lz/2), name='T midplane')
        slices.add_task("interp(w,         y={})".format(Ly/2), name='w')
        slices.add_task("interp(w,         z={})".format(0.95*Lz), name='w near top')
        slices.add_task("interp(w,         z={})".format(Lz/2), name='w midplane')
        slices.add_task("interp(enstrophy,         y={})".format(Ly/2),    name='enstrophy')
        slices.add_task("interp(enstrophy,         z={})".format(0.95*Lz), name='enstrophy near top')
        slices.add_task("interp(enstrophy,         z={})".format(Lz/2),    name='enstrophy midplane')
        analysis_tasks['slices'] = slices

        analysis_tasks['profiles'].add_task('plane_avg(Oz)', name="z_vorticity")
        analysis_tasks['scalar'].add_task('vol_avg(Rossby)', name='Ro')

        if volumes:
            analysis_volume = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=vol_output_dt, max_writes=max_writes, mode=mode)
            analysis_volume.add_task("T1 + T0", name="T")
            analysis_volume.add_task("enstrophy", name="enstrophy")
            analysis_volume.add_task("Oz", name="z_vorticity")
            analysis_tasks['volumes'] = analysis_volume
    else:
        # Analysis
        slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=slice_output_dt, max_writes=max_writes, mode=mode)
        slices.add_task("T1 + T0", name='T')
        slices.add_task("enstrophy")
        slices.add_task("vel_rms")
        slices.add_task("u")
        slices.add_task("w")
        analysis_tasks['slices'] = slices

    powers = solver.evaluator.add_file_handler(data_dir+'powers', sim_dt=slice_output_dt, max_writes=max_writes, mode=mode)
    powers.add_task("interp(T1,         z={})".format(Lz/2),    name='T midplane', layout='c')
    powers.add_task("interp(T1,         z={})".format(0.05*Lz), name='T near bot', layout='c')
    powers.add_task("interp(T1,         z={})".format(0.95*Lz), name='T near top', layout='c')
    powers.add_task("interp(u,         z={})".format(Lz/2),    name='u midplane' , layout='c')
    powers.add_task("interp(u,         z={})".format(0.05*Lz), name='u near bot' , layout='c')
    powers.add_task("interp(u,         z={})".format(0.95*Lz), name='u near top' , layout='c')
    powers.add_task("interp(w,         z={})".format(Lz/2),    name='w midplane' , layout='c')
    powers.add_task("interp(w,         z={})".format(0.05*Lz), name='w near bot' , layout='c')
    powers.add_task("interp(w,         z={})".format(0.95*Lz), name='w near top' , layout='c')
    for i in range(10):
        fraction = 0.1*i
        powers.add_task("interp(T1,     x={})".format(fraction*Lx), name='T at x=0.{}Lx'.format(i), layout='c')
    analysis_tasks['powers'] = powers

    return analysis_tasks

