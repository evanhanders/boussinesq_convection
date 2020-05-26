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
    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=max_writes*10, mode=mode)
    profiles.add_task("plane_avg(T1+T0)", name="T")
    profiles.add_task("plane_avg(dz(T1+T0))", name="Tz")
    profiles.add_task("plane_avg(T1)", name="T1")
    profiles.add_task("plane_avg(u)", name="u")
    profiles.add_task("plane_avg(w)", name="w")
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    profiles.add_task('plane_avg(Oy)', name="y_vorticity")
    profiles.add_task("plane_avg(Nu)", name="Nu")
    profiles.add_task("plane_avg(Re)", name="Re")
    profiles.add_task("plane_avg(Pe)", name="Pe")
    profiles.add_task("plane_avg(enth_flux)", name="enth_flux")
    profiles.add_task("plane_avg(cond_flux)", name="kappa_flux")
    profiles.add_task("plane_avg(cond_flux + enth_flux)", name="tot_flux")

    analysis_tasks['profiles'] = profiles

    scalar = solver.evaluator.add_file_handler(data_dir+'scalar', sim_dt=output_dt, max_writes=max_writes*100, mode=mode)
    scalar.add_task("vol_avg(T1)", name="IE")
    scalar.add_task("vol_avg(0.5*vel_rms**2)", name="KE")
    scalar.add_task("vol_avg(T1) + vol_avg(0.5*vel_rms**2)", name="TE")
    scalar.add_task("vol_avg(Nu)", name="Nu")
    scalar.add_task("vol_avg(Re)", name="Re")
    scalar.add_task("vol_avg(Pe)", name="Pe")
    scalar.add_task("vol_avg(enstrophy)", name="enstrophy")
    scalar.add_task("vol_avg(-inv_Re_ff*enstrophy)", name="visc_KE_source")
    scalar.add_task("vol_avg(w*T1)", name="buoy_KE_source")
    scalar.add_task("vol_avg(w*(T0+T1))", name="wT")
    scalar.add_task("vol_avg(w*(T0+T1) - inv_Re_ff*enstrophy)", name="KE_change")
    scalar.add_task("vol_avg(left(T0+T1) - right(T0+T1))", name="delta_T")
    scalar.add_task("vol_avg(left(T0+T1))", name="left_T")
    scalar.add_task("vol_avg(right(T0+T1))", name="right_T")
    scalar.add_task("vol_avg(left(cond_flux))", name="left_flux")
    scalar.add_task("vol_avg(right(cond_flux))", name="right_flux")
    scalar.add_task("vol_avg(2*T1*dx(Oy))", name="enstrophy_buoy_source")
    scalar.add_task("vol_avg(w*dz(Oy**2) + u*dx(Oy**2))", name="enstrophy_advec_source")
    scalar.add_task("vol_avg(-2*inv_Re_ff*(dz(Oy)**2 + dx(Oy)**2))", name="enstrophy_visc_source")
    scalar.add_task("vol_avg(u)",  name="u")
    scalar.add_task("vol_avg(w)",  name="w")
    analysis_tasks['scalar'] = scalar

    if threeD:
        #Analysis
        slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=slice_output_dt, max_writes=max_writes, mode=mode)
        slices.add_task("interp(T1 + T0,         y={})".format(0), name='T')
        slices.add_task("interp(T1 + T0,         z={})".format(0.49), name='T near top')
        slices.add_task("interp(T1 + T0,         z={})".format(-0.49), name='T near bot 1')
        slices.add_task("interp(T1 + T0,         z={})".format(-0.48), name='T near bot 2')
        slices.add_task("interp(T1 + T0,         z={})".format(-0.47), name='T near bot 3')
        slices.add_task("interp(T1 + T0,         z={})".format(0), name='T midplane')
        slices.add_task("interp(w,         y={})".format(0), name='w')
        slices.add_task("interp(w,         z={})".format(0.49), name='w near top')
        slices.add_task("interp(w,         z={})".format(-0.49), name='w near bot')
        slices.add_task("interp(w,         z={})".format(0), name='w midplane')
        slices.add_task("interp(enstrophy,         y={})".format(0),    name='enstrophy')
        slices.add_task("interp(enstrophy,         z={})".format(0.49), name='enstrophy near top')
        slices.add_task("interp(enstrophy,         z={})".format(-0.49), name='enstrophy near bot')
        slices.add_task("interp(enstrophy,         z={})".format(0),    name='enstrophy midplane')
        analysis_tasks['slices'] = slices

        analysis_tasks['profiles'].add_task('plane_avg(Oz)', name="z_vorticity")
        analysis_tasks['profiles'].add_task('plane_avg(Ox)', name="x_vorticity")
        analysis_tasks['profiles'].add_task('plane_avg(v)', name="v")


        analysis_tasks['scalar'].add_task("vol_avg(v)",  name="v")

        if volumes:
            analysis_volume = solver.evaluator.add_file_handler(data_dir+'volumes', sim_dt=vol_output_dt, max_writes=5, mode=mode)
            analysis_volume.add_task("T1 + T0", name="T")
            analysis_volume.add_task("w*(T1 + T0)", name="wT")
            analysis_tasks['volumes'] = analysis_volume
    else:
        # Analysis
        slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=slice_output_dt, max_writes=max_writes, mode=mode)
        slices.add_task("T1 + T0", name='T')
        slices.add_task("u")
        slices.add_task("w")
        slices.add_task("enth_flux")
        slices.add_task("enstrophy")
        #slices.add_task("2*T1*dx(Oy)",                  name="enstrophy_buoy_source")
        #slices.add_task("w*dz(Oy**2) + u*dx(Oy**2)",    name="enstrophy_advec_source")
        #slices.add_task("-2*R*(dz(Oy)**2 + dx(Oy)**2)", name="enstrophy_visc_source")
        analysis_tasks['slices'] = slices

        powers = solver.evaluator.add_file_handler(data_dir+'powers', sim_dt=slice_output_dt, max_writes=max_writes*10, mode=mode)
        powers.add_task("interp(T1,         z={})".format(0),    name='T midplane', layout='c')
        powers.add_task("interp(T1,         z={})".format(-0.49), name='T near bot', layout='c')
        powers.add_task("interp(T1,         z={})".format(0.49), name='T near top', layout='c')
        powers.add_task("interp(u,         z={})".format(0),    name='u midplane' , layout='c')
        powers.add_task("interp(u,         z={})".format(-0.49), name='u near bot' , layout='c')
        powers.add_task("interp(u,         z={})".format(0.49), name='u near top' , layout='c')
        powers.add_task("interp(w,         z={})".format(0),    name='w midplane' , layout='c')
        powers.add_task("interp(w,         z={})".format(-0.49), name='w near bot' , layout='c')
        powers.add_task("interp(w,         z={})".format(0.49), name='w near top' , layout='c')
        for i in range(10):
            fraction = 0.1*i
            powers.add_task("interp(T1,     x={})".format(fraction*Lx), name='T at x=0.{}Lx'.format(i), layout='c')
        analysis_tasks['powers'] = powers

    return analysis_tasks


def initialize_rotating_output(*args, **kwargs):
    analysis_tasks = initialize_output(*args, threeD=True, **kwargs)
    analysis_tasks['scalar'].add_task("vol_avg(Ro)", name="Ro")
    analysis_tasks['scalar'].add_task("vol_avg(true_Ro)", name="true_Ro")
    analysis_tasks['scalar'].add_task("vol_avg(Ox)", name="Ox")
    analysis_tasks['scalar'].add_task("vol_avg(Oy)", name="Oy")
    analysis_tasks['scalar'].add_task("vol_avg(Oz)", name="Oz")


    analysis_tasks['slices'].add_task("interp(Oz,         y={})".format(0),    name='vort_z')
    analysis_tasks['slices'].add_task("interp(Oz,         z={})".format(0.45), name='vort_z near top')
    analysis_tasks['slices'].add_task("interp(Oz,         z={})".format(0),    name='vort_z midplane')
    analysis_tasks['slices'].add_task("integ( Oz,          'z')",              name='vort_z integ')

    if 'volumes' in analysis_tasks.keys():
        analysis_tasks['volumes'].add_task('Oz', name='z_vorticity')
        analysis_tasks['volumes'].add_task('w', name='w')
        analysis_tasks['volumes'].add_task('u', name='u')

    return analysis_tasks

def initialize_magnetic_output(*args, plot_boundaries=True, **kwargs): #A or B here ?
    analysis_tasks = initialize_output(*args, **kwargs)
    analysis_tasks['scalar'].add_task("vol_avg(b_mag)", name="b_mag")
    analysis_tasks['scalar'].add_task("vol_avg(b_perp)", name="b_perp")
    analysis_tasks['scalar'].add_task("vol_avg(Bx)", name="Bx")
    analysis_tasks['scalar'].add_task("vol_avg(By)", name="By")
    analysis_tasks['scalar'].add_task("vol_avg(Bz)", name="Bz")
    analysis_tasks['scalar'].add_task("sqrt(vol_avg(Bx**2))", name="Bx_rms")
    analysis_tasks['scalar'].add_task("sqrt(vol_avg(By**2))", name="By_rms")
    analysis_tasks['scalar'].add_task("sqrt(vol_avg(Bz**2))", name="Bz_rms")
    for fd in ['Bx', 'By', 'Jx', 'Jy']:
        analysis_tasks['scalar'].add_task("vol_avg(right({}))".format(fd), name="right_{}".format(fd))
        analysis_tasks['scalar'].add_task("vol_avg(left({}))".format(fd), name="left_{}".format(fd))


       

    if kwargs['threeD']:
        boundary_type = 'slices'
        analysis_tasks['slices'].add_task("interp(Bz,         y={})".format(0),    name='mag_field_z')
        analysis_tasks['slices'].add_task("interp(Bz,         z={})".format(0.45), name='mag_field_z near top')
        analysis_tasks['slices'].add_task("interp(Bz,         z={})".format(0),    name='mag_field_z midplane')
        analysis_tasks['slices'].add_task("integ( Bz,          'z')",              name='mag_field_z integ')
    else:
        boundary_type = 'profiles'
        analysis_tasks['slices'].add_task("Bz")
        analysis_tasks['slices'].add_task("By")
        analysis_tasks['slices'].add_task("Bx")

    if plot_boundaries:
        analysis_tasks[boundary_type].add_task("right(Bx)")
        analysis_tasks[boundary_type].add_task("right(By)")
        analysis_tasks[boundary_type].add_task("right(Bz)")
        analysis_tasks[boundary_type].add_task("right(Jx)")
        analysis_tasks[boundary_type].add_task("right(Jy)")
        analysis_tasks[boundary_type].add_task("right(Jz)")
        analysis_tasks[boundary_type].add_task("right(Ex)")
        analysis_tasks[boundary_type].add_task("right(Ey)")
        analysis_tasks[boundary_type].add_task("right(Ez)")
        analysis_tasks[boundary_type].add_task("left(Bx)")
        analysis_tasks[boundary_type].add_task("left(By)")
        analysis_tasks[boundary_type].add_task("left(Bz)")
        analysis_tasks[boundary_type].add_task("left(Jx)")
        analysis_tasks[boundary_type].add_task("left(Jy)")
        analysis_tasks[boundary_type].add_task("left(Jz)")
        analysis_tasks[boundary_type].add_task("left(Ex)")
        analysis_tasks[boundary_type].add_task("left(Ey)")
        analysis_tasks[boundary_type].add_task("left(Ez)")
 
    return analysis_tasks
