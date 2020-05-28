import sys
import os
from collections import OrderedDict
from mpi4py import MPI

def construct_BC_dict(args_dict, default_T_BC='TT', default_u_BC='FS', default_M_BC='MC'):
    """
    Given a parsed docstring from docopt, construct a dictionary of bools to set boundary conditions.

    Tracked boundary conditions are:
    - TT --- Fixed temperature at the top and bottom
    - FT --- Fixed flux (bottom) / fixed temp (top)
    - FF --- Fixed flux at the top and bottom
    - NS --- No-slip boundaries at the top and bottom
    - FS --- Free-slip boundaries at the top and bottom
    - MC --- Magnetically-conducting boundary conditions
    - MI --- Magnetically-insulating boundary conditions.

    # Arguments
        args_dict (dict) :
            The argument dictionary parsed from the simulation docstring
        default_T_BC (str, optional) :
            The default temperature BC to use if none are specified in args_dict
        default_u_BC (str, optional) :
            The default velocity BC to use if none are specified in args_dict
        default_M_BC (str, optional) :
            The default magnetic BC to use if none are specified in args_dict

    # Returns
        OrderedDict[bool] :
            A dictionary of bools specifying which boundary conditions are being used.
    """
    boundary_conditions = OrderedDict()
    T_keys = ['TT', 'FT', 'FF']
    u_keys = ['NS', 'FS']
    M_keys = ['MC', 'MI']
    bc_lists    = (T_keys, u_keys, M_keys)
    bc_defaults = (default_T_BC, default_u_BC, default_M_BC)

    # Set options specified by user
    all_keys = T_keys + u_keys + M_keys
    for k in all_keys:
        boundary_conditions[k] = False
        input_k = '--{}'.format(k)
        if input_k in args_dict.keys():
            boundary_conditions[k] = args_dict[input_k]

    # Fill in default boundary conditions when not specified.
    for keys, default in zip(bc_lists, bc_defaults):
        if default is None:
            continue
        bc_specified = False
        for k in keys:
            if boundary_conditions[k]:
                bc_specific = True
                break
        if not bc_specified:
            boundary_conditions[default] = True

    return boundary_conditions

def construct_out_dir(args_dict, bc_dict, base_flags=['3D', 'Ra', 'Pr', 'a'], label_flags=['Nu_ICs', 'AE', 'TT_to_FT'], resolution_flags=['nx', 'nz'], parent_dir_flag='root_dir'):
    """
    Make a unique case-based string and directory for easily identifiable dedalus output

    # Arguments
        args_dict (dict) :
            The argument dictionary parsed from the simulation docstring
        bc_dict (dict) :
            A dictionary of booleans for each possible boundary condition in the problem 
        base_flags (list, optional) :
            A list of strings of default docopt flags to include if flagged
        label_flags (list, optional ) :
            A list of strings of docopt flags to add to the label at the end of the directory
        resolution_flags (list, optional) :
            A lsit of strings of docopt flags that specify the coefficient resolution of the simulation
        parent_dir_flag (str, optional) :
            The name of the flag that points to the parent directory where the output directory will be created.

    # Returns
        str :
           the output directory name 
    """
    data_dir = args_dict['--{}'.format(parent_dir_flag)] + '/' + sys.argv[0].split('.py')[0]

    #Base inputs
    for f in base_flags:
        flag = '--{}'.format(f)
        if type(args_dict[flag]) is bool:   
            if args_dict[flag]:
                data_dir += '_{}'.format(f) 
        elif args_dict[flag] is not None:
            data_dir += '_{}{}'.format(f, args_dict[flag]) 

    #Boundary conditions
    for k, val in bc_dict.items():
        if val:
            data_dir += '_{}'.format(k)

    #Resolution
    data_dir += '_'
    for f in resolution_flags:
        flag = '--{}'.format(f)
        if args_dict[flag]: 
            data_dir += '{}x'.format(args_dict[flag])
    data_dir = data_dir[:-1]

    #Labels / special cases
    for f in label_flags:
        flag = '--{}'.format(f)
        if type(args_dict[flag]) is bool:   
            if args_dict[flag]:
                if args_dict['--label'] is not None:
                    args_dict['--label'] += '_{}'.format(f) 
                else:
                    args_dict['--label'] = '{}'.format(f)
        elif args_dict[flag] is not None:
            if args_dict['--label'] is not None:
                args_dict['--label'] += '_{}{}'.format(f, args_dict[flag]) 
            else:
                args_dict['--label'] = '{}{}'.format(f, args_dict[flag])
            
    if args_dict['--label'] is not None:
        data_dir += "_{}".format(args_dict['--label'])
    data_dir += '/'
    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.makedirs('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
    return data_dir

