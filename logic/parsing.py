from collections import OrderedDict

def construct_BC_dict(args_dict, default_T_BC='TT', default_u_BC='FS', default_M_BC='MC'):
    """
    Given a parsed docstring from docopt, construct a dictionary of bools to set boundary conditions.

    Tracked boundary conditions are:
    - TT : Fixed temperature at the top and bottom
    - FT : Fixed flux (bottom) / fixed temp (top)
    - FF : Fixed flux at the top and bottom
    - NS : No-slip boundaries at the top and bottom
    - FS : Free-slip boundaries at the top and bottom
    - MC : Magnetically-conducting boundary conditions
    - MI : Magnetically-insulating boundary conditions.

    Arguments:
        args_dict (dict) :
            The argument dictionary parsed from the simulation docstring
        default_T_BC (str, optional) :
            The default temperature BC to use if none are specified in args_dict
        default_u_BC (str, optional) :
            The default velocity BC to use if none are specified in args_dict
        default_M_BC (str, optional) :
            The default magnetic BC to use if none are specified in args_dict

    Returns:
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
