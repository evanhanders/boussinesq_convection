"""
Script for plotting probability distribution functions of 2D quantities.

Usage:
    plot_pdfs.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Output directory for figures [default: pdfs]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 25]
    --dpi=<dpi>                         Image pixel density [default: 150]
    --bins=<bins>                       Number of bins per pdf [default: 200]
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
args = docopt(__doc__)
from plot_logic.pdfs import PdfPlotter

root_dir = args['<root_dir>']
fig_name  = args['--fig_name']
start_file = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: n_files = int(n_files)



# Load in figures and make plots
if '3D' in root_dir:
    threeD = True
    bases  = ['x', 'y', 'z']
    pdfs_to_plot = ['T', 'wT']
    plotter = PdfPlotter(root_dir, file_dir='volumes', fig_name=fig_name, start_file=start_file, n_files=n_files)


    bases_2  = ['x', 'y']
    pdfs_to_plot2 = ['T near top', 'T near bot 1']
    plotter2 = PdfPlotter(root_dir, file_dir='slices', fig_name=fig_name+'_xy', start_file=start_file, n_files=n_files)


    bases_3  = ['x', 'z']
    pdfs_to_plot3 = ['T']
    plotter3 = PdfPlotter(root_dir, file_dir='slices', fig_name=fig_name+'_xz', start_file=start_file, n_files=n_files)
else:
    threeD = False
    bases  = ['x', 'z']
    pdfs_to_plot = ['T', 'enstrophy', 'enth_flux', 'w']
    plotter = PdfPlotter(root_dir, file_dir='slices', fig_name=fig_name, start_file=start_file, n_files=n_files)


plotter.calculate_pdfs(pdfs_to_plot, bins=int(args['--bins']), threeD=threeD, bases=bases, uneven_basis='z')
plotter.plot_pdfs(dpi=int(args['--dpi']), row_in=5, col_in=8.5)

if threeD:
    for bases, plotter, pdfs_to_plot in zip([bases_2, bases_3], [plotter2, plotter3], [pdfs_to_plot2, pdfs_to_plot3]):
        plotter.calculate_pdfs(pdfs_to_plot, bins=int(args['--bins']), threeD=False, bases=bases, uneven_basis='z')
        plotter.plot_pdfs(dpi=int(args['--dpi']), row_in=5, col_in=8.5)



