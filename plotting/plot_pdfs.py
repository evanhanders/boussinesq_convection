"""
Script for plotting probability distribution functions of 2D quantities.

Usage:
    plot_pdfs.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Output directory for figures [default: pdfs]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
    --bins=<bins>                       Number of bins per pdf [default: 100]
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

pdfs_to_plot = ['T', 'enstrophy', 'enth_flux', 'w']


# Load in figures and make plots
plotter = PdfPlotter(root_dir, file_dir='slices', fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter.calculate_pdfs(pdfs_to_plot, bins=int(args['--bins']), uneven_basis='z')
plotter.plot_pdfs(dpi=int(args['--dpi']), row_in=5, col_in=8.5)
