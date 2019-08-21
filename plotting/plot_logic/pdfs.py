import os
import logging
from collections import OrderedDict
from sys import stdout
from sys import path

import numpy as np
import h5py
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 9})

from dedalus.tools.parallel import Sync

path.insert(0, './plot_logic')
from plot_logic.file_reader import SingleFiletypePlotter
from plot_logic.plot_grid import PlotGrid

logger = logging.getLogger(__name__.split('.')[-1])


class PdfPlotter(SingleFiletypePlotter):
    """
    A class for plotting probability distributions of a dedalus output.
    PDF plots are currently only implemented for 2D slices. When one axis is
    represented by polynomials that exist on an uneven basis (e.g., Chebyshev),
    that basis is evenly interpolated to avoid skewing of the distribution by
    uneven grid sampling.

    Additional Attributes:
    -----------
    pdfs : OrderedDict
        Contains PDF data (x, y, dx)
    pdf_stats : OrderedDict
        Contains scalar stats for the PDFS (mean, stdev, skew, kurtosis)
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the PDF plotter.

        Attributes:
        -----------
        *args, **kwargs : Additional keyword arguments for super().__init__() 
        """
        super(PdfPlotter, self).__init__(*args, distribution='even', **kwargs)
        self.pdfs = OrderedDict()
        self.pdf_stats = OrderedDict()


    def _calculate_pdf_statistics(self):
        """ Calculate statistics of the PDFs stored in self.pdfs. Store results in self.pdf_stats. """
        for k, data in self.pdfs.items():
            pdf, x_vals, dx = data

            mean = np.sum(x_vals*pdf*dx)
            stdev = np.sqrt(np.sum((x_vals-mean)**2*pdf*dx))
            skew = np.sum((x_vals-mean)**3*pdf*dx)/stdev**3
            kurt = np.sum((x_vals-mean)**4*pdf*dx)/stdev**4
            self.pdf_stats[k] = (mean, stdev, skew, kurt)

    
    def _get_interpolated_slices(self, pdf_list, bases=['x', 'z'], uneven_basis=None):
        """
        For data on an uneven grid, interpolates that data on to an evenly spaced grid.

        Attributes:
        ----------
        pdf_list : list
            list of strings of the Dedalus tasks to make PDFs of.
        bases : list, optional
            The names of the Dedalus bases on which the data exists
        uneven_basis : string, optional
            The basis on which the grid has uneven spacing.
        """
        with self.my_sync:
            if self.idle: return
            #Read data
            tasks = []
            for i, f in enumerate(self.files):
                if self.reader.comm.rank == 0:
                    print('reading file {}/{}...'.format(i+1, len(self.files)))
                    stdout.flush()
                bs, tsk, writenum, times = self.reader.read_file(f, bases=bases, tasks=pdf_list)
                tasks.append(tsk)
                if i == 0:
                    total_shape = list(tsk[pdf_list[0]].shape)
                else:
                    total_shape[0] += tsk[pdf_list[0]].shape[0]

            # Put data on an even grid
            x, y = bs[bases[0]], bs[bases[1]]
            if bases[0] == uneven_basis:
                even_x = np.linspace(x.min(), x.max(), len(x))
                even_y = y
            elif bases[1] == uneven_basis:
                even_x = x
                even_y = np.linspace(y.min(), y.max(), len(y))
            else:
                even_x, even_y = x, y
            eyy, exx = np.meshgrid(even_y, even_x)

            full_data = OrderedDict()
            for k in pdf_list: full_data[k] = np.zeros(total_shape)
            count = 0
            for i in range(len(tasks)):
                for j in range(tasks[i][pdf_list[0]].shape[0]):
                    for k in pdf_list:
                        interp = RegularGridInterpolator((x.flatten(), y.flatten()), tasks[i][k][j,:], method='linear')
                        full_data[k][count,:] = interp((exx, eyy))
                    count += 1
            return full_data


    def calculate_pdfs(self, pdf_list, bins=100, **kwargs):
        """
        Calculate probability distribution functions of the specified tasks.

        Arguments:
        ----------
        pdf_list : list
            The names of the tasks to create PDFs of
        bins : int, optional
            The number of bins the PDF should have
        **kwargs : additional keyword arguments for the self._get_interpolated_slices() function.
        """
        with self.my_sync:
            if self.idle : return

            full_data = self._get_interpolated_slices(pdf_list, **kwargs)

            # Create histograms of data
            bounds = OrderedDict()
            minv, maxv = np.zeros(1), np.zeros(1)
            buffmin, buffmax = np.zeros(1), np.zeros(1)
            for k in pdf_list:
                minv[0] = np.min(full_data[k])
                maxv[0] = np.max(full_data[k])
                self.dist_comm.Allreduce(minv, buffmin, op=MPI.MIN)
                self.dist_comm.Allreduce(maxv, buffmax, op=MPI.MAX)
                bounds[k] = (np.copy(buffmin[0]), np.copy(buffmax[0]))
                buffmin *= 0
                buffmax *= 0

                loc_hist, bin_edges = np.histogram(full_data[k], bins=bins, range=bounds[k])
                loc_hist = np.array(loc_hist, dtype=np.float64)
                global_hist = np.zeros_like(loc_hist, dtype=np.float64)
                self.dist_comm.Allreduce(loc_hist, global_hist, op=MPI.SUM)
                local_counts, global_counts = np.zeros(1), np.zeros(1)
                local_counts[0] = np.prod(full_data[k].shape)
                self.dist_comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

                dx = bin_edges[1]-bin_edges[0]
                x_vals = bin_edges[:-1] + dx/2
                pdf = global_hist/global_counts/dx
                self.pdfs[k] = (pdf, x_vals, dx)
            self._calculate_pdf_statistics()
        

    def plot_pdfs(self, dpi=150, **kwargs):
        """
        Plot the probability distribution functions and save them to file.

        Arguments:
        ----------
        dpi : int, optional
            Pixel density of output image.
        **kwargs : additional keyword arguments for PlotGrid()
        """
        with self.my_sync:
            if self.reader.comm.rank != 0: return

            grid = PlotGrid(1,1, **kwargs)
            ax = grid.axes['ax_0-0']
            
            for k, data in self.pdfs.items():
                pdf, xs, dx = data
                mean, stdev, skew, kurt = self.pdf_stats[k]
                title = r'$\mu$ = {:.2g}, $\sigma$ = {:.2g}, skew = {:.2g}, kurt = {:.2g}'.format(mean, stdev, skew, kurt)
                ax.set_title(title)
                ax.axvline(mean, c='orange')

                ax.plot(xs, pdf, lw=2, c='k')
                ax.fill_between((mean-stdev, mean+stdev), pdf.min(), pdf.max(), color='orange', alpha=0.5)
                ax.fill_between(xs, 1e-16, pdf, color='k', alpha=0.5)
                ax.set_xlim(xs.min(), xs.max())
                ax.set_ylim(pdf.min(), pdf.max())
                ax.set_yscale('log')
                ax.set_xlabel(k)
                ax.set_ylabel('P({:s})'.format(k))

                grid.fig.savefig('{:s}/{:s}_pdf.png'.format(self.out_dir, k), dpi=dpi, bbox_inches='tight')
                ax.clear()

            self._save_pdfs()

    def _save_pdfs(self):
        """ 
        Save PDFs to file. For each PDF, e.g., 'entropy' and 'w', the file will have a dataset with:
            xs  - the x-values of the PDF
            pdf - the (normalized) y-values of the PDF
            dx  - the spacing between x values, for use in integrals.
        """
        if self.reader.comm.rank == 0:
            with h5py.File('{:s}/pdf_data.h5'.format(self.out_dir), 'w') as f:
                for k, data in self.pdfs.items():
                    pdf, xs, dx = data
                    this_group = f.create_group(k)
                    for d, n in ((pdf, 'pdf'), (xs, 'xs')):
                        dset = this_group.create_dataset(name=n, shape=d.shape, dtype=np.float64)
                        f['{:s}/{:s}'.format(k, n)][:] = d
                    dset = this_group.create_dataset(name='dx', shape=(1,), dtype=np.float64)
                    f['{:s}/dx'.format(k)][0] = dx

