import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import sys

from .nbody import nbody

DEBUG = True


def gen_x1_x2_and_x1_x3_plots(run: nbody,
                              ref_run: nbody = None,
                              ref_run_time_scale: float = 1.,
                              path: str = "./",
                              run_title: str = None,
                              ref_run_title: str = None,
                              scatter_kw_args: dict = {},
                              base_file_name: str = "",
                              image_ext: str = "png",
                              xlim: tuple = None,
                              ylim: tuple = None,
                              n_neigh: int = 80,
                              save_neighbour_rho_to_snap: bool = True,
                              force_neighbour_rho_recalc: bool = False,
                              trace_particle_name: int = None,
                              n_procs: int = None,
                              ):
    """
    generates X-Y and X-Z plots. Calculates `RHO` using `n_neigh` and uses it
    for color representation

    :param run: nbody project
    :type run: nbody
    :param ref_run: adds a reference plot below the main one
    :type ref_run: nbody
    :param ref_run_time_scale: Scale the time to be loaded for the ref run by this factor
    :type ref_run_time_scale: float
    :param path: where to store the the output plots
    :type path: str
    :param run_title: Title to put above plots referring to the run
    :type run_title: str
    :param ref_run_title: Title to put above plots referring to the reference run
    :type ref_run_title: str
    :param scatter_kw_args: arguments to pass to scatter
    :type scatter_kw_args: dict
    :param base_file_name: Prepend to file name
    :type base_file_name: str
    :param image_ext: extension/format to use for images. See
        matplotlib.pyplot.savefig for available formats.
    :type image_ext: str
    :param xlim: set x-range for scatter plot
    :type xlim: tuple
    :param ylim: set y-range for scatter plot
    :type ylim: tuple
    :param n_neigh: use `n_neigh` neighbour for `NEIGHBOUR_RHO` calculation
    :param save_neighbour_rho_to_snap: As Neighbour_rho needs to be calculated
        anyway here, this can be used to save them to the snap files!
    :type save_neighbour_rho_to_snap: bool
    :param force_neighbour_rho_recalc: Force recalculation of neighbour_rho data
    :type force_neighbour_rho_recalc: bool
    :param trace_particle_name: trace particle by NAME used in Nbody
    :type trace_particle_name: int
    :param n_procs: number of processes to use during multiprocessing
    :type n_procs: int
    """
    
    # get number of processors if nothing was passend
    if n_procs is None:
        n_procs = mp.cpu_count()
        # leave 1 cpu idle (or 2 cpus if there are more than ten!)
        if n_procs > 2:
            n_procs -= 1
        if n_procs > 10:
            n_procs -= 1

    with mp.Manager() as manager:
        # Initialise lock and function
        lock = manager.Lock()
        func = partial(_gen_x1_x2_and_x1_x3_plot,
                       lock=lock,
                       run=run,
                       ref_run=ref_run,
                       ref_run_time_scale=ref_run_time_scale,
                       path=path,
                       run_title=run_title,
                       ref_run_title=ref_run_title,
                       scatter_kw_args=scatter_kw_args,
                       base_file_name=base_file_name,
                       image_ext=image_ext,
                       xlim=xlim,
                       ylim=ylim,
                       n_neigh=n_neigh,
                       save_neighbour_rho_to_snap=save_neighbour_rho_to_snap,
                       force_neighbour_rho_recalc=force_neighbour_rho_recalc,
                       trace_particle_name=trace_particle_name)
        # start pool
        with mp.Pool(processes=n_procs) as pool:
            # make sure to have nice tqdm output
            max_time = run.snap.snap_data.index.values.max()
            if ref_run is not None:
                max_time = int(min(max_time, ref_run.snap.snap_data.index.values.max()/ref_run_time_scale))
            mask = run.snap.snap_data.index.values <= max_time
            _total = mask.sum()
            with tqdm(total=_total) as pbar:
                for _ in pool.imap_unordered(func, run.snap.snap_data.index.values[mask]):
                    pbar.update()

def _gen_x1_x2_and_x1_x3_plot(time: float,
                              run: nbody,
                              lock: mp.Lock,
                              path: str,
                              scatter_kw_args: dict,
                              ref_run: nbody = None,
                              ref_run_time_scale: int = 1,
                              run_title: str = None,
                              ref_run_title: str = None,
                              base_file_name: str = "",
                              image_ext: str = "png",
                              xlim: tuple = None,
                              ylim: tuple = None,
                              n_neigh: int = 80,
                              save_neighbour_rho_to_snap: bool = True,
                              force_neighbour_rho_recalc: bool = False,
                              trace_particle_name: int = None,
                              ):

    # load time step (also in ref cluster!)
    run.snap.load_cluster(time)
    if ref_run is not None:
        ref_run.snap.load_cluster(time*ref_run_time_scale)

    if (("NEIGHBOUR_RHO_M" not in run.snap.cluster_data.columns)
        or ("NEIGHBOUR_RHO_N" not in run.snap.cluster_data.columns)
        or force_neighbour_rho_recalc):

        if lock is not None:
            lock.acquire()
            try:
                run.snap.cluster_data.calc_NEIGHBOUR_RHO(n_neigh=n_neigh)
                if DEBUG:
                    print(f"NEIGHBOUR_RHO_M at time = {time}: "
                          f"mean: {np.log10(run.snap.cluster_data['NEIGHBOUR_RHO_M']).mean():.03f} +- "
                          f"{np.log10(run.snap.cluster_data['NEIGHBOUR_RHO_M']).std():.03f} "
                          f"min: {np.log10(run.snap.cluster_data['NEIGHBOUR_RHO_M']).min():.03f} "
                          f"max: {np.log10(run.snap.cluster_data['NEIGHBOUR_RHO_M']).max():.03f} "
                          )
                    print(f"NEIGHBOUR_RHO_N at time = {time}: "
                          f"mean: {np.log10(run.snap.cluster_data['NEIGHBOUR_RHO_N']).mean():.03f} +- "
                          f"{np.log10(run.snap.cluster_data['NEIGHBOUR_RHO_N']).std():.03f} "
                          f"min: {np.log10(run.snap.cluster_data['NEIGHBOUR_RHO_N']).min():.03f} "
                          f"max: {np.log10(run.snap.cluster_data['NEIGHBOUR_RHO_N']).max():.03f} "
                          )
                if save_neighbour_rho_to_snap:
                    run.snap.save_cols({"NEIGHBOUR_RHO_N": "PNB_CD_NEIGHBOUR_RHO_N",
                                        "NEIGHBOUR_RHO_M": "PNB_CD_NEIGHBOUR_RHO_M"})
            finally:
                lock.release()
    
    if (ref_run is not None
        and (("NEIGHBOUR_RHO_M" not in ref_run.snap.cluster_data.columns)
        or ("NEIGHBOUR_RHO_N" not in ref_run.snap.cluster_data.columns)
        or force_neighbour_rho_recalc)):

        if lock is not None:
            lock.acquire()
            try:
                ref_run.snap.cluster_data.calc_NEIGHBOUR_RHO(n_neigh=n_neigh)
                if DEBUG:
                    print(f"ref NEIGHBOUR_RHO_M at time = {time}: "
                          f"mean: {np.log10(ref_run.snap.cluster_data['NEIGHBOUR_RHO_M']).mean():.03f} +- "
                          f"{np.log10(ref_run.snap.cluster_data['NEIGHBOUR_RHO_M']).std():.03f} "
                          f"min: {np.log10(ref_run.snap.cluster_data['NEIGHBOUR_RHO_M']).min():.03f} "
                          f"max: {np.log10(ref_run.snap.cluster_data['NEIGHBOUR_RHO_M']).max():.03f} "
                          )
                    print(f"ref NEIGHBOUR_RHO_N at time = {time}: "
                          f"mean: {np.log10(ref_run.snap.cluster_data['NEIGHBOUR_RHO_N']).mean():.03f} +- "
                          f"{np.log10(ref_run.snap.cluster_data['NEIGHBOUR_RHO_N']).std():.03f} "
                          f"min: {np.log10(ref_run.snap.cluster_data['NEIGHBOUR_RHO_N']).min():.03f} "
                          f"max: {np.log10(ref_run.snap.cluster_data['NEIGHBOUR_RHO_N']).max():.03f} "
                          )
                if save_neighbour_rho_to_snap:
                    ref_run.snap.save_cols({"NEIGHBOUR_RHO_N": "PNB_CD_NEIGHBOUR_RHO_N",
                                            "NEIGHBOUR_RHO_M": "PNB_CD_NEIGHBOUR_RHO_M"})
            finally:
                lock.release()
    
    run.snap.cluster_data.sort_values("NEIGHBOUR_RHO_M",
                                      ignore_index=True,
                                      inplace=True,
                                      ascending=True)
    if ref_run is not None:
        ref_run.snap.cluster_data.sort_values("NEIGHBOUR_RHO_M",
                                              ignore_index=True,
                                              inplace=True,
                                              ascending=True)

    fig, axes = plt.subplots(nrows=1 if ref_run is None else 2,
                             ncols=2,
                             figsize=(6.4*2, 6.4 if ref_run is None else 6.4*2))
    axes = axes.flatten()
    pcm = axes[0].scatter(run.snap.cluster_data["X1"],
                          run.snap.cluster_data["X2"],
                          c=np.log10(run.snap.cluster_data["NEIGHBOUR_RHO_M"]),
                          **scatter_kw_args)
    axes[1].scatter(run.snap.cluster_data["X1"],
                    run.snap.cluster_data["X3"],
                    c=np.log10(run.snap.cluster_data["NEIGHBOUR_RHO_M"]),
                    **scatter_kw_args)
    
    if (trace_particle_name is not None):
        if (trace_particle_name in run.snap.cluster_data["NAME"].values):
            idx = run.snap.cluster_data[run.snap.cluster_data["NAME"] == trace_particle_name].index.values[0]

            axes[0].scatter(run.snap.cluster_data.loc[idx, "X1"],
                            run.snap.cluster_data.loc[idx, "X2"],
                            c="r")
            axes[1].scatter(run.snap.cluster_data.loc[idx, "X1"],
                            run.snap.cluster_data.loc[idx, "X3"],
                            c="r")
        else:
            print(f"Couldn't find particle with name \"{trace_particle_name}\" "
                  f"in particle Name list", file=sys.stderr)
            if DEBUG:
                print(run.snap.cluster_data["NAME"].values, file=sys.stderr)


    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")
    axes[1].set_xlabel("X1")
    axes[1].set_ylabel("X3")
    if xlim is not None:
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
        axes[1].set_ylim(*ylim)

    if ref_run is not None:
        axes[2].scatter(ref_run.snap.cluster_data["X1"],
                        ref_run.snap.cluster_data["X2"],
                        c=np.log10(ref_run.snap.cluster_data["NEIGHBOUR_RHO_M"]),
                        **scatter_kw_args)
        axes[3].scatter(ref_run.snap.cluster_data["X1"],
                        ref_run.snap.cluster_data["X3"],
                        c=np.log10(ref_run.snap.cluster_data["NEIGHBOUR_RHO_M"]),
                        **scatter_kw_args)

        axes[2].set_xlabel("X1")
        axes[2].set_ylabel("X2")
        axes[3].set_xlabel("X1")
        axes[3].set_ylabel("X3")
        if xlim is not None:
            axes[2].set_xlim(*xlim)
            axes[3].set_xlim(*xlim)
        if ylim is not None:
            axes[2].set_ylim(*ylim)
            axes[3].set_ylim(*ylim)
    
    fig.tight_layout()
    fig.colorbar(pcm, ax=axes, shrink=0.6, label=r"$\log_{10}(\rho)$")

    for i, ax in enumerate(axes):
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plot_title = run_title if i < 2 else ref_run_title
        ax.set_title(f"{'X1-X2' if i % 2 == 0 else 'X1-X3'} {plot_title + ' ' if plot_title is not None else ''}at time = {time if i < 2 else time * ref_run_time_scale}")
        

    #fig.suptitle(f"Time = {time}")

    max_time = run.snap.snap_data.index.max()
    zero_pad = int(np.ceil(np.log10(max_time))) + 2
    fig.savefig(f"{path}/{base_file_name}_{str(time).zfill(zero_pad)}.{image_ext}")
    plt.close()


def gen_x1_x2_and_x1_x3_nbody_rho_plots(run: nbody,
                                        path: str = "./",
                                        xylims: list = ["full"],
                                        run_title: str = None,
                                        scatter_kw_args: dict = {},
                                        base_file_name: str = "",
                                        image_ext: str = "png",
                                        trace_particle_name: int = None,
                                        n_procs: int = None,
                                        ):
    """
    generates X-Y and X-Z plots. Uses `RHO` calculated from nbody in the
    conf.3 files for colour coding!

    :param run: nbody project
    :type run: nbody
    :param path: where to store the the output plots
    :type path: str
    :param xylims: plot for different regions around the center of mass.
        `full` means plotting all data.
    :type xylim: set containing tuple(floats) or "full"
    :param run_title: Title to put above plots referring to the run
    :type run_title: str
    :param scatter_kw_args: arguments to pass to scatter
    :type scatter_kw_args: dict
    :param base_file_name: Prepend to file name
    :type base_file_name: str
    :param image_ext: extension/format to use for images. See
        matplotlib.pyplot.savefig for available formats.
    :type image_ext: str
    :param trace_particle_name: trace particle by NAME used in Nbody
    :type trace_particle_name: int
    :param n_procs: number of processes to use during multiprocessing
    :type n_procs: int

    """
    
    # get number of processors if nothing was passend
    if n_procs is None:
        n_procs = mp.cpu_count()
        # leave 1 cpu idle (or 2 cpus if there are more than ten!)
        if n_procs > 2:
            n_procs -= 1
        if n_procs > 10:
            n_procs -= 1

    # Initialise function
    func = partial(_gen_x1_x2_and_x1_x3_nbody_rho_plot,
                   run=run,
                   path=path,
                   xylims=xylims,
                   run_title=run_title,
                   scatter_kw_args=scatter_kw_args,
                   base_file_name=base_file_name,
                   image_ext=image_ext,
                   trace_particle_name=trace_particle_name)
    # start pool
    with mp.Pool(processes=n_procs) as pool:
        # make sure to have nice tqdm output
        _total = run.conf.files.shape[0]
        with tqdm(total=_total) as pbar:
            for _ in pool.imap_unordered(func, run.conf.files.index.values):
                pbar.update()


def _gen_x1_x2_and_x1_x3_nbody_rho_plot(time: float,
                                        run: nbody,
                                        path: str,
                                        xylims: list,
                                        scatter_kw_args: dict,
                                        run_title: str,
                                        base_file_name: str,
                                        image_ext: str,
                                        trace_particle_name: int
                                        ):
    # Prepare Data
    run.conf.load(time)
    
    if DEBUG:
        print(f"RHO mean: {run.conf.data['RHO'].mean():.03f}; "
              f"median: {run.conf.data['RHO'].median():.03f}; "
              f"std: {run.conf.data['RHO'].std():.03f}; "
              f"min: {run.conf.data['RHO'].min():.03f}; "
              f"max: {run.conf.data['RHO'].max():.03f}; "
              )

    # Insert RHO from conf file according to particle NAME
    #  set 0 to 1e-3 otherwise we'll later get errors when using log
    run.conf.data.loc[run.conf.data["RHO"] == 0, "RHO"] = np.full(np.sum(run.conf.data["RHO"] == 0), 1e-3)


    # set plots
    nrows = len(xylims)
    fig, axes = plt.subplots(nrows=nrows,
                             ncols=2,
                             figsize=(6.4*2, 6.4*nrows)
                             )
    pcm = None
    # do X1-X2 and X1-X3 plots
    for i, ax in enumerate(axes.flatten()):
        row = int(i/2)
        if i % 2 == 0:
            pcm = axes[row][0].scatter(run.conf.data["X1"], 
                                       run.conf.data["X2"],
                                       c=np.log10(run.conf.data["RHO"]),
                                       **scatter_kw_args)
            ax.set_title(f"{run_title} X1-X2 at time = {time} NB")
            ax.set_ylabel("X2")
        else:
            axes[row][1].scatter(run.conf.data["X1"], 
                                 run.conf.data["X3"],
                                 c=np.log10(run.conf.data["RHO"]),
                                 **scatter_kw_args)
            ax.set_title(f"{run_title} X1-X3 at time = {time} NB")
            ax.set_ylabel("X3")
        
        ax.set_xlabel("X1")
   
        # plot traced particle
        if (trace_particle_name is not None):
            if (trace_particle_name in run.conf.data["NAME"].values):
                idx = run.conf.data[run.conf.data["NAME"] == trace_particle_name].index.values[0]

                axes[row][0].scatter(run.conf.data.loc[idx, "X1"],
                                     run.conf.data.loc[idx, "X2"],
                                     c="r")
                axes[row][1].scatter(run.conf.data.loc[idx, "X1"],
                                     run.conf.data.loc[idx, "X3"],
                                     c="r") 
            else:
                print(f"TIME = {time} Couldn't find particle with name \"{trace_particle_name}\" "
                      f"in particle Name list", file=sys.stderr)
                if DEBUG:
                    print(run.conf.data["NAME"].values, file=sys.stderr)
        
        # set X and Y lims on plots
        if xylims[row] == "full":
            max_value = max(run.conf.data["X1"].max(),
                            run.conf.data["X2"].max(),
                            run.conf.data["X3"].max(),
                            -run.conf.data["X1"].min(),
                            -run.conf.data["X2"].min(),
                            -run.conf.data["X3"].min(),
                            )
            ax.set_xlim(-max_value, max_value)
            ax.set_ylim(-max_value, max_value)
        elif type(xylims[row]) == tuple and all([type(i) in [int, float] for i in xylims[row]]):
            ax.set_xlim(xylims[row][0], xylims[row][1])
            ax.set_ylim(xylims[row][0], xylims[row][1])
        elif type(xylims[row]) == tuple and all([type(i) in [tuple, list] for i in xylims[row]]):
            if len(xylims[row]) == 2:
                ax.set_xlim(*xylims[row][0])
                ax.set_ylim(*xylims[row][1])
            elif len(xylims[row]) == 4:
                ax.set_xlim(*xylims[row][0 + ((i % 2) * 2)])
                ax.set_ylim(*xylims[row][1 + ((i % 2) * 2)])

        elif type(xylims[row]) in [int, float]:
            ax.set_xlim(-xylims[row], xylims[row])
            ax.set_ylim(-xylims[row], xylims[row])

    fig.tight_layout()
    fig.colorbar(pcm, ax=axes, shrink=0.4,label=r"$\log_{10}(\rho)$")
    for i, ax in enumerate(axes.flatten()):
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    max_time = run.conf.files.index.max()
    zero_pad = int(np.ceil(np.log10(max_time))) + 2
    fig.savefig(f"{path}/{base_file_name}_{str(time).zfill(zero_pad)}.{image_ext}")
    plt.close()

