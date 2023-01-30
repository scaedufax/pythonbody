import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

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
                              trace_particle_j: int = None,
                              n_procs: int = None,
                              ):
    """
    generates X-Y and X-Z plots

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
    :param trace_particle_j: trace particle by index J used in Nbody
    :type trace_particle_j: int
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
                       trace_particle_j=trace_particle_j)
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


    """for time in tqdm(run.snap.snap_data.index.values):
        _gen_x1_x2_and_x1_x3_plot(time,
                                  run, 
                                  path,
                                  scatter_kw_args,
                                  base_file_name,
                                  image_ext,
                                  xlim,
                                  ylim,
                                  save_neighbour_rho_to_snap,
                                  )"""


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
                              trace_particle_j: int = None,
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
    axes[0].scatter(run.snap.cluster_data["X1"],
                    run.snap.cluster_data["X2"],
                    c=np.log10(run.snap.cluster_data["NEIGHBOUR_RHO_M"]),
                    **scatter_kw_args)
    axes[1].scatter(run.snap.cluster_data["X1"],
                    run.snap.cluster_data["X3"],
                    c=np.log10(run.snap.cluster_data["NEIGHBOUR_RHO_M"]),
                    **scatter_kw_args)
    
    if (trace_particle_j is not None 
        and run.snap.cluster_data.index.shape[0] >= trace_particle_j):

        axes[0].scatter(run.snap.cluster_data.loc[trace_particle_j - 1, "X1"],
                        run.snap.cluster_data.loc[trace_particle_j - 1, "X2"],
                        c="r")
        axes[1].scatter(run.snap.cluster_data.loc[trace_particle_j - 1, "X1"],
                        run.snap.cluster_data.loc[trace_particle_j - 1, "X3"],
                        c="r")

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

    for i, ax in enumerate(axes):
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
        plot_title = run_title if i < 2 else ref_run_title
        ax.set_title(f"{'X1-X2' if i % 2 == 0 else 'X1-X3'} {plot_title + ' ' if plot_title is not None else ''}at time = {time if i < 2 else time * ref_run_time_scale}")
        

    #fig.suptitle(f"Time = {time}")
    fig.tight_layout()

    max_time = run.snap.snap_data.index.max()
    zero_pad = int(np.ceil(np.log10(max_time))) + 2
    fig.savefig(f"{path}/{base_file_name}_{str(time).zfill(zero_pad)}.{image_ext}")
    plt.close()
