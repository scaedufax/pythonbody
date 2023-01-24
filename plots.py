import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from .nbody import nbody


def gen_x1_x2_and_x1_x3_plots(run: nbody,
                              path: str,
                              scatter_kw_args: dict,
                              base_file_name: str = "",
                              image_ext: str = "png",
                              xlim: tuple = None,
                              ylim: tuple = None,
                              save_neighbour_rho_to_snap: bool = True,
                              force_neighbour_rho_recalc: bool = False,
                              n_procs: int = None,
                              ):
    """
    generates X-Y and X-Z plots

    :param run: nbody project
    :type run: nbody
    :param path: where to store the the output plots
    :type path: str
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
    :param save_neighbour_rho_to_snap: As Neighbour_rho needs to be calculated
        anyway here, this can be used to save them to the snap files!
    :type save_neighbour_rho_to_snap: bool
    """

    print(run.snap.snap_data.index.values)

    n_procs = mp.cpu_count() if n_procs is None else n_procs

    with mp.Manager() as manager:
        lock = manager.Lock()
        func = partial(_gen_x1_x2_and_x1_x3_plot,
                       lock=lock,
                       run=run,
                       path=path,
                       scatter_kw_args=scatter_kw_args,
                       base_file_name=base_file_name,
                       image_ext=image_ext,
                       xlim=xlim,
                       ylim=ylim,
                       save_neighbour_rho_to_snap=save_neighbour_rho_to_snap,
                       force_neighbour_rho_recalc=force_neighbour_rho_recalc)
        with mp.Pool(processes=n_procs) as pool:
            _total = run.snap.snap_data.index.values.shape[0]
            with tqdm(total=_total) as pbar:
                for _ in pool.imap_unordered(func, run.snap.snap_data.index.values):
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
                              base_file_name: str = "",
                              image_ext: str = "png",
                              xlim: tuple = None,
                              ylim: tuple = None,
                              save_neighbour_rho_to_snap: bool = True,
                              force_neighbour_rho_recalc: bool = False,
                              ):
    run.snap.load_cluster(time)

    if (("NEIGHBOUR_RHO_M" not in run.snap.cluster_data.columns)
        or ("NEIGHBOUR_RHO_N" not in run.snap.cluster_data.columns)
        or force_neighbour_rho_recalc):

        if lock is not None:
            lock.acquire()
            try:
                run.snap.cluster_data.calc_NEIGHBOUR_RHO()

                if save_neighbour_rho_to_snap:
                    run.snap.save_cols({"NEIGHBOUR_RHO_N": "PNB_CD_NEIGHBOUR_RHO_N",
                                        "NEIGHBOUR_RHO_M": "PNB_CD_NEIGHBOUR_RHO_M"})
            finally:
                lock.release()
    
    run.snap.cluster_data.sort_values("NEIGHBOUR_RHO_M",
                                      ignore_index=True,
                                      inplace=True,
                                      ascending=True)

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=(6.4*2, 6.4))
    
    axes[0].scatter(run.snap.cluster_data["X1"],
                    run.snap.cluster_data["X2"],
                    c=np.log10(run.snap.cluster_data["NEIGHBOUR_RHO_M"]),
                    **scatter_kw_args)
    axes[1].scatter(run.snap.cluster_data["X1"],
                    run.snap.cluster_data["X3"],
                    c=np.log10(run.snap.cluster_data["NEIGHBOUR_RHO_M"]),
                    **scatter_kw_args)

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

    fig.suptitle(f"Time = {time}")
    fig.tight_layout()

    max_time = run.snap.snap_data.index.max()
    zero_pad = int(np.ceil(np.log10(max_time))) + 2
    fig.savefig(f"{path}/{base_file_name}_{str(time).zfill(zero_pad)}.{image_ext}")
    plt.close()
