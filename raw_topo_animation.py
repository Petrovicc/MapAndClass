import copy
import itertools
from functools import partial
from numbers import Integral
import warnings
import numpy as np
from mne.viz.topomap import _adjust_meg_sphere, _fnirs_types, _average_fnirs_overlaps, _check_extrapolate, _init_anim, \
    _animate, _pause_anim, _key_press

from ..baseline import rescale
from ..channels.channels import _get_ch_type
from ..channels.layout import (
    _find_topomap_coords, find_layout, _pair_grad_sensors, _merge_ch_data)
from ..defaults import _EXTRAPOLATE_DEFAULT, _BORDER_DEFAULT
from ..io.pick import (pick_types, _picks_by_type, pick_info, pick_channels,
                       _pick_data_channels, _picks_to_idx, _get_channel_types,
                       _MEG_CH_TYPES_SPLIT)
from ..utils import (_clean_names, _time_mask, verbose, logger, fill_doc,
                     _validate_type, _check_sphere, _check_option, _is_numeric)
from .utils import (tight_layout, _setup_vmin_vmax, _prepare_trellis,
                    _check_delayed_ssp, _draw_proj_checkbox, figure_nobar,
                    plt_show, _process_times, DraggableColorbar,
                    _validate_if_list_of_axes, _setup_cmap, _check_time_unit)
from ..time_frequency import psd_multitaper
from ..defaults import _handle_default
from ..transforms import apply_trans, invert_transform
from ..io.meas_info import Info, _simplify_info
from ..io.proj import Projection


def animate_raw_topomap(self, ch_type=None, times=None, frame_rate=None,
                    butterfly=False, blit=True, show=True, time_unit='s',
                    sphere=None, *, extrapolate=_EXTRAPOLATE_DEFAULT,
                    verbose=None):
    """Make animation of raw data as topomap timeseries.
    The animation can be paused/resumed with left mouse button.
    Left and right arrow keys can be used to move backward or forward
    in time.
    Parameters
    ----------
    ch_type : str | None
        Channel type to plot. Accepted data types: 'mag', 'grad', 'eeg',
        'hbo', 'hbr', 'fnirs_cw_amplitude',
        'fnirs_fd_ac_amplitude', 'fnirs_fd_phase', and 'fnirs_od'.
        If None, first available channel type from the above list is used.
        Defaults to None.
    times : array of float | None
        The time points to plot. If None, 10 evenly spaced samples are
        calculated over the evoked time series. Defaults to None.
    frame_rate : int | None
        Frame rate for the animation in Hz. If None,
        frame rate = sfreq / 10. Defaults to None.
    butterfly : bool
        Whether to plot the data as butterfly plot under the topomap.
        Defaults to False.
    blit : bool
        Whether to use blit to optimize drawing. In general, it is
        recommended to use blit in combination with ``show=True``. If you
        intend to save the animation it is better to disable blit.
        Defaults to True.
    show : bool
        Whether to show the animation. Defaults to True.
    time_unit : str
        The units for the time axis, can be "ms" (default in 0.16)
        or "s" (will become the default in 0.17).
        .. versionadded:: 0.16
    %(topomap_sphere_auto)s
    %(topomap_extrapolate)s
        .. versionadded:: 0.22
    %(verbose_meth)s
    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure.
    anim : instance of matplotlib.animation.FuncAnimation
        Animation of the topomap.
    Notes
    -----
    .. versionadded:: 0.12.0
    """
    return _topomap_animation(
        self, ch_type=ch_type, times=times, frame_rate=frame_rate,
        butterfly=butterfly, blit=blit, show=show, time_unit=time_unit,
        sphere=sphere, extrapolate=extrapolate, verbose=verbose)


def _prepare_topomap_plot(inst, ch_type, sphere=None):
    """Prepare topo plot."""
    info = copy.deepcopy(inst if isinstance(inst, Info) else inst.info)
    sphere, clip_origin = _adjust_meg_sphere(sphere, info, ch_type)

    clean_ch_names = _clean_names(info['ch_names'])
    for ii, this_ch in enumerate(info['chs']):
        this_ch['ch_name'] = clean_ch_names[ii]
    info['bads'] = _clean_names(info['bads'])
    for comp in info['comps']:
        comp['data']['col_names'] = _clean_names(comp['data']['col_names'])

    info._update_redundant()
    info._check_consistency()

    # special case for merging grad channels
    layout = find_layout(info)
    if (ch_type == 'grad' and layout is not None and
            (layout.kind.startswith('Vectorview') or
             layout.kind.startswith('Neuromag_122'))):
        picks, _ = _pair_grad_sensors(info, layout)
        pos = _find_topomap_coords(info, picks[::2], sphere=sphere)
        merge_channels = True
    elif ch_type in _fnirs_types:
        # fNIRS data commonly has overlapping channels, so deal with separately
        picks, pos, merge_channels, overlapping_channels = \
            _average_fnirs_overlaps(info, ch_type, sphere)
    else:
        merge_channels = False
        if ch_type == 'eeg':
            picks = pick_types(info, meg=False, eeg=True, ref_meg=False,
                               exclude='bads')
        elif ch_type == 'csd':
            picks = pick_types(info, meg=False, csd=True, ref_meg=False,
                               exclude='bads')
        elif ch_type == 'dbs':
            picks = pick_types(info, meg=False, dbs=True, ref_meg=False,
                               exclude='bads')
        elif ch_type == 'seeg':
            picks = pick_types(info, meg=False, seeg=True, ref_meg=False,
                               exclude='bads')
        else:
            picks = pick_types(info, meg=ch_type, ref_meg=False,
                               exclude='bads')

        if len(picks) == 0:
            raise ValueError("No channels of type %r" % ch_type)

        pos = _find_topomap_coords(info, picks, sphere=sphere)

    ch_names = [info['ch_names'][k] for k in picks]
    if ch_type in _fnirs_types:
        # Remove the chroma label type for cleaner labeling.
        ch_names = [k[:-4] for k in ch_names]

    if merge_channels:
        if ch_type == 'grad':
            # change names so that vectorview combined grads appear as MEG014x
            # instead of MEG0142 or MEG0143 which are the 2 planar grads.
            ch_names = [ch_names[k][:-1] + 'x' for k in
                        range(0, len(ch_names), 2)]
        else:
            assert ch_type in _fnirs_types
            # Modify the nirs channel names to indicate they are to be merged
            # New names will have the form  S1_D1xS2_D2
            # More than two channels can overlap and be merged
            for set in overlapping_channels:
                idx = ch_names.index(set[0][:-4])
                new_name = 'x'.join(s[:-4] for s in set)
                ch_names[idx] = new_name

    pos = np.array(pos)[:, :2]  # 2D plot, otherwise interpolation bugs
    return picks, pos, merge_channels, ch_names, ch_type, sphere, clip_origin


def _topomap_animation(evoked, ch_type, times, frame_rate, butterfly, blit,
                       show, time_unit, sphere, extrapolate, *, verbose=None):
    """Make animation of evoked data as topomap timeseries.
    See mne.evoked.Evoked.animate_topomap.
    """
    from matplotlib import pyplot as plt, animation
    if ch_type is None:
        ch_type = _picks_by_type(evoked.info)[0][0]
    if ch_type not in ('mag', 'grad', 'eeg',
                       'hbo', 'hbr', 'fnirs_od', 'fnirs_cw_amplitude'):
        raise ValueError("Channel type not supported. Supported channel "
                         "types include 'mag', 'grad', 'eeg'. 'hbo', 'hbr', "
                         "'fnirs_cw_amplitude', and 'fnirs_od'.")
    time_unit, _ = _check_time_unit(time_unit, evoked.times)
    if times is None:
        times = np.linspace(evoked.times[0], evoked.times[-1], 10)
    times = np.array(times)

    if times.ndim != 1:
        raise ValueError('times must be 1D, got %d dimensions' % times.ndim)
    if max(times) > evoked.times[-1] or min(times) < evoked.times[0]:
        raise ValueError('All times must be inside the evoked time series.')
    frames = [np.abs(evoked.times - time).argmin() for time in times]

    picks, pos, merge_channels, _, ch_type, sphere, clip_origin = \
        _prepare_topomap_plot(evoked, ch_type, sphere=sphere)
    data = evoked.data[picks, :]
    data *= _handle_default('scalings')[ch_type]

    fig = plt.figure(figsize=(6, 5))
    shape = (8, 12)
    colspan = shape[1] - 1
    rowspan = shape[0] - bool(butterfly)
    ax = plt.subplot2grid(shape, (0, 0), rowspan=rowspan, colspan=colspan)
    if butterfly:
        ax_line = plt.subplot2grid(shape, (rowspan, 0), colspan=colspan)
    else:
        ax_line = None
    if isinstance(frames, Integral):
        frames = np.linspace(0, len(evoked.times) - 1, frames).astype(int)
    ax_cbar = plt.subplot2grid(shape, (0, colspan), rowspan=rowspan)
    ax_cbar.set_title(_handle_default('units')[ch_type], fontsize=10)
    extrapolate = _check_extrapolate(extrapolate, ch_type)

    params = dict(data=data, pos=pos, all_times=evoked.times, frame=0,
                  frames=frames, butterfly=butterfly, blit=blit,
                  pause=False, times=times, time_unit=time_unit,
                  clip_origin=clip_origin)
    init_func = partial(_init_anim, ax=ax, ax_cbar=ax_cbar, ax_line=ax_line,
                        params=params, merge_channels=merge_channels,
                        sphere=sphere, ch_type=ch_type,
                        extrapolate=extrapolate, verbose=verbose)
    animate_func = partial(_animate, ax=ax, ax_line=ax_line, params=params)
    pause_func = partial(_pause_anim, params=params)
    fig.canvas.mpl_connect('button_press_event', pause_func)
    key_press_func = partial(_key_press, params=params)
    fig.canvas.mpl_connect('key_press_event', key_press_func)
    if frame_rate is None:
        frame_rate = evoked.info['sfreq'] / 10.
    interval = 1000 / frame_rate  # interval is in ms
    anim = animation.FuncAnimation(fig, animate_func, init_func=init_func,
                                   frames=len(frames), interval=interval,
                                   blit=blit)
    fig.mne_animation = anim  # to make sure anim is not garbage collected
    plt_show(show, block=False)
    if 'line' in params:
        # Finally remove the vertical line so it does not appear in saved fig.
        params['line'].remove()

    return fig, anim


