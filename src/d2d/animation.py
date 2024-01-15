
#
# Animations
#
import numpy as np
import matplotlib.pyplot as plt    
import matplotlib.animation as animation
import matplotlib.image, matplotlib.offsetbox, matplotlib.transforms

def animate(time, Xs=None, U=None, Yrefs=None, Xref=None, Extra=None, title='Animation', _drawings=False, _imgs=True, figure=None, ax=None, extends=None):
    if extends is None: extends = (0, 100, 0, 100)
    _xmin, _xmax, _ymin, _ymax = extends
    time_factor = 1. # Nope :(2.
    _decim = int(4*time_factor) # sim at 100hz, I want anim at 25 fps
    fig = figure or plt.figure(figsize=(10., 8.))
    if ax is None:
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(_xmin, _xmax),
                             ylim=(_ymin, _ymax), facecolor=(0.5, 0.9, 0.9))
    if title is not None:
        ax.set_title(title, {'fontsize': 20 })
        fig.canvas.set_window_title(title)
    ax.grid()
    time_template = 'time = {:0.1f}s'
    time_text = ax.text(0.025, 0.92, 'Hello', transform=ax.transAxes)
    marker_acs, marker_refs, marker_extras = [], [], []
    if Yrefs is not None:
        for i, Yref in enumerate(Yrefs):
            _track_ref,  = ax.plot(Yref[:,0,0], Yref[:,0,1], linestyle='dashed', lw=1., label=f'reference {i}', alpha=0.5)
            _marker_ref, = ax.plot([], [], '.-', lw=3, zorder=1, label='_ref' , color=_track_ref.get_color())
            marker_refs.append(_marker_ref)
    ax.set_prop_cycle(None)
    if Xs is not None:
        for i, X in enumerate(Xs):
            _track_ac, = ax.plot(X[:,0], X[:,1], linestyle='dotted', lw=1., label=f'aircraft {i}', alpha=0.5)
            _line_ac, = ax.plot([], [], '-', lw=3, zorder=1, label='_ac', color = _track_ac.get_color())
            marker_acs.append(_line_ac)
    if Extra is not None:
        for i, X in enumerate(Extra): 
            _line_marker, = ax.plot([], [], '.', lw=3, zorder=1, label='extra')
            marker_extras.append(_line_marker)
           
    ax.legend(loc='best')

    def init():
        for _l in marker_acs:  _l.set_data([], [])
        for _l in marker_refs:  _l.set_data([], [])
        for _l in marker_extras:  _l.set_data([], [])
        return [time_text] + marker_acs + marker_refs + marker_extras

    def _get_points(x,y,psi,l=0.5):
        _c = np.array([x, y])                             # center of gravity
        _b = l*np.array([np.cos(psi), np.sin(psi)])       # front facing unit vect
        _p1 = _c + _b                                     # front
        _p2 = _c - _b                                     # back
        return [_p1[0], _p2[0]], [_p1[1], _p2[1]]

    def _get_points2(x,y): return [[x]], [[y]]

        
    def animate(i):
        if Xs is not None:
            for _l, X in zip(marker_acs, Xs):
                x, y, psi, phi, v = X[int(i*_decim), :]
                pts = _get_points(x,y,psi,l=0.5)
                _l.set_data(pts)
        if Yrefs is not None:
            for _l, Yref in zip(marker_refs, Yrefs):
                xr, yr = Yref[int(i*_decim), :][0]
                prefs = _get_points2(xr, yr)
                _l.set_data(prefs)
        if Extra is not None:
            for _l, extra in zip(marker_extras, Extra):
                xr, yr = extra[int(i*_decim)]
                _l.set_data(_get_points2(xr, yr))
            
        time_text.set_text(time_template.format(i*_decim * dt))
        return [time_text] + marker_acs + marker_refs + marker_extras

    dt = time[1]-time[0]
    dt_mili = dt*1000*_decim
    anim = animation.FuncAnimation(fig, animate, np.arange(1, len(time)/_decim),
                                   interval=dt_mili, blit=True, init_func=init, repeat_delay=200)

    return anim

def save_anim(filename, an, dt):
    print('encoding animation video, please wait, it will take a while')
    _start = time.time()
    fps = 1./dt/4; print(f'dt {dt} fps {fps}')
    an.save(filename, writer=animation.PillowWriter(fps=1./dt/4)) # gif?
    _end = time.time()
    print(f'video encoded, saved to {filename}, Bye (took {_end-_start:.1f} s)')


