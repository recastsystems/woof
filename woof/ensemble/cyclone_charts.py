"""Time-series analysis charts from the native renderer's track receipt."""
from datetime import datetime
import json
from pathlib import Path


def intensity_chart(root, *, experiment_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as dates
    import matplotlib.pyplot as plt
    root = Path(root)
    data = json.loads((root/'intensity-tracks.json').read_text())
    members = data['members']
    parse = lambda t: datetime.fromisoformat(t.replace('Z', '+00:00'))
    times = [parse(f['valid_time']) for f in members[0]['fixes']]
    issue, landfall = parse(data['issue_time']), parse(data['landfall_reference'])
    lead = (landfall-issue).total_seconds()/3600
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout='constrained')
    fig.get_layout_engine().set(rect=(0, .035, 1, .965))
    for ax, key, label in zip(axes, ('mslp_hpa', 'max_wind_m_s'),
                              ('Minimum pressure (hPa)', 'Maximum instantaneous 10 m wind (m/s)')):
        columns = [[f[key] for f in m['fixes']] for m in members]
        weights = [m['weight'] for m in members]
        for member, column in zip(members, columns):
            ax.plot(times, column, color=member['color'], alpha=.65, linewidth=.8)
        ax.fill_between(times, [min(c[i] for c in columns) for i in range(len(times))],
                        [max(c[i] for c in columns) for i in range(len(times))],
                        color='gray', alpha=.2, label='Ensemble range')
        ax.plot(times, [sum(w*c[i] for w, c in zip(weights, columns)) for i in range(len(times))],
                color='navy', linewidth=2, label='Ancestry-weighted mean')
        # Include the neighboring synoptic fixes. A three-hour proof can lie
        # entirely between them; the dashed line is explicitly interpolation,
        # rather than silently omitting the verification reference.
        before = [f for f in data['best_track'] if parse(f['valid_time']) < times[0]]
        after = [f for f in data['best_track'] if parse(f['valid_time']) > times[-1]]
        best = (before[-1:] +
                [f for f in data['best_track'] if times[0] <= parse(f['valid_time']) <= times[-1]] +
                after[:1])
        if best:
            ax.plot([parse(f['valid_time']) for f in best], [f[key] for f in best],
                    color='black', marker='o', linestyle='--',
                    label='NHC verification (linear between fixes)')
        if times[0] <= landfall <= times[-1]:
            ax.axvline(landfall, color='black', linestyle=':', label=landfall.strftime('%H:%MZ landfall'))
        ax.set_ylabel(label)
        ax.set_xlim(times[0], times[-1])
        ax.grid(alpha=.25)
        ax.legend(loc='best', fontsize=8)
    axes[-1].xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%MZ'))
    axes[-1].set_xlabel('UTC valid time')
    fig.suptitle(f"{experiment_label}: intensity evolution\n"
                 f"Analysis {data['analysis_time']} | cutoff {data['observation_cutoff']}\n"
                 f"Issued {data['issue_time']} | remaining lead {lead:.2f} h (retrospective)", fontsize=11)
    footer = 'Model instantaneous winds are not NHC one-minute sustained winds.'
    if times[-1] < landfall:
        footer += ' This short proof does not reach landfall.'
    fig.text(.5, .005, footer, ha='center', fontsize=8)
    day = times[-1].strftime('%Y-%m-%d')
    out = root/'d02'/'intensity-evolution'/day/'intensity-evolution.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
