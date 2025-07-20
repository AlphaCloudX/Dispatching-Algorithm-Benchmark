import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Global Seaborn theme for modern aesthetics
sns.set_theme(style="whitegrid", font_scale=1.2, rc={
    'axes.edgecolor': '0.8',
    'axes.linewidth': 0.8,
    'axes.facecolor': '#F9F9F9',
    'figure.facecolor': 'white'
})


def rounded_rects(ax, bars, radius=0.2):
    # Applies rounded corners to bar containers
    for bar in bars:
        bar.set_linewidth(0)
        bar.set_edgecolor('none')
        bar.set_path_effects([])  # Remove outlines
        bar.set_clip_on(False)
        bar.set_zorder(3)

        # Create rounded rectangle path
        from matplotlib.patches import FancyBboxPatch
        bbox = bar.get_bbox()
        fancy = FancyBboxPatch((bbox.x0, bbox.y0),
                               bbox.width, bbox.height,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               linewidth=0, facecolor=bar.get_facecolor())
        ax.add_patch(fancy)
        bar.remove()


def plot_driver_status_bar(stats):
    statuses = ['Idle', 'En Routed', 'On Location', 'Towing']
    palette = sns.color_palette("Set2", len(statuses))

    final_driver_infos = stats[-1]['driver_infos']
    final_driver_infos.sort(key=lambda x: (x['truck_type'], x['driver_id']))
    driver_labels = [f"{info['driver_id']} ({info['truck_type']})" for info in final_driver_infos]
    data = np.array([info['call_statistics'] for info in final_driver_infos])

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(final_driver_infos))

    for i, status in enumerate(statuses):
        bars = ax.bar(driver_labels, data[:, i], bottom=bottom, label=status, color=palette[i])
        rounded_rects(ax, bars, radius=0.2)
        bottom += data[:, i]

    ax.set_xlabel("Driver ID (Truck Type)")
    ax.set_ylabel("Total Ticks")
    ax.set_title("Total Time per Driver in Each Status", weight='bold', fontsize=16)
    ax.legend(title="Status", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_call_idle_times_horizontal_bar(stats):
    aggregated = {}
    for rec in stats:
        call_idle_times_by_truck_type = rec.get('call_idle_times_by_truck_type', {})
        for truck_type, idle_times in call_idle_times_by_truck_type.items():
            aggregated.setdefault(truck_type, []).extend(idle_times)

    sorted_truck_types = sorted(aggregated.keys())
    means = [np.mean(aggregated[t]) for t in sorted_truck_types]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_truck_types, means, color=sns.color_palette("pastel"))

    rounded_rects(ax, bars, radius=0.2)

    ax.set_xlabel("Average Call Idle Time (Ticks)")
    ax.set_ylabel("Required Truck Type")
    ax.set_title("Average Call Idle Time per Required Truck Type", weight='bold', fontsize=16)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


def plot_call_wait_vs_available_drivers(stats):
    ticks = [rec['tick'] for rec in stats]
    calls_waiting = [rec['calls_waiting_on_drivers'] for rec in stats]
    available_drivers = [rec['driver_load']['idle'] for rec in stats]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.lineplot(x=ticks, y=calls_waiting, ax=ax1, color='tomato', label='Calls Waiting on Drivers')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Calls Waiting on Drivers', color='tomato')
    ax1.tick_params(axis='y', labelcolor='tomato')

    ax2 = ax1.twinx()
    sns.lineplot(x=ticks, y=available_drivers, ax=ax2, color='steelblue', label='Available Drivers')
    ax2.set_ylabel('Available Drivers', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')

    plt.title('Calls Waiting vs Available Drivers Over Time', weight='bold', fontsize=16)
    fig.tight_layout()
    plt.show()
