from Dispatcher import Dispatcher
from Environment import generate_traffic, generate_calls, generate_drivers
from Utils import load_config_file
from VisualizeStats import plot_driver_status_bar, plot_call_wait_vs_available_drivers, \
    plot_call_idle_times_horizontal_bar

if __name__ == '__main__':
    config = load_config_file('config.yaml')

    # Create Blank Grid of 1's
    x, y = config['grid_length'], config['grid_width']
    traffic_distribution = config['traffic_distribution']
    grid = generate_traffic(x, y, traffic_distribution)

    calls_to_run = generate_calls(x, y, config['calls'], config['ticks_to_run'], config['calls']['number_of_calls'])
    # print(calls_to_run)

    driver_list = generate_drivers(config)
    # print(driver_list)

    # Initialize dispatcher
    dispatcher = Dispatcher(driver_list, calls_to_run, grid)
    dispatcher.run_until_complete()

    dispatcher.stats_collector.export_json('simulation_stats.json')
    dispatcher.stats_collector.export_csv('simulation_stats.csv')

    stats = dispatcher.stats_collector.data  # your collected per-tick stats

    plot_driver_status_bar(stats)
    plot_call_idle_times_horizontal_bar(stats)
    plot_call_wait_vs_available_drivers(stats)
