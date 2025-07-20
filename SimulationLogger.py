from Utils import CallStatus


class SimulationLogger:
    def __init__(self):
        self.data = []

    def record(self, tick, drivers, calls):
        load_counts = {
            'idle': 0,
            'en_routed': 0,
            'on_location': 0,
            'towing': 0
        }
        total_idle_time = 0
        total_completed_calls = 0

        driver_infos = []

        for driver in drivers:
            status_name = driver.current_status.name.lower()
            load_counts[status_name] += 1
            total_idle_time += driver.call_statistics[CallStatus.IDLE.value]
            total_completed_calls += len(driver.completed_calls)

            driver_infos.append({
                'driver_id': driver.driver_id,
                'truck_type': driver.truck_type,
                'call_statistics': driver.call_statistics.copy()
            })

        total_call_queue_time = sum(call.call_statistics[0] for call in calls)

        calls_waiting_on_drivers = 0
        call_idle_times_by_truck_type = {}

        for call in calls:
            if call.current_status == CallStatus.IDLE:
                compatible_drivers = [d for d in drivers if call.can_truck_complete_job(d.truck_type)]
                available_compatible_drivers = [d for d in compatible_drivers if d.is_available()]
                if len(available_compatible_drivers) == 0:
                    calls_waiting_on_drivers += 1

            # Log idle times by required truck type(s)
            idle_time = call.call_statistics[0]
            for truck_type in call.required_truck_type:
                if truck_type not in call_idle_times_by_truck_type:
                    call_idle_times_by_truck_type[truck_type] = []
                call_idle_times_by_truck_type[truck_type].append(idle_time)

        record = {
            'tick': tick,
            'calls_remaining': len(calls),
            'calls_waiting_on_drivers': calls_waiting_on_drivers,
            'driver_load': load_counts,
            'total_idle_driver_time': total_idle_time,
            'total_completed_calls': total_completed_calls,
            'total_call_queue_time': total_call_queue_time,
            'driver_infos': driver_infos,
            'call_idle_times_by_truck_type': call_idle_times_by_truck_type  # NEW
        }

        self.data.append(record)

    def export_json(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def export_csv(self, filepath):
        import csv
        if not self.data:
            return

        keys = list(self.data[0].keys())
        # Flatten driver_load dictionary keys as well
        driver_load_keys = self.data[0]['driver_load'].keys()
        keys.remove('driver_load')
        keys.extend([f'driver_load_{k}' for k in driver_load_keys])

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for record in self.data:
                flat_record = record.copy()
                # Flatten driver_load dict
                for k in driver_load_keys:
                    flat_record[f'driver_load_{k}'] = flat_record['driver_load'][k]
                del flat_record['driver_load']
                writer.writerow(flat_record)
