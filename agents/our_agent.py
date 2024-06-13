from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track

def get_entry_point():
    '''
    Used to automatically instantiate our agent
    '''
    return 'OurAgent'

class OurAgent (AutonomousAgent):
    def __init__(self) -> None:
        pass

    def setup (self, path_to_cfg_file):
        '''
        - Do all the initializations needed by our agent
        - Will be called each time a route is instantiated
        - Arguments can include an addtional configuration file for the agent, the parsing of that file must be handled by us
        '''
        self.track = Track.SENSORS  # At a minimum, this method sets the Leaderboard modality. In this case, SENSORS

    def sensors (self):
        sensors = []

        # add camera
        sensors.append({    # following TCP
                        'type': 'sensor.camera.rgb',
                        'x': -1.5, 'y': 0.0, 'z':2.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': 900, 'height': 256, 'fov': 100,
                        'id': 'rgb'
                    })
            
        # add imu
        sensors.append({
                        'type': 'sensor.other.imu',
                        'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'sensor_tick': 0.05,    # 20 times / sec
                        'id': 'imu'
                    })

        # add gnss
        sensors.append({
                        'type': 'sensor.other.gnss',
                        'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'sensor_tick': 0.01,    # 100 times / sec
                        'id': 'gps'
                    })

        # add speedometer
        sensors.append({
                        'type': 'sensor.speedometer',
                        'reading_frequency': 20,
                        'id': 'speed'  
                    })


