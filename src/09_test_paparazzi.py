#!/usr/bin/env python3

import time
from timeit import default_timer as timer

import d2d.paparazzi_backend as d2pb

import d2d.trajectory as d2traj


def main():
    backend = d2pb.PprzBackend()
    traj = d2traj.TrajectoryLine((0,0), (200, 200))
    #traj = d2traj.TrajectoryCircle( c=[30., 30.],  r=30., v=12.)
    start = timer()
    while True:
            time.sleep(0.5)
            # your code...
            now = timer(); elapsed = now - start
            try:
                backend.send_command(ac_id=12, phic=12.5, vac=15.)
                backend.publish_track(traj, elapsed)
            except KeyboardInterrupt:
                backend.shutdown()

if __name__ == '__main__':
    main()



    
