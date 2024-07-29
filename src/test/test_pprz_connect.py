import time

import pprz_connect
from pprzlink.ivy import IvyMessagesInterface

if __name__ == '__main__':
    """
    test program
    """
    if 0:
        ivy_interface = IvyMessagesInterface("d2d pprz_backend")
        connect =  pprz_connect.PprzConnect(verbose=True, ivy=ivy_interface)
    else:
        connect = pprz_connect.PprzConnect(verbose=True)
        ivy_interface = connect.ivy
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping on request")

    connect.shutdown()
