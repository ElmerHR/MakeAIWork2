'''
====== Legal notices

Copyright (C) 2013 - 2021 GEATEC engineering

This program is free software.
You can use, redistribute and/or modify it, but only under the terms stated in the QQuickLicense.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY, without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the QQuickLicense for details.

The QQuickLicense can be accessed at: http://www.qquick.org/license.html

__________________________________________________________________________


 THIS PROGRAM IS FUNDAMENTALLY UNSUITABLE FOR CONTROLLING REAL SYSTEMS !!

__________________________________________________________________________

It is meant for training purposes only.

Removing this header ends your license.
'''

import socket as sc
import time as tm

import simpylc as sp
import socket_wrapper as sw

class ControlServer:
    def __init__ (self):
        with sc.socket (*sw.socketType) as serverSocket:
            serverSocket.bind (sw.address)
            serverSocket.listen (sw.maxNrOfConnectionRequests)

            while True:
                self.clientSocket = serverSocket.accept ()[0]
                self.socketWrapper = sw.SocketWrapper (self.clientSocket)

                with self.clientSocket:
                    while True:
                        sensors = {
                            'courseAngle': sp.eva (sp.world.vessel.courseAngle),
                            'vaneAngle': sp.eva (sp.world.vessel.vaneAngle),
                            'lattitude': sp.eva (sp.world.vessel.lattitude),
                            'longitude': sp.eva (sp.world.vessel.longitude)
                        }

                        self.socketWrapper.send (sensors)
                        tm.sleep (0.02)
                        actuators = self.socketWrapper.recv ()
                        sp.world.vessel.rudderAngle.set (actuators ['rudderAngle'])
                        sp.world.vessel.sheetLength.set (actuators ['sheetLength'])

