# View.py
# Kebing li
# 02/27/2018
# CS251 Project3

import numpy as np
import math
import copy


class View:

    def __init__(self):
        self.reset()

    # let the user reset the view
    # assign default values
    def reset(self):
        self.vrp = np.matrix([0.5, 0.5, 1])
        self.vpn = np.matrix([0, 0, -1])
        self.vup = np.matrix([0, 1, 0])
        self.u = np.matrix([[-1, 0, 0]])
        self.extent = [1.0, 1.0, 1.0]
        self.screen = [400.0, 400.0]
        self.offset = [20.0, 20.0]

    # uses the current viewing parameters to return a view matrix
    def build(self):
        vtm = np.identity(4, float)

        # translation matrix to move the vrp to the origin
        t1 = np.matrix([[1, 0, 0, -self.vrp[0, 0]],
                        [0, 1, 0, -self.vrp[0, 1]],
                        [0, 0, 1, -self.vrp[0, 2]],
                        [0, 0, 0, 1]])
        vtm = t1 * vtm

        tu = np.cross(self.vup, self.vpn)
        tvup = np.cross(self.vpn, tu)
        tvpn = self.vpn.copy()
        tu = self.normalize(tu.tolist()[0])
        tvup = self.normalize(tvup.tolist()[0])
        tvpn = self.normalize(tvpn.tolist()[0])
        self.u = tu.copy()
        self.vup = tvup.copy()
        self.vpn = tvpn.copy()

        # rotation matrix to align the axes
        r1 = np.matrix([[tu[0, 0], tu[0, 1], tu[0, 2], 0.0],
                        [tvup[0, 0], tvup[0, 1], tvup[0, 2], 0.0],
                        [tvpn[0, 0], tvpn[0, 1], tvpn[0, 2], 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        vtm = r1 * vtm

        # translate the lower left corner of the view space to the origin
        t2 = np.matrix([[1, 0, 0, 0.5 * self.extent[0]],
                        [0, 1, 0, 0.5 * self.extent[1]],
                        [0, 0, 1, 0.0],
                        [0, 0, 0, 1]])
        vtm = t2 * vtm

        # use the extent and screen size values to scale to the screen
        s1 = np.matrix([[-self.screen[0] / self.extent[0], 0.0, 0.0, 0.0],
                        [0.0, -self.screen[1] / self.extent[1], 0.0, 0.0],
                        [0.0, 0.0, 1.0 / self.extent[2], 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        vtm = s1 * vtm

        # translate the lower left corner to the origin and add the view offset
        t3 = np.matrix([[1, 0, 0, self.screen[0] + self.offset[0]],
                        [0, 1, 0, self.screen[1] + self.offset[1]],
                        [0, 0, 1, 0.0],
                        [0, 0, 0, 1]])
        vtm = t3 * vtm

        return vtm

    # the normalize method that takes in a vector and returns a normalized vector
    def normalize(self, V):
        Vx = V[0]
        Vy = V[1]
        Vz = V[2]
        Vnorm = [0, 0, 0]
        length = math.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)
        Vnorm[0] = Vx / length
        Vnorm[1] = Vy / length
        Vnorm[2] = Vz / length
        Vnorm = np.matrix(Vnorm)

        return Vnorm

    # makes and returns a duplicate view object
    def clone(self):
        view = View()
        view.vrp = self.vrp.copy()
        view.vpn = self.vpn.copy()
        view.vup = self.vup.copy()
        view.u = self.u.copy()
        view.extent = copy.copy(self.extent)
        view.screen = copy.copy(self.screen)
        view.offset = copy.copy(self.offset)

        return view

    # rotates about the center of the view volume.
    # a1 is how much to rotate about the vup axis and a2 is how much to rotate about the u axis
    def rotateVRC(self, a1, a2):
        # Make a translation matrix to move the point ( VRP + VPN * extent[Z] * 0.5 ) to the origin
        t1 = np.matrix([[1, 0, 0, -self.vrp[0, 0] - self.extent[2] / 2 * self.vpn[0, 0]],
                        [0, 1, 0, -self.vrp[0, 1] - self.extent[2] / 2 * self.vpn[0, 1]],
                        [0, 0, 1, -self.vrp[0, 2] - self.extent[2] / 2 * self.vpn[0, 2]],
                        [0, 0, 0, 1]])

        # Make an axis alignment matrix Rxyz using u, vup and vpn.
        Rxyz = np.matrix([[self.u[0, 0], self.u[0, 1], self.u[0, 2], 0.0],
                          [self.vup[0, 0], self.vup[0, 1], self.vup[0, 2], 0.0],
                          [self.vpn[0, 0], self.vpn[0, 1], self.vpn[0, 2], 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

        # Make a rotation matrix about the Y axis by the VUP angle
        r1 = np.matrix([[math.cos(a1), 0.0, math.sin(a1), 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-math.sin(a1), 0.0, math.cos(a1), 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

        # Make a rotation matrix about the X axis by the U angle. Put it in r2.
        r2 = np.matrix([[1.0, 0.0, 0.0, 0.0],
                        [0.0, math.cos(a2), -math.sin(a2), 0.0],
                        [0.0, math.sin(a2), math.cos(a2), 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

        # Make a translation matrix that has the opposite translation from step 1.
        t2 = np.matrix([[1, 0, 0, self.vrp[0, 0] + self.extent[2] / 2 * self.vpn[0, 0]],
                        [0, 1, 0, self.vrp[0, 1] + self.extent[2] / 2 * self.vpn[0, 1]],
                        [0, 0, 1, self.vrp[0, 2] + self.extent[2] / 2 * self.vpn[0, 2]],
                        [0, 0, 0, 1]])

        # Make a numpy matrix where the VRP is on the first row, with a 1 in the homogeneous
        # coordinate, and u, vup, and vpn are the next three rows, with a 0 in the homogeneous coordinate.
        tvrc = np.matrix([[self.vrp[0, 0], self.vrp[0, 1], self.vrp[0, 2], 1.0],
                          [self.u[0, 0], self.u[0, 1], self.u[0, 2], 0.0],
                          [self.vup[0, 0], self.vup[0, 1], self.vup[0, 2], 0.0],
                          [self.vpn[0, 0], self.vpn[0, 1], self.vpn[0, 2], 0.0]])

        tvrc = (t2 * Rxyz.T * r2 * r1 * Rxyz * t1 * tvrc.T).T
        self.vrp = np.matrix(self.normalize([tvrc[0, 0], tvrc[0, 1], tvrc[0, 2]])).copy()
        self.u = np.matrix(self.normalize([tvrc[1, 0], tvrc[1, 1], tvrc[1, 2]])).copy()
        self.vup = np.matrix(self.normalize([tvrc[2, 0], tvrc[2, 1], tvrc[2, 2]])).copy()
        self.vpn = np.matrix(self.normalize([tvrc[3, 0], tvrc[3, 1], tvrc[3, 2]])).copy()


if __name__ == "__main__":
    view = View()
    print(view.build())
    clone = view.clone()
    print(clone.build())
