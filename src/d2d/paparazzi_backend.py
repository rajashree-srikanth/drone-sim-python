#!/usr/bin/env python3

import sys, time
import numpy as np, matplotlib.pyplot as plt
import json, pyproj

## Paparazzi plumbing
from os import path, getenv
PPRZ_HOME = getenv("PAPARAZZI_HOME", path.normpath(path.join(path.dirname(path.abspath(__file__)), '../../../../')))
PPRZ_SRC = getenv("PAPARAZZI_SRC", path.normpath(path.join(path.dirname(path.abspath(__file__)), '../../../../')))
sys.path.append(PPRZ_HOME + "/var/lib/python/")
sys.path.append(PPRZ_SRC + "/sw/lib/python")

import pprz_connect
import settings
from pprzlink.ivy import IvyMessagesInterface
from pprzlink.message import PprzMessage
###

class PprzAircraft:
    def __init__(self, _id):
        self.id = _id

    def set_local_position(self, x, y):
        self.x, self.y = x, y

    def set_state(self, psi, lat, lon, vel, phi):
        self.psi, self.lat, self.lon, self.vel, self.phi = psi, lat, lon, vel, phi

    def set_utm(self, utm_east, utm_north, utm_zone):
        self.utm_e, self.utm_n, self.utm_z = utm_east, utm_north, utm_zone


    def get_state(self): # returns state as in dynamics.py , ie x, y, psi, phi, va, in standard units
        return self.x, self.y, np.pi/2 - np.deg2rad(self.psi), np.deg2rad(self.phi), self.vel


class PprzBackend:
    def __init__(self):
        if 0: # FIXME : BUG?
            self.ivy_interface = IvyMessagesInterface("d2d pprz_backend")
            self.connect =  pprz_connect.PprzConnect(verbose=True, notify=self._new_ac_cb, ivy=self.ivy_interface)
            #self.connect =  pprz_connect.PprzConnect(verbose=True, ivy=self.ivy_interface)
        else:
            self.connect = pprz_connect.PprzConnect(verbose=False, notify=self._new_ac_cb)
            self.ivy_interface = self.connect.ivy

        
        self.ivy_interface.subscribe(self._nav_msg_cb, PprzMessage("telemetry", "NAVIGATION"))
        self.ivy_interface.subscribe(self._gps_msg_cb, PprzMessage("telemetry", "GPS"))
        self.ivy_interface.subscribe(self._nav_ref_msg_cb, PprzMessage("telemetry", "NAVIGATION_REF"))
        self.ivy_interface.subscribe(self._flight_param_msg_cb, PprzMessage("ground", "FLIGHT_PARAM"))
        self.mngr = {}
        self.aircraft = {}
        self.nav_initialized = False
        self.proj = pyproj.Proj(proj='utm', zone=31, ellps='WGS84', preserve_units=True)
        #self.geod = pyproj.Geod(ellps='WGS84')
        self.display_cnt = 0

    def _nav_ref_msg_cb(self, ac_id, msg):
        self.nav_ref_utm_east, self.nav_ref_utm_north, self.nav_ref_utm_zone, self.nav_ref_ground_alt =\
          float(msg.get_field(0)), float(msg.get_field(1)), int(msg.get_field(2)), float(msg.get_field(3))
        self.nav_initialized = True

    def _gps_msg_cb(self, ac_id, msg):
         #print(f' _gps_msg_cb {type(ac_id)} {msg}')
         # <message name="GPS" id="8">
         # <field name="mode"       type="uint8"  unit="byte_mask"/>
         # <field name="utm_east"   type="int32"  unit="cm" alt_unit="m"/>
         # <field name="utm_north"  type="int32"  unit="cm" alt_unit="m"/>
         # <field name="course"     type="int16"  unit="decideg" alt_unit="deg"/>
         # <field name="alt"        type="int32"  unit="mm" alt_unit="m">Altitude above geoid (MSL)</field>
         # <field name="speed"      type="uint16" unit="cm/s" alt_unit="m/s">norm of 2d ground speed in cm/s</field>
         # <field name="climb"      type="int16"  unit="cm/s" alt_unit="m/s"/>
         # <field name="week"       type="uint16" unit="weeks"/>
         # <field name="itow"       type="uint32" unit="ms"/>
         # <field name="utm_zone"   type="uint8"/>
         # <field name="gps_nb_err" type="uint8"/>
         # </message>
         utm_east, utm_north, utm_zone =  float(msg.get_field(1))/100., float(msg.get_field(2))/100., int(msg.get_field(9))
         try:
             self.aircraft[ac_id].set_utm(utm_east, utm_north, utm_zone)
         except KeyError:
             print(f'gps message from unknown aircraft {ac_id}')

    def _nav_msg_cb(self, ac_id, msg):
        # <message name="NAVIGATION" id="10">
        #   <field name="cur_block" type="uint8"/>
        #   <field name="cur_stage" type="uint8"/>
        #   <field name="pos_x" type="float" unit="m" format="%.1f"/>
        #   <field name="pos_y" type="float" unit="m" format="%.1f"/>
        #   <field name="dist_wp" type="float" format="%.1f" unit="m"/>
        #   <field name="dist_home" type="float" format="%.1f" unit="m"/>
        #   <field name="flight_time" type="uint16" unit="s"/>
        #   <field name="block_time" type="uint16" unit="s"/>
        #   <field name="stage_time" type="uint16" unit="s"/>
        #   <field name="kill_auto_throttle" type="uint8" unit="bool"/>
        #   <field name="circle_count" type="uint8"/>
        #   <field name="oval_count" type="uint8"/>
        # </message>
        #print(f' _nav_msg_cb {type(ac_id)}')# {msg}')
        #print(float(msg.get_field(2)), float(msg.get_field(3)))
        #if msg.name == "NAVIGATION":
            #self.aircraft[self.ids.index(ac_id)].set_position(float(msg.get_field(2)), float(msg.get_field(3)))
            #_ac.x, _ac.y = float(msg.get_field(2)), float(msg.get_field(3))
        x, y = float(msg.get_field(2)), float(msg.get_field(3))
        try:
            self.aircraft[ac_id].set_local_position(x, y)
        except KeyError:
            print(f'nav message from unknown aircraft {ac_id}')

    def _flight_param_msg_cb(self, ac_id, msg):
        # FIXME: ac_id is string ?
        #print(f' _flight_param_msg_cb {ac_id} {type(ac_id)} {msg}')
        ac_id = int(ac_id)
        phi, theta, psi = float(msg.get_field(1)), float(msg.get_field(2)), float(msg.get_field(3))
        lat, lon = float(msg.get_field(4)), float(msg.get_field(5))
        vel, course = float(msg.get_field(6)), float(msg.get_field(7))
        alt, climb = float(msg.get_field(8)), float(msg.get_field(9))
        #print(ac_id, self.aircraft, type(ac_id))
        try:
            self.aircraft[ac_id].set_state(psi, lat, lon, vel, phi)
        except KeyError:
            print(f'flight_param message from unknown aircraft {ac_id}')

    def _new_ac_cb(self, conf): # FIXME - get real filename
        print(f'new ac {conf}')
        settings_xml_path = '/home/poine/paparazzi/var/aircrafts/ZAGI/settings.xml'
        ac_id = 12
        self.mngr[ac_id] = settings.PprzSettingsManager(settings_xml_path, ac_id, ivy=self.ivy_interface)
        self.aircraft[ac_id] = PprzAircraft(ac_id)

    def local_to_wgs(self, x, y):
        u_e, u_n = x + self.nav_ref_utm_east, y + self.nav_ref_utm_north 
        lon, lat = self.proj(u_e, u_n, inverse=True)    
        return lon, lat
        
    def wgs_to_local(lon, lat):
        pass
        
    def publish_track(self, traj, t):
        self.display_cnt += 1
        # https://docs.paparazziuav.org/latest/paparazzi_messages.html#SHAPE
        msg = PprzMessage("ground", "SHAPE")
        msg['text'] = '.'
        if self.nav_initialized:
            # track
            dt = 0.05; ts = np.arange(traj.t0, traj.t0+traj.duration, dt)
            ps = np.array([traj.get(_t)[0] for _t in ts])
            ps_lon_lat = (np.array([self.local_to_wgs(_x, _y) for _x, _y in ps])*1e7).astype(int)
            msg['id'] = 0
            msg['shape'] = 2 
            msg['status'] = 0
            msg['lonarr'] = ps_lon_lat[:,0]
            msg['latarr'] = ps_lon_lat[:,1]
            self.ivy_interface.send(msg)
            # point on track
            p = traj.get(t)[0]
            plonlat = (np.array(self.local_to_wgs(p[0], p[1]))*1e7).astype(int)
            msg['id'] = 1
            msg['shape'] = 0
            msg['status'] = 0
            msg['lonarr'] = [plonlat[0]]
            msg['latarr'] = [plonlat[1]]
            msg['radius'] = 1.
            self.ivy_interface.send(msg)
        else:
            msg['status'] = 1
            msg['id'] = 0
            self.ivy_interface.send(msg)
            msg['id'] = 1
            self.ivy_interface.send(msg)

    def test_shape(self):
        ac = self.aircraft[12]
        print(f'lat lon {ac.lat}, {ac.lon}')
        ue, un = self.proj(ac.lon, ac.lat)
        print(f'ue, un {ue} {ac.utm_e} {un} {ac.utm_n}, {ue-ac.utm_e} {un-ac.utm_n}')
        
        lx, ly = ac.x, ac.y
        print(f'x, y {lx} {ly}')
        try:
            lx1, ly1 = ac.utm_e - self.nav_ref_utm_east, ac.utm_n - self.nav_ref_utm_north
            print(f'x, y {lx1} {ly1}')
            lon1, lat1 = self.local_to_wgs(lx1, ly1)
            print(lon1, lat1)

            msg = PprzMessage("ground", "SHAPE")
            msg['linecolor'] = 'red'
            msg['fillcolor'] = 'yellow'
            msg['opacity'] = 3
            _Circle, _Polygon, _Line = range(3)
            msg['shape'] = _Circle
            msg['status'] = 0
            msg['latarr'] = [int(lat1*1e7)]
            msg['lonarr'] = [int(lon1*1e7)]
            msg['radius'] = 10.
            #msg['text'] = 'hello'
            self.ivy_interface.send(msg)
        except AttributeError:
            print('no nav ref')
        
            
        #p0 = np.array([12721170, 434633172])
        #p3 = np.array([12722170, 434633172])
        #p1 = p0 + np.array([1, 1]) * 1e3
        #p2 = p1.astype(int)

        # if 0:
        #     foo = p0*np.ones((10,2))
        #     foo[:,0] +=  np.arange(0, 1., 0.1)*1e5
        #     lon_lat_foo =  foo.astype(int)
        # else:
        #     ps *= 1e3
        #     ps += p0
        #     lon_lat_foo = ps.astype(int)
        # #print(ts, ps, lon_lat_foo)
        # msg = PprzMessage("ground", "SHAPE")
        # msg['linecolor'] = 'red'
        # msg['fillcolor'] = 'yellow'
        # msg['opacity'] = 3
        # _Circle, _Polygon, _Line = range(3)
        # msg['shape'] = _Line
        # #msg['shape'] = 2
        # msg['status'] = 0
        # #breakpoint()
        # #print(lon_lat_foo[:,1])
        # #print(lon_lat_foo[:,0])
        # msg['latarr'] = lon_lat_foo[:,1] #[434633172][p2[1]]#
        # msg['lonarr'] = lon_lat_foo[:,0] #[ 12721170][p2[0]]#
        # msg['radius'] = 40.
        # #msg['text'] = 'hello'
        # self.ivy_interface.send(msg)


    def send_command(self, ac_id, phic, vac):
        try:
            #print('send command')
            self.mngr[ac_id]["fp_roll"] = phic
            #self.mngr[ac_id]["fp_vel"] = vac
        except KeyError:
            print(f'send_command: unknown aircrafts {ac_id}')

    def shutdown(self):
        self.connect.shutdown()


    def jump_to_block(self, ac_id, block_id):
        msg = PprzMessage("ground", "JUMP_TO_BLOCK")
        msg['ac_id'] = ac_id
        msg['block_id'] = block_id
        self.ivy_interface.send(msg)
