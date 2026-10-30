#!/usr/bin/env python3
"""A person on the G923 drives one of AWSIM's karts: the human side of a human vs AI race.

Runs in the dev image on the slot's ROS domain (tools/race.sh), with no Autoware for that slot: the wheel bridge's
WebSocket ({"type":"input", steer deg, gas, brake 0..1, codes[]} -- autoware-pc-setup g923_autoware_manual.py speaks
the same) becomes the kart's /control/command/control_cmd, as teleop_manager does for a gamepad: the commanded
acceleration, and a target speed 6 m/s beyond the current one at full pedal (AWSIM's bridge tracks .speed).

    gas    the pedal, or R2 / R3 held (the desk has no pedals)     brake  the pedal, or L2 / L3 held
    rim    +-LOCK_DEG deg = +-0.64 rad at the tyres (the kart's max_steer_angle)
    ✕      boost (AWSIM's /awsim/cmd, as teleop's boost button)    paddles  R: reverse, L: drive
The wheel gets its centring feel back from the kart's speed and tyre angle (the bridge's 'calm' feel).
    ROS_DOMAIN_ID=2 WHEEL_URL=ws://10.42.0.90:8771 python3 human_driver.py"""
import base64
import json
import math
import os
import socket
import struct
import threading
import time
from urllib.parse import urlparse

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import GearCommand, SteeringReport, VelocityReport
from std_msgs.msg import Bool, Float32MultiArray

LOCK_DEG = float(os.environ.get('LOCK_DEG', '150'))   # rim degrees for full lock: quick, as a kart is
MAX_TIRE = 0.64                                       # rad, the kart's max_steer_angle
# AWSIM flags a commanded acceleration above the kart's cap (base max acceleration, 1.37 m/s2) as OVER and holds the
# kart at 5 km/h from then on (3.5 did it on the first press; 1.5 crept past): the command stays under it, and the
# speed target does the pulling
A_GAS, A_BRAKE = 1.3, 1.3                             # |m/s2| commanded (the kart's physics sets what it really does)
VMAX = float(os.environ.get('VMAX', '30'))           # m/s target ceiling (the kart runs out of power well before)
WB = 1.087                                            # m, the kart's wheelbase (for the wheel's feel only)
G923 = {288: 'cross', 289: 'square', 290: 'circle', 291: 'triangle', 292: 'paddleR', 293: 'paddleL', 294: 'R2', 295: 'L2',
        296: 'share', 297: 'options', 298: 'R3', 299: 'L3', 707: 'plus', 708: 'minus', 709: 'dialCW', 710: 'dialCCW', 711: 'enter',
        712: 'ps'}


class WheelClient(threading.Thread):
    """the smallest WebSocket client that reads the bridge's text frames (stdlib only)"""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.state, self.t = urlparse(url), None, 0.0
        self.sock, self.lock = None, threading.Lock()

    def send(self, text):
        if self.sock is not None:
            self._frame(self.sock, 0x1, text.encode())

    def _frame(self, s, op, data):
        mask = os.urandom(4)
        n = len(data)
        head = bytes([0x80 | op, 0x80 | n]) if n < 126 else bytes([0x80 | op, 0xFE]) + struct.pack('>H', n)
        try:
            with self.lock:
                s.sendall(head + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(data)))
        except OSError:
            self.sock = None

    def run(self):
        while True:
            try:
                self._session()
            except OSError as e:
                print(f'wheel: {e}; retrying', flush=True)
            self.sock = None
            time.sleep(2)

    def _session(self):
        s = socket.create_connection((self.url.hostname, self.url.port or 80), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f'GET {self.url.path or "/"} HTTP/1.1\r\nHost: {self.url.hostname}\r\nUpgrade: websocket\r\n'
                   f'Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n'
                   f'User-Agent: aic-human-driver\r\n\r\n').encode())
        buf = b''
        while b'\r\n\r\n' not in buf:
            buf += s.recv(4096)
        buf = buf.split(b'\r\n\r\n', 1)[1]
        print(f'wheel: connected {self.url.geturl()}', flush=True)
        s.settimeout(None)
        self.sock = s

        def need(n):
            nonlocal buf
            while len(buf) < n:
                chunk = s.recv(65536)
                if not chunk:
                    raise OSError('closed')
                buf += chunk
            out, buf = buf[:n], buf[n:]
            return out
        while True:
            b0, b1 = need(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack('>H', need(2))[0]
            elif n == 127:
                n = struct.unpack('>Q', need(8))[0]
            mask = need(4) if b1 & 0x80 else None
            data = need(n)
            if mask:
                data = bytes(c ^ mask[i % 4] for i, c in enumerate(data))
            op = b0 & 0x0F
            if op == 1:
                try:
                    m = json.loads(data)
                except ValueError:
                    continue
                if m.get('type') == 'input':
                    self.state, self.t = m, time.time()
            elif op == 9:
                self._frame(s, 0xA, data)
            elif op == 8:
                raise OSError('closed by the bridge')


class Human(Node):
    def __init__(self):
        # stamped on the sim's clock, as the AI's commands are
        super().__init__('human_driver', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.p_cmd = self.create_publisher(AckermannControlCommand, '/control/command/control_cmd', 1)
        self.p_gear = self.create_publisher(GearCommand, '/control/command/gear_cmd', 1)
        self.p_mode = self.create_publisher(Bool, '/awsim/control_mode_request_topic', 1)
        self.p_boost = self.create_publisher(Float32MultiArray, '/awsim/cmd', 1)
        self.v, self.tire = 0.0, 0.0
        self.create_subscription(VelocityReport, '/vehicle/status/velocity_status', lambda m: setattr(self, 'v', m.longitudinal_velocity), 1)
        self.create_subscription(SteeringReport, '/vehicle/status/steering_status', lambda m: setattr(self, 'tire', m.steering_tire_angle), 1)
        self.wheel = WheelClient(os.environ.get('WHEEL_URL', 'ws://10.42.0.90:8771'))
        self.wheel.start()
        self.gear, self.prev = GearCommand.DRIVE, set()
        self.t0, self.last = time.time(), time.time()
        self.create_timer(1 / 50, self.tick)
        self.create_timer(1 / 30, self.ffb)
        self.get_logger().info(f'human driver: lock {LOCK_DEG:.0f} deg, wheel {self.wheel.url.geturl()}')

    def tick(self):
        now = time.time()
        dt, self.last = now - self.last, now
        stamp = self.get_clock().now().to_msg()
        # the control mode (autonomous control from ROS) and the gear, twice a second
        if now - getattr(self, '_req', 0.0) > 0.5:      # (AWSIM restarts for every race: asked for all along)
            self._req = now
            self.p_mode.publish(Bool(data=True))
            self.p_gear.publish(GearCommand(stamp=stamp, command=self.gear))
        w = self.wheel.state if now - self.wheel.t < 0.5 else None
        out = AckermannControlCommand()
        out.stamp = out.lateral.stamp = out.longitudinal.stamp = stamp
        if w is None:                     # no wheel: brake to a stop, wheels straight
            out.longitudinal.acceleration = -A_BRAKE
            out.longitudinal.speed = max(0.0, self.v - A_BRAKE * dt)
            self.p_cmd.publish(out)
            return
        held = {G923.get(c, str(c)) for c in w.get('codes', [])}
        gas = max(float(w.get('gas', 0.0)), 1.0 if held & {'R2', 'R3'} else 0.0)
        brake = max(float(w.get('brake', 0.0)), 1.0 if held & {'L2', 'L3'} else 0.0)
        for name, gear, label in (('paddleR', GearCommand.REVERSE, 'R'), ('paddleL', GearCommand.DRIVE, 'D')):
            if name in held and name not in self.prev and abs(self.v) < 0.5:
                self.gear = gear
                self.p_gear.publish(GearCommand(stamp=stamp, command=gear))
                self.get_logger().info(f'gear {label}')
        if 'cross' in held and 'cross' not in self.prev:
            self.p_boost.publish(Float32MultiArray(data=[1.0]))
            self.p_boost.publish(Float32MultiArray(data=[0.0]))
            self.get_logger().info('boost')
        self.prev = held
        acc = A_GAS * gas - A_BRAKE * brake
        sign = -1.0 if self.gear == GearCommand.REVERSE else 1.0
        # AWSIM's bridge chases .speed: teleop's current + acc * dt sat 0.07 m/s above the car and it crept at
        # ~0.9 m/s2; a target 6 m/s beyond (scaled by the pedal) pulls the kart's full power, and the brake likewise
        speed = abs(self.v) + 6.0 * gas - 6.0 * brake if (gas or brake) else abs(self.v)
        out.longitudinal.acceleration = acc
        out.longitudinal.speed = sign * max(0.0, min(VMAX if sign > 0 else 3.0, speed))
        out.lateral.steering_tire_angle = max(-1.0, min(1.0, float(w.get('steer', 0.0)) / LOCK_DEG)) * MAX_TIRE
        out.lateral.steering_tire_rotation_rate = 2.0
        self.p_cmd.publish(out)

    def ffb(self):
        v, d = self.v, self.tire
        ay = v * v * math.tan(d) / WB
        self.wheel.send(json.dumps({
            'kmh': round(abs(v) * 3.6, 2), 'lock': int(LOCK_DEG), 'range': int(2 * LOCK_DEG), 'feel': 'calm', 'mode': 'manual',
            'g': {'style': 'steady', 'gain': 0.6, 'align': 0.8, 'road': 0.3, 'weight': 0.8, 'base': 0.7, 'servo': 0.0, 'tf': 0.0},
            'at': [round(d - 0.02 * ay, 5), round(MAX_TIRE / LOCK_DEG, 6), 8.5, 1.35, round(min(1.0, abs(v) / 4.0), 3)],
            'al': 0.0, 'tx': 0.0, 'hz': 8.0, 'jo': [0, 0], 'ts': int(time.time() * 1000) % 100000000}))


def main():
    rclpy.init()
    rclpy.spin(Human())


if __name__ == '__main__':
    main()
