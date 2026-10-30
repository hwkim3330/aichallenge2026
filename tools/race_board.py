#!/usr/bin/env python3
"""Scoreboard of a human vs AI race in AWSIM: http://127.0.0.1:8097/

One rclpy context per car's ROS domain (AWSIM puts vehicle N on domain N): /awsim/status (data[1] the lap being
driven, data[2] its running time -- multi_purpose_mpc_ros reads it the same way) and the speed. A lap is done when
data[1] goes up; its time is the last running time seen before. 새 레이스 asks tools/race.sh (a file in /tools)
to restart the AI and then reset AWSIM (/admin/awsim/reset on domain 0, POST /reset?now): every car back on the grid.
    python3 race_board.py [--cars 1:Metamong AI,2:사람]"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from autoware_auto_vehicle_msgs.msg import VelocityReport
from std_msgs.msg import Bool, Empty, Float32MultiArray

ap = argparse.ArgumentParser()
ap.add_argument('--cars', default='1:Metamong AI,2:사람')
ap.add_argument('--port', type=int, default=8097)
a = ap.parse_args()
CARS = [(int(d), n) for d, n in (c.split(':', 1) for c in a.cars.split(','))]
lock = threading.Lock()
board = {}


def fresh():
    return {d: {'name': n, 'lap': 0, 'cur': 0.0, 'laps': [], 'kmh': 0.0, 'seen': 0.0} for d, n in CARS}


board = fresh()

# the race's settings: the page's 설정 panel, applied by tools/race.sh at the next 새 레이스
CONFIG = '/tools/race_config.json'
DEFAULTS = {'ai_style': 'safe', 'ai_ctrl': 'mpc', 'ai_speed': '85', 'collisions': 'on', 'boosts': '2', 'laps': 'unlimited',
            'view': 'p2', 'lock': '150'}


def load_config():
    try:
        return {**DEFAULTS, **json.load(open(CONFIG))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def car_node(domain):
    ctx = Context()
    rclpy.init(context=ctx, domain_id=domain)
    node = Node(f'race_board_{domain}', context=ctx)

    def status(m):
        lap, cur = int(m.data[1]), float(m.data[2])
        with lock:
            c = board[domain]
            if c['lap'] and lap > c['lap'] and c['cur'] > 1.0:
                c['laps'].append(round(c['cur'], 3))
            c['lap'], c['cur'], c['seen'] = lap, cur, time.time()

    def vel(m):
        with lock:
            board[domain]['kmh'] = round(abs(m.longitudinal_velocity) * 3.6, 1)
    node.create_subscription(Float32MultiArray, '/awsim/status', status, 10)
    node.create_subscription(VelocityReport, '/vehicle/status/velocity_status', vel, 1)
    ex = SingleThreadedExecutor(context=ctx)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()


for d, _ in CARS:
    car_node(d)
ctx0 = Context()
rclpy.init(context=ctx0, domain_id=0)
admin = Node('race_board_admin', context=ctx0)
p_reset = admin.create_publisher(Empty, '/admin/awsim/reset', 1)
p_start = admin.create_publisher(Bool, '/admin/awsim/start', 1)
go = {'at': 0.0}      # the start's wall time (the page counts down to it); AWSIM (race2: --start-mode sync) holds the
                      # karts on the grid until /admin/awsim/start


def starter():
    while True:
        time.sleep(0.05)
        if go['at'] and time.time() >= go['at']:
            for _ in range(3):
                p_start.publish(Bool(data=True))
                time.sleep(0.1)
            go['at'] = 0.0


threading.Thread(target=starter, daemon=True).start()

PAGE = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>사람 vs AI</title>
<style>body{margin:0;background:#0d1117;color:#e6edf3;font:18px/1.4 'Noto Sans KR',sans-serif}
h1{margin:16px 20px;font-size:24px}.row{display:flex;flex-wrap:wrap;gap:14px;padding:0 20px}
.car{flex:1 1 240px;background:#161b22;border-radius:14px;padding:18px 22px}.car h2{margin:0 0 6px;font-size:24px}
.big{font-size:54px;font-weight:700;font-variant-numeric:tabular-nums}.k{color:#8b949e;font-size:15px}
.best{color:#3fb950}.lead{outline:3px solid #f0b72f}button{margin:16px 24px;font-size:18px;padding:10px 18px;border-radius:10px;border:0;background:#f0b72f;color:#111;cursor:pointer}
table{width:100%;margin-top:8px;border-collapse:collapse;font-variant-numeric:tabular-nums}td{padding:2px 0}
.cfg{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px 14px;margin:16px 20px 0}
.cfg label{display:flex;flex-direction:column;font-size:13px;color:#8b949e;gap:3px}
.cfg select{font-size:15px;padding:5px 8px;border-radius:8px;background:#21262d;color:#e6edf3;border:1px solid #30363d}</style>
<h1>사람 vs Metamong AI · AWSIM citycircuit <span id="go" style="color:#f0b72f;font-size:44px;margin-left:24px"></span></h1><div class="row" id="cars"></div>
<div class="cfg" id="cfg"></div>
<button onclick="apply()">설정 적용 · 새 레이스</button>
<p class="k" style="margin:0 24px">휠: R2 가속 · L2 브레이크 · ✕ 부스트 · 패들 R 후진 / L 전진</p>
<script>const f=t=>t?t.toFixed(3)+' s':'-';
const OPTS={ai_style:['AI 성격',[['safe','안전 (라이다로 피함)'],['race','레이싱 (안 피함)']]],
 ai_ctrl:['AI 두뇌',[['mpc','MPC (최적 경로)'],['tiny_lidar_net','TinyLidarNet (신경망)']]],
 ai_speed:['AI 속도',[['55','55 %'],['70','70 %'],['85','85 %'],['100','100 % (대회 그대로)']]],
 collisions:['충돌',[['on','켬'],['off','끔 (통과)']]], boosts:['부스트',[['0','0'],['2','2'],['5','5']]],
 laps:['랩 수',[['unlimited','무제한'],['1','1'],['6','6'],['10','10']]],
 view:['내 화면',[['p2','사람 전체'],['split','반반'],['p1','AI 전체']]],
 lock:['핸들 감도',[['90','예민 (90°)'],['150','보통 (150°)'],['240','둔함 (240°)']]]};
async function cfg(){const c=await (await fetch('/config')).json();
document.getElementById('cfg').innerHTML=Object.entries(OPTS).map(([k,[n,o]])=>`<label>${n}<select id="c_${k}">${o.map(([v,t])=>`<option value="${v}" ${c[k]==v?'selected':''}>${t}</option>`).join('')}</select></label>`).join('')}
async function apply(){const c={};for(const k in OPTS)c[k]=document.getElementById('c_'+k).value;
await fetch('/config',{method:'POST',body:JSON.stringify(c)});await fetch('/reset',{method:'POST'})}
cfg();
async function tick(){const S=await (await fetch('/state')).json();const s=S.cars;const cs=Object.values(s);
document.getElementById('go').textContent=S.go_in!=null?(S.go_in>0?Math.ceil(S.go_in):'출발!'):'';
const key=c=>[c.lap,c.cur];const lead=cs.slice().sort((a,b)=>b.lap-a.lap||a.cur-b.cur)[0];
document.getElementById('cars').innerHTML=cs.map(c=>{const best=c.laps.length?Math.min(...c.laps):0;
return `<div class="car ${c===lead&&c.lap?'lead':''}"><h2>${c.name}</h2><div class="k">랩 ${c.lap||'-'} · ${c.kmh} km/h</div>
<div class="big">${c.cur.toFixed(1)}</div><div class="k">이번 랩</div><div class="k best">최고 ${f(best)}</div>
<table>${c.laps.map((t,i)=>`<tr><td>${i+1}랩</td><td style="text-align:right" class="${t===best?'best':''}">${f(t)}</td></tr>`).join('')}</table></div>`}).join('')}
setInterval(tick,200);tick()</script></html>'''


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        b = body.encode()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith('/config'):
            self._send(json.dumps(load_config()), 'application/json')
            return
        if self.path.startswith('/state'):
            with lock:
                self._send(json.dumps({'cars': board, 'go_in': round(go['at'] - time.time(), 1) if go['at'] else None}, ensure_ascii=False), 'application/json')
        else:
            self._send(PAGE, 'text/html; charset=utf-8')

    def do_POST(self):
        global board
        if self.path.startswith('/config'):
            n = int(self.headers.get('Content-Length', 0))
            try:
                c = {k: str(v) for k, v in json.loads(self.rfile.read(n)).items() if k in DEFAULTS}
                json.dump({**load_config(), **c}, open(CONFIG, 'w'))
            except ValueError:
                pass
        elif self.path.startswith('/reset'):
            # tools/race.sh watch restarts the AI's Autoware (its start sequence runs once), then sends the reset
            # here (/reset?now): every car back on the grid together, the AI ready for it
            if 'now' in self.path:
                p_reset.publish(Empty())
            else:
                open('/tools/.new_race', 'w').close()
            with lock:
                board = fresh()
        elif self.path.startswith('/clear'):            # race.sh: a fresh AWSIM has started, the board starts over
            with lock:
                board = fresh()
        elif self.path.startswith('/start'):
            go['at'] = time.time() + (3.0 if 'in=3' in self.path else 0.0)
        self._send('{}', 'application/json')


print(f'race board on http://127.0.0.1:{a.port}/ cars {CARS}', flush=True)
ThreadingHTTPServer(('0.0.0.0', a.port), H).serve_forever()
