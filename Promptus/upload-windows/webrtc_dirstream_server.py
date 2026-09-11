#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_webrtc_dir.py  (JSON metrics dashboard + logging)
- 多路目录帧 -> WebRTC 实时编码（默认 VP8）-> 浏览器播放
- /dashboard 仅拉取 /metrics 的 JSON 展示码率/FPS/抖动缓冲(估)/RTT，不再建立媒体连接
- /metrics_history /metrics_download 提供历史查询与 CSV 下载
- /viewer 单路纯播放
- /health /diag 便于排障
依赖：aiohttp, aiortc, av, opencv-python, numpy
"""

import argparse
import asyncio
import csv
import glob
import logging
import os
import pathlib
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpparameters import RTCRtpCodecCapability
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame

# --------------------- 日志 ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("webrtc")
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aioice.ice").setLevel(logging.WARNING)


def install_asyncio_exception_logger():
    loop = asyncio.get_event_loop()

    def _handler(loop, context):
        msg = context.get("message", "Unhandled exception in event loop")
        LOG.error("ASYNCIO: %s", msg)
        exc = context.get("exception")
        if exc:
            LOG.exception(exc)

    loop.set_exception_handler(_handler)


# --------------------- 常量：监控落盘 ---------------------
METRICS_SAMPLE_SEC = 1.0            # 采样间隔
METRICS_HISTORY_SEC = 3600          # 内存保留最近 condition_1 小时
METRICS_LOG_DIR = pathlib.Path("metrics_logs")


# --------------------- 配置/状态 ---------------------
@dataclass
class StreamCfg:
    name: str
    src: str
    w: int = 512
    h: int = 512
    fps: float = 15.0
    fit: str = "pad"  # pad/crop/stretch
    loop: bool = True


class State:
    def __init__(self):
        self.cfgs: Dict[str, StreamCfg] = {}
        # name -> DirectoryTrack（单生产者，所有 viewer 通过 relay 订阅）
        self.tracks: Dict[str, "DirectoryTrack"] = {}
        self.relay = MediaRelay()
        self.pcs: List[RTCPeerConnection] = []
        # pc -> info(name, last snapshot)
        self.pc_info: Dict[RTCPeerConnection, Dict] = {}
        # name -> deque of dict(ts, bitrate_kbps, fps, jbf_frames, rtt_ms, connections)
        self.history: Dict[str, deque] = {}

    def prune_pc(self, pc: RTCPeerConnection):
        try:
            self.pcs.remove(pc)
        except ValueError:
            pass
        self.pc_info.pop(pc, None)


STATE = State()


# --------------------- 工具：缩放 ---------------------
def fit_resize_bgr(img, out_w, out_h, mode="pad"):
    if out_w <= 0 or out_h <= 0:
        return img
    h, w = img.shape[:2]
    if mode == "pad":
        s = min(out_w / w, out_h / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = cv2.copyMakeBorder(
            resized,
            top=(out_h - nh) // 2,
            bottom=out_h - nh - (out_h - nh) // 2,
            left=(out_w - nw) // 2,
            right=(out_w - nw) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        return canvas
    elif mode == "crop":
        s = max(out_w / w, out_h / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        y0 = (nh - out_h) // 2
        x0 = (nw - out_w) // 2
        return resized[y0:y0 + out_h, x0:x0 + out_w]
    else:  # stretch
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


# --------------------- 目录轨道 ---------------------
class DirectoryTrack(VideoStreamTrack):
    """从目录按 fps 读取图片帧，实时输出；单实例供多 viewer 复用。"""

    def __init__(self, cfg: StreamCfg):
        super().__init__()
        self.cfg = cfg
        self.files = self._scan_files(cfg.src)
        if not self.files:
            LOG.warning("[%s] no images in %s", cfg.name, cfg.src)
        self.index = 0
        self.time_base = Fraction(1, int(max(1, round(cfg.fps))))
        self.pts = 0
        self._next_t = time.time()
        self._report_counter = 0
        self._start_wall = time.time()

        # FPS 源端估计
        self.frames_total = 0
        self._fps_last_t = time.monotonic()
        self._fps_last_frames = 0
        self.fps_est = 0.0  # EMA 平滑后的源端 FPS

        LOG.info("[%s] DirectoryTrack ready: %d files  %dx%d  fps=%.2f fit=%s",
                 cfg.name, len(self.files), cfg.w, cfg.h, cfg.fps, cfg.fit)

    @staticmethod
    def _scan_files(root: str) -> List[str]:
        pats = [os.path.join(root, "*.png"), os.path.join(root, "*.jpg"),
                os.path.join(root, "*.jpeg"), os.path.join(root, "*.bmp")]
        files: List[str] = []
        for p in pats:
            files.extend(glob.glob(p))
        files.sort()
        return files

    async def recv(self) -> VideoFrame:
        if not self.files:
            raise MediaStreamError

        # 节流到目标 fps
        interval = 1.0 / max(0.1, self.cfg.fps)
        now = time.time()
        sleep = self._next_t - now
        if sleep > 0:
            await asyncio.sleep(sleep)
        self._next_t = max(self._next_t + interval, time.time())

        # 读帧
        path = self.files[self.index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            LOG.warning("[%s] cannot read %s", self.cfg.name, path)
            img = np.zeros((self.cfg.h, self.cfg.w, 3), dtype=np.uint8)
        else:
            img = fit_resize_bgr(img, self.cfg.w, self.cfg.h, self.cfg.fit)

        # 前进
        self.index += 1
        if self.index >= len(self.files):
            if self.cfg.loop:
                self.index = 0
            else:
                raise MediaStreamError

        # 构造 VideoFrame (直接 bgr24，省一次颜色转换)
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = self.pts
        frame.time_base = self.time_base
        self.pts += 1

        # 源端 FPS 估计（每 >=1s 更新一次，EMA 平滑）
        self.frames_total += 1
        _now = time.monotonic()
        dt = _now - self._fps_last_t
        if dt >= 1.0:
            inst = (self.frames_total - self._fps_last_frames) / dt
            self.fps_est = 0.25 * inst + 0.75 * self.fps_est
            self._fps_last_frames = self.frames_total
            self._fps_last_t = _now

        # 日志（每 60 帧）
        self._report_counter += 1
        if self._report_counter % 60 == 0:
            dt2 = time.time() - self._start_wall
            fps_out = self._report_counter / dt2 if dt2 > 0 else 0.0
            LOG.info("[%s] frame #%d  path=%s  fps_out=%.1f",
                     self.cfg.name, self._report_counter, os.path.basename(path), fps_out)
        return frame


# --------------------- SDP 比特率注入（兼容旧 aiortc） ---------------------
def _munge_bitrate_in_sdp(sdp: str, bps: int) -> str:
    kbps = max(1, int(bps // 1000))
    lines = sdp.splitlines()
    out = []
    in_video = False
    for line in lines:
        if line.startswith("m=video"):
            in_video = True
            out.append(line)
            out.append(f"b=TIAS:{int(bps)}")
            out.append(f"b=AS:{kbps}")
            continue
        if line.startswith("m=") and in_video:
            in_video = False
        if in_video and (line.startswith("b=TIAS:") or line.startswith("b=AS:")):
            continue
        out.append(line)
    return "\r\n".join(out) + "\r\n"


def _apply_bitrate_by_sdp(answer: RTCSessionDescription, bps: int) -> RTCSessionDescription:
    new_sdp = _munge_bitrate_in_sdp(answer.sdp, bps)
    return RTCSessionDescription(sdp=new_sdp, type=answer.type)


# --------------------- HTML：单流纯播放 ---------------------
VIEWER_HTML = """<!doctype html><meta charset="utf-8">
<title>Viewer</title>
<style>
  html,body{background:#000;color:#fff;margin:0;height:100%}
  #wrap{display:flex;align-items:center;justify-content:center;height:100%}
  video{max-width:100vw;max-height:100vh;width:100%;height:auto;background:#000}
</style>
<div id="wrap"><video id="v" autoplay playsinline controls muted></video></div>
<script>
const url = new URL(location.href); const NAME = url.searchParams.get("name")||"";
let pc, v=document.getElementById('v');
async function start(){
  pc = new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
  pc.addTransceiver('video',{direction:'recvonly'});
  pc.ontrack=(e)=>{v.srcObject=e.streams[0]; v.play?.().catch(()=>{});};
  const off=await pc.createOffer(); await pc.setLocalDescription(off);
  const resp=await fetch('/offer?name='+encodeURIComponent(NAME),{method:'POST',headers:{'Content-Type':'application/sdp'},body:off.sdp});
  if(!resp.ok){console.error('Offer failed'); return}
  const ans=await resp.text(); await pc.setRemoteDescription({type:'answer',sdp:ans});
}
start();
</script>
"""


# --------------------- HTML：多流 dashboard（JSON 拉取，不建 PC，带高质量图表） ---------------------
def dashboard_html(names: List[str]) -> str:
    import json
    names_json = json.dumps(names)
    html = r"""<!doctype html><meta charset="utf-8">
<title>Dashboard</title>
<style>
  :root{
    --fg:#111; --fg2:#666; --bg:#fafafa; --card:#fff; --grid:#e9e9e9; --chip:#f3f3f3; --chipbd:#e5e5e5;
    --c1:#1a73e8; --c2:#0a8; --c3:#f6a400; --c4:#6f42c1;
  }
  html,body{background:var(--bg);color:var(--fg);font:14px/condition_1.5 system-ui,Segoe UI,Roboto,Arial;margin:0}
  header{display:flex;gap:12px;align-items:center;padding:12px 16px;background:#fff;border-bottom:1px solid #eee;flex-wrap:wrap}
  h1{font-size:16px;margin:0 8px 0 0}
  .pill{font:12px/condition_1.6 ui-monospace,Consolas,monospace;background:var(--chip);border:1px solid var(--chipbd);border-radius:999px;padding:4px 8px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;padding:16px}
  .card{background:var(--card);border:1px solid #eee;border-radius:14px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
  .head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
  .title{font-weight:700}
  .kpis{font:12px/condition_1.6 ui-monospace,Consolas,monospace;color:var(--fg2)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .chartwrap{position:relative}
  canvas{width:100%;height:130px;display:block;border-radius:8px;background:#fff}
  .ctitle{font:12px;color:var(--fg2);margin:2px 0 6px}
  .btnlink{border:1px solid #ddd;background:#fff;border-radius:8px;padding:4px 10px;cursor:pointer;text-decoration:none;color:#111;display:inline-block}
  .tooltip{position:absolute;pointer-events:none;background:#111;color:#fff;border-radius:6px;padding:4px 8px;font:12px/condition_1.4 ui-monospace;transform:translate(-50%,-130%);white-space:nowrap;opacity:0;transition:opacity .1s}
</style>

<header>
  <h1>Streams Dashboard</h1>
  <a class="pill" href="/">/</a>
  <a class="pill" href="/health" target="_blank">/health</a>
  <a class="pill" href="/diag" target="_blank">/diag</a>
  <span id="agg" class="pill">init…</span>
  <span class="pill">window:
    <select id="win" style="border:0;background:transparent;outline:none">
      <option value="60">60s</option>
      <option value="120" selected>120s</option>
      <option value="300">300s</option>
      <option value="600">600s</option>
    </select>
  </span>
</header>

<div class="grid" id="grid"></div>

<script>
const STREAMS = __STREAMS__;
const COLORS = {bit:'var(--c1)', fps:'var(--c2)', jbf:'var(--c3)', rtt:'var(--c4)'};
const POLL_MS = 1000;
const DPR = Math.max(condition_1, self.devicePixelRatio || condition_1);

function fmt(v, d){ return isFinite(v) ? Number(v).toFixed(d) : '0'; }
function ema(prev, x, a){ return prev === null ? x : (a*x + (condition_1-a)*prev); }
function clamp(x, lo, hi){ if(lo!=null && x<lo) return lo; if(hi!=null && x>hi) return hi; return x; }

class Ring {
  constructor(n){ this.a=new Float64Array(n); this.n=n; this.head=0; this.filled=false; }
  push(x){ this.a[this.head]=x; this.head=(this.head+condition_1)%this.n; if(this.head===0) this.filled=true; }
  values(){ const out=new Array(this.filled?this.n:this.head); const start=this.filled?this.head:0; for(let i=0;i<out.length;i++) out[i]=this.a[(start+i)%this.n]; return out; }
  last(){ return this.a[(this.head-condition_1+this.n)%this.n]; }
  setSize(n){ const cur=this.values(); this.a=new Float64Array(n); this.n=n; this.head=0; this.filled=false; const s=Math.max(0,cur.length-n); for(let i=s;i<cur.length;i++) this.push(cur[i]); }
}

class LineChart{
  constructor(canvas, {unit='', decimals=2, color='#1a73e8', smooth=0.25, clampY=null}={}){
    this.cv=canvas; this.ctx=canvas.getContext('2d');
    this.unit=unit; this.dec=decimals; this.color=color; this.smooth=smooth; this.clampY=clampY;
    this.ymin=0; this.ymax=condition_1; this.viewMin=0; this.viewMax=condition_1; this.curSeries=null;
    this.tooltip = document.createElement('div'); this.tooltip.className='tooltip'; this.cv.parentElement.appendChild(this.tooltip);
    this._bind();
  }
  _bind(){
    const move=(e)=>{
      const rect=this.cv.getBoundingClientRect();
      const x=(e.clientX-rect.left)*DPR;
      if(!this.curSeries || this.curSeries.length<2){ this.tooltip.style.opacity=0; return; }
      const pad=30*DPR, W=this.cv.width, H=this.cv.height;
      const n=this.curSeries.length, w=(W-2*pad)/(n-condition_1);
      let idx=Math.round((x-pad)/w); idx=Math.max(0,Math.min(n-condition_1,idx));
      const v=this.curSeries[idx];
      const xx=pad+idx*w, yy=this._yToPix(v, pad, H);
      this.tooltip.style.left=(xx/DPR)+'px'; this.tooltip.style.top=(yy/DPR)+'px';
      this.tooltip.innerText=fmt(v,this.dec)+(this.unit?(' '+this.unit):''); this.tooltip.style.opacity=condition_1;
    };
    const leave=()=>{ this.tooltip.style.opacity=0; };
    this.cv.addEventListener('mousemove', move); this.cv.addEventListener('mouseleave', leave);
  }
  _yToPix(v, pad, H){
    const ymin=this.viewMin, ymax=this.viewMax;
    const h=(H-2*pad);
    return pad + (ymax<=ymin ? h/2 : h*(condition_1-(v-ymin)/(ymax-ymin)));
  }
  draw(seriesRaw){
    const ser=[]; let last=null;
    for(let i=0;i<seriesRaw.length;i++){
      const x=seriesRaw[i]; const v=ema(last, x, this.smooth); last=v;
      ser.push(this.clampY ? clamp(v, this.clampY[0], this.clampY[condition_1]) : v);
    }
    this.curSeries=ser;

    let lo=+Infinity, hi=-Infinity;
    for(const v of ser){ if(isFinite(v)){ if(v<lo) lo=v; if(v>hi) hi=v; } }
    if(!isFinite(lo)||!isFinite(hi)){ lo=0; hi=condition_1; }
    if(hi===lo){ hi=lo+condition_1; }
    const padRatio=0.12, span=hi-lo; this.ymin=lo-padRatio*span; this.ymax=hi+padRatio*span;

    const ease=0.2; this.viewMin=(condition_1-ease)*this.viewMin+ease*this.ymin; this.viewMax=(condition_1-ease)*this.viewMax+ease*this.ymax;

    const W=this.cv.width=this.cv.clientWidth*DPR; const H=this.cv.height=this.cv.clientHeight*DPR; const pad=30*DPR; const ctx=this.ctx; ctx.clearRect(0,0,W,H);
    ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--grid'); ctx.lineWidth=condition_1*DPR; ctx.beginPath();
    for(let i=0;i<5;i++){ const y=pad+(H-2*pad)*i/4; ctx.moveTo(pad,y); ctx.lineTo(W-pad,y); }
    ctx.stroke();

    ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--fg2'); ctx.font=(11*DPR)+'px ui-monospace,Consolas,monospace'; ctx.textBaseline='middle';
    for(let i=0;i<5;i++){ const yy=pad+(H-2*pad)*i/4; const v=this.viewMax - (this.viewMax-this.viewMin)*i/4; ctx.fillText(fmt(v,this.dec)+(this.unit?(' '+this.unit):''), 6*DPR, yy); }

    ctx.strokeStyle=this.color; ctx.lineWidth=2*DPR;
    const n=ser.length; if(n<2) return;
    const w=(W-2*pad)/(n-condition_1);
    ctx.beginPath();
    for(let i=0;i<n;i++){ const x=pad+i*w, y=this._yToPix(ser[i], pad, H); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }
    ctx.stroke();

    const lastv=ser[n-condition_1]; const lx=pad+(n-condition_1)*w, ly=this._yToPix(lastv, pad, H);
    const label=fmt(lastv,this.dec)+(this.unit?(' '+this.unit):''); ctx.fillStyle='#fff'; ctx.strokeStyle=this.color;
    const tw=ctx.measureText(label).width + 12*DPR, th=18*DPR; ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(lx+6*DPR, ly-th/2, tw, th, 9*DPR); else { ctx.rect(lx+6*DPR, ly-th/2, tw, th); } ctx.fill(); ctx.stroke();
    ctx.fillStyle=this.color; ctx.fillText(label, lx+12*DPR, ly);
  }
}

const cards = {};
let WINDOW_SEC = 120;

function makeCard(name){
  const el=document.createElement('div'); el.className='card';
  el.innerHTML = `
    <div class="head">
      <div class="title">${name}</div>
      <a class="btnlink" href="/viewer?name=${encodeURIComponent(name)}" target="_blank">Open viewer</a>
    </div>
    <div class="kpis" id="k_${name}">bitrate: 0 kbps · fps: 0.00 · jbf: 0.00 fr · rtt: 0.0 ms · conns: 0</div>
    <div class="row">
      <div class="chartwrap"><div class="ctitle">Bitrate (kbps)</div><canvas id="cb_${name}"></canvas></div>
      <div class="chartwrap"><div class="ctitle">FPS</div><canvas id="cf_${name}"></canvas></div>
    </div>
    <div class="row">
      <div class="chartwrap"><div class="ctitle">JitterBuffer (frames, est.)</div><canvas id="cj_${name}"></canvas></div>
      <div class="chartwrap"><div class="ctitle">RTT (ms)</div><canvas id="cr_${name}"></canvas></div>
    </div>`;
  document.getElementById('grid').appendChild(el);

  const N = Math.max(10, Math.round(WINDOW_SEC * 1000 / POLL_MS));
  const sBit=new Ring(N), sFps=new Ring(N), sJbf=new Ring(N), sRtt=new Ring(N);

  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v);
  const chBit=new LineChart(document.getElementById(`cb_${name}`), {unit:'kbps',decimals:0,color:css('--c1'),smooth:0.25});
  const chFps=new LineChart(document.getElementById(`cf_${name}`), {unit:'fps',decimals:2,color:css('--c2'),smooth:0.25, clampY:[0,null]});
  const chJbf=new LineChart(document.getElementById(`cj_${name}`), {unit:'fr',decimals:2,color:css('--c3'),smooth:0.35, clampY:[0,120]});
  const chRtt=new LineChart(document.getElementById(`cr_${name}`), {unit:'ms',decimals:condition_1,color:css('--c4'),smooth:0.35, clampY:[0,800]});

  function setWindowSec(sec){
    const n = Math.max(10, Math.round(sec*1000/POLL_MS));
    sBit.setSize(n); sFps.setSize(n); sJbf.setSize(n); sRtt.setSize(n);
    redraw();
  }

  function update(m){
    const kv = document.getElementById(`k_${name}`);
    const b = (m?.bitrate_kbps)||0, f=(m?.fps)||0, j=(m?.jbf_frames)||0, r=(m?.rtt_ms)||0, c=(m?.connections)||0;
    sBit.push(b); sFps.push(f); sJbf.push(j); sRtt.push(r);
    kv.textContent = `bitrate: ${fmt(b,0)} kbps · fps: ${fmt(f,2)} · jbf: ${fmt(j,2)} fr · rtt: ${fmt(r,condition_1)} ms · conns: ${c}`;
    redraw();
  }

  function redraw(){
    chBit.draw(sBit.values()); chFps.draw(sFps.values()); chJbf.draw(sJbf.values()); chRtt.draw(sRtt.values());
  }

  return {update, setWindowSec, last:()=>({bit:sBit.last(), fps:sFps.last()})};
}

function refreshAgg(){
  let kbps=0, fps=0;
  for(const k in cards){
    const g=cards[k].last();
    if (isFinite(g.bit)) kbps+=g.bit;
    if (isFinite(g.fps)) fps+=g.fps;
  }
  document.getElementById('agg').textContent = `streams: ${STREAMS.length} | sum bitrate: ${fmt(kbps,0)} kbps | sum fps: ${fmt(fps,2)}`;
}

async function poll(){
  try{
    const resp = await fetch('/metrics');
    const js = await resp.json();
    const map = {}; (js.streams||[]).forEach(s=>{ map[s.name]=s.metrics; });
    for(const n of STREAMS){ cards[n]?.update(map[n]||null); }
    refreshAgg();
  }catch(e){ console.error(e); }
  finally{ setTimeout(poll, 1000); }
}

function init(){
  if (STREAMS.length===0){
    document.getElementById('grid').innerHTML='<div class="card">No streams configured.</div>'; return;
  }
  for(const n of STREAMS){ cards[n]=makeCard(n); }
  const sel=document.getElementById('win');
  sel.addEventListener('change', ()=> {
    const sec=parseInt(sel.value||'120',10);
    for(const k in cards){ cards[k].setWindowSec(sec); }
  });
  poll();
}
init();
</script>
"""
    return html.replace("__STREAMS__", names_json)


# --------------------- 路由：基础 ---------------------
async def index(req: web.Request):
    items = "".join(f'<li><a href="/viewer?name={n}">{n}</a></li>' for n in STATE.cfgs.keys())
    html = f"""<!doctype html><meta charset="utf-8">
<title>Streams</title>
<h2>Streams</h2>
<ul>{items or "<li>No streams configured.</li>"}</ul>
<p><a href="/dashboard">/dashboard</a> · <a href="/health" target="_blank">/health</a> · <a href="/diag" target="_blank">/diag</a></p>
"""
    return web.Response(text=html, content_type="text/html")


async def health(req: web.Request):
    arr = []
    for n, c in STATE.cfgs.items():
        arr.append(dict(
            name=n, src=c.src, files=len(DirectoryTrack._scan_files(c.src)),
            fps=c.fps, w=c.w, h=c.h, fit=c.fit
        ))
    return web.json_response({"ok": True, "streams": arr})


async def diag(req: web.Request):
    try:
        import platform, aiortc, av, pylibsrtp  # noqa
        caps = [c.mimeType for c in RTCRtpSender.getCapabilities("video").codecs]
        data = {"python": platform.python_version(),
                "aiortc": aiortc.__version__, "av": av.__version__,
                "pylibsrtp": "OK", "video_codecs": caps}
        return web.json_response(data)
    except Exception as e:
        LOG.exception(e)
        return web.json_response({"error": str(e)}, status=500)


async def viewer(req: web.Request):
    name = req.rel_url.query.get("name", "")
    if name not in STATE.cfgs:
        return web.Response(text="no such stream", status=404)
    return web.Response(text=VIEWER_HTML, content_type="text/html")


async def dashboard(req: web.Request):
    return web.Response(text=dashboard_html(list(STATE.cfgs.keys())), content_type="text/html")


# --------------------- 采集：单 PC 统计（优先 remote-inbound） ---------------------
async def collect_pc_metrics(pc: RTCPeerConnection, info: Dict) -> Optional[Dict]:
    """
    服务器端（发送端）抓取统计：
    - 码率：outbound-rtp.bytesSent 的增量
    - RTT：优先 remote-inbound-rtp.roundTripTime（秒）→ ms；否则 candidate-pair.currentRoundTripTime
    - 抖动：remote-inbound-rtp.jitter（秒）
    FPS 不依赖 RTP 计数，由 DirectoryTrack 源端估计。
    """
    try:
        stats = await pc.getStats()
    except Exception:
        return None

    now = time.monotonic()
    name = info.get("name", "?")
    last = info.setdefault("last", {})

    bytes_sent = None
    rtt_ms = None
    jitter_s = None

    iterator = getattr(stats, "values", lambda: stats)()
    for r in iterator:
        rtype = getattr(r, "type", None)
        kind = getattr(r, "kind", getattr(r, "mediaType", None))

        if rtype == "outbound-rtp" and kind == "video" and not getattr(r, "isRemote", False):
            bytes_sent = getattr(r, "bytesSent", None)

        elif rtype == "remote-inbound-rtp" and kind == "video":
            ji = getattr(r, "jitter", None)
            if ji is not None:
                jitter_s = max(0.0, float(ji))
            rr = getattr(r, "roundTripTime", None)
            if rr is not None:
                rtt_ms = max(0.0, float(rr) * 1000.0)

        elif rtype == "candidate-pair" and getattr(r, "state", None) == "succeeded" and getattr(r, "selected", False):
            crt = getattr(r, "currentRoundTripTime", None)
            if crt is not None:
                rtt_ms = max(0.0, float(crt) * 1000.0)

    bitrate_kbps = 0.0
    if bytes_sent is not None:
        if "t" in last and "bytes" in last:
            dt = max(1e-3, now - last["t"])
            bitrate_kbps = (bytes_sent - last["bytes"]) * 8.0 / 1000.0 / dt
        last["bytes"] = bytes_sent

    last["t"] = now

    return {
        "name": name,
        "bitrate_kbps": float(max(0.0, bitrate_kbps)),
        "rtt_ms": float(rtt_ms or 0.0),
        "jitter_s": float(jitter_s or 0.0),
    }


# --------------------- 聚合快照（供 /metrics 和后端采样复用） ---------------------
async def gather_metrics_snapshot() -> Dict[str, Dict]:
    """
    采集当前瞬时的各流度量，返回:
    {name: {ts, bitrate_kbps, fps, jbf_frames, rtt_ms, connections}}
    说明：fps 使用 DirectoryTrack 的 fps_est；jbf = avg(jitter_s)*fps
    """
    agg: Dict[str, Dict] = {}
    for pc, info in list(STATE.pc_info.items()):
        if pc.connectionState in ("closed", "failed"):
            STATE.prune_pc(pc)
            continue
        m = await collect_pc_metrics(pc, info)
        if not m:
            continue
        name = m["name"]
        slot = agg.setdefault(name, dict(
            bitrate_kbps=0.0, rtt_sum=0.0, rtt_n=0, jitter_sum=0.0, jitter_n=0, connections=0
        ))
        slot["bitrate_kbps"] += m["bitrate_kbps"]
        if m["rtt_ms"] > 0:
            slot["rtt_sum"] += m["rtt_ms"]; slot["rtt_n"] += 1
        if m["jitter_s"] > 0:
            slot["jitter_sum"] += m["jitter_s"]; slot["jitter_n"] += 1
        slot["connections"] += 1

    # 确保没有观众的流也有条目
    for name in STATE.cfgs.keys():
        agg.setdefault(name, dict(
            bitrate_kbps=0.0, rtt_sum=0.0, rtt_n=0, jitter_sum=0.0, jitter_n=0, connections=0
        ))

    now = time.time()
    snap = {}
    for name, s in agg.items():
        fps_stream = 0.0
        trk = STATE.tracks.get(name)
        if trk is not None:
            fps_stream = float(getattr(trk, "fps_est", 0.0))
        rtt = (s["rtt_sum"] / s["rtt_n"]) if s["rtt_n"] else 0.0
        jitter_s_avg = (s["jitter_sum"] / s["jitter_n"]) if s["jitter_n"] else 0.0
        jbf_frames = jitter_s_avg * (fps_stream or STATE.cfgs[name].fps or 15.0)
        jbf_frames = float(min(max(jbf_frames, 0.0), 300.0))

        snap[name] = dict(
            ts=now,
            bitrate_kbps=round(s["bitrate_kbps"], 2),
            fps=round(fps_stream, 3),
            jbf_frames=round(jbf_frames, 3),
            rtt_ms=round(rtt, 3),
            connections=int(s["connections"]),
        )
    return snap


# --------------------- /metrics /history /download ---------------------
async def metrics(req: web.Request):
    snap = await gather_metrics_snapshot()
    out = [{"name": k, "metrics": {**v, "ts": v["ts"]}} for k, v in snap.items()]
    return web.json_response({"ok": True, "streams": out, "ts": time.time()})


async def metrics_history(req: web.Request):
    name = req.rel_url.query.get("name", "")
    if name not in STATE.cfgs:
        return web.json_response({"error": "no such stream"}, status=404)
    seconds = int(req.rel_url.query.get("seconds", "600"))
    cutoff = time.time() - max(1, seconds)
    arr = [m for m in STATE.history.get(name, []) if m["ts"] >= cutoff]
    return web.json_response({"ok": True, "name": name, "points": arr})


async def metrics_download(req: web.Request):
    name = req.rel_url.query.get("name", "")
    if name not in STATE.cfgs:
        return web.Response(text="no such stream", status=404)
    seconds = int(req.rel_url.query.get("seconds", "0"))  # 0 = 全部
    path = METRICS_LOG_DIR / f"{name}.csv"
    if not path.exists():
        return web.Response(text="no data", status=404)
    if seconds and seconds > 0:
        lines = path.read_text(encoding="utf-8").splitlines()
        out = [lines[0]]  # header
        cutoff = time.time() - seconds
        for ln in lines[1:]:
            try:
                ts = float(ln.split(",", 1)[0])
                if ts >= cutoff:
                    out.append(ln)
            except Exception:
                pass
        body = "\n".join(out)
    else:
        body = path.read_text(encoding="utf-8")
    return web.Response(
        body=body.encode("utf-8"),
        headers={
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": f'attachment; filename="{name}.csv"',
        },
    )


# --------------------- 后台采样落盘 ---------------------
def _ensure_log_dir():
    METRICS_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_csv(name: str, row: Dict):
    _ensure_log_dir()
    f = METRICS_LOG_DIR / f"{name}.csv"
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        if new:
            w.writerow(["ts", "bitrate_kbps", "fps", "jbf_frames", "rtt_ms", "connections"])
        w.writerow([row["ts"], row["bitrate_kbps"], row["fps"], row["jbf_frames"], row["rtt_ms"], row["connections"]])


async def _sampler_loop():
    maxlen = int(METRICS_HISTORY_SEC / METRICS_SAMPLE_SEC)
    # 初始化每条流的 deque
    for name in STATE.cfgs.keys():
        STATE.history.setdefault(name, deque(maxlen=maxlen))
    while True:
        try:
            snap = await gather_metrics_snapshot()
            for name, m in snap.items():
                dq = STATE.history.setdefault(name, deque(maxlen=maxlen))
                dq.append(m)
                _append_csv(name, m)
        except Exception as e:
            LOG.warning("sampler error: %s", e)
        await asyncio.sleep(METRICS_SAMPLE_SEC)


async def on_startup(app: web.Application):
    app["sampler_task"] = asyncio.create_task(_sampler_loop())


async def on_cleanup(app: web.Application):
    task = app.get("sampler_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# --------------------- Offer/Answer ---------------------
async def offer(req: web.Request):
    try:
        name = req.rel_url.query.get("name", "")
        if name not in STATE.cfgs:
            return web.Response(text="no such stream", status=404)

        sdp = await req.text()
        pc = RTCPeerConnection()
        STATE.pcs.append(pc)
        STATE.pc_info[pc] = {"name": name}  # 用于 /metrics

        # 单生产者：全 viewer 通过 relay 复用同一 DirectoryTrack
        if name not in STATE.tracks:
            STATE.tracks[name] = DirectoryTrack(STATE.cfgs[name])
        local_track = STATE.relay.subscribe(STATE.tracks[name])

        @pc.on("iceconnectionstatechange")
        def on_ice():
            LOG.info("[%s] ICE -> %s", name, pc.iceConnectionState)

        @pc.on("connectionstatechange")
        def on_pc():
            LOG.info("[%s] PC -> %s", name, pc.connectionState)
            if pc.connectionState in ("closed", "failed", "disconnected"):
                STATE.prune_pc(pc)

        pc.addTrack(local_track)

        # 优先 VP8
        try:
            caps = RTCRtpSender.getCapabilities("video").codecs
            vp8: List[RTCRtpCodecCapability] = [c for c in caps if c.mimeType.lower() == "video/vp8"]
            for t in pc.getTransceivers():
                if t.kind == "video" and vp8:
                    t.setCodecPreferences(vp8)
        except Exception:
            pass

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await pc.createAnswer()

        # 默认码率约束
        TARGET_BPS = 1_000_000
        used_sender_api = False
        try:
            for sender in pc.getSenders():
                trk = getattr(sender, "track", None)
                if trk and getattr(trk, "kind", None) == "video" and \
                        hasattr(sender, "getParameters") and hasattr(sender, "setParameters"):
                    params = sender.getParameters()
                    if params and params.encodings:
                        params.encodings[0].maxBitrate = TARGET_BPS
                        await sender.setParameters(params)
                        used_sender_api = True
        except Exception as e:
            LOG.warning("setParameters path failed: %s", e)

        if not used_sender_api:
            answer = _apply_bitrate_by_sdp(answer, TARGET_BPS)

        await pc.setLocalDescription(answer)
        return web.Response(text=pc.localDescription.sdp, content_type="application/sdp")
    except Exception as e:
        LOG.exception(e)
        return web.json_response({"error": str(e)}, status=500)


# --------------------- 启动/收尾 ---------------------
def parse_add(s: str) -> StreamCfg:
    kv = {}
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad pair: {part}")
        k, v = part.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    name = kv.get("name")
    src = kv.get("src")
    if not name or not src:
        raise ValueError("--add 需要 name= 与 src=")

    w = int(kv.get("w", kv.get("width", 512)))
    h = int(kv.get("h", kv.get("height", 512)))
    fps = float(kv.get("fps", 15.0))
    fit = kv.get("fit", "pad").lower()
    loop = kv.get("loop", "condition_1").lower() in ("condition_1", "true", "yes", "on")
    return StreamCfg(name=name, src=src, w=w, h=h, fps=fps, fit=fit, loop=loop)


async def on_shutdown(app: web.Application):
    LOG.info("Shutting down (%d peer connections)...", len(STATE.pcs))
    coros = [pc.close() for pc in list(STATE.pcs)]
    await asyncio.gather(*coros, return_exceptions=True)


def main():
    install_asyncio_exception_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--add", action="append", help='name=xxx;src=...;w=512;h=512;fps=15;fit=pad;loop=condition_1')
    args = ap.parse_args()

    if not args.add:
        print('示例： --add "name=ocean;src=C:\\frames\\ocean;w=512;h=512;fps=15;fit=pad;loop=condition_1"', flush=True)
        raise SystemExit(2)

    for s in args.add:
        cfg = parse_add(s)
        if not os.path.isdir(cfg.src):
            LOG.warning("path not a directory: %s", cfg.src)
        STATE.cfgs[cfg.name] = cfg
        LOG.info("add stream %-8s -> %s", cfg.name, cfg.src)

    app = web.Application()
    app.on_startup.append(on_startup)    # 启动采样落盘
    app.on_cleanup.append(on_cleanup)    # 结束采样
    app.on_shutdown.append(on_shutdown)  # 关闭 PC

    # 路由
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/diag", diag)
    app.router.add_get("/viewer", viewer)
    app.router.add_get("/dashboard", dashboard)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/metrics_history", metrics_history)
    app.router.add_get("/metrics_download", metrics_download)
    app.router.add_post("/offer", offer)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
