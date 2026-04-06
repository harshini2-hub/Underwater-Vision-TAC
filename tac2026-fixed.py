#!/usr/bin/env python3
"""
tac2026.py  —  TAC 2026  |  ALL-IN-ONE  |  Dual Camera  Transmitter + Receiver
================================================================================

  ┌─────────────────────────────────────────────────────────────────┐
  │  CONDITION 1 — Laptop Only (cameras plugged into laptop)        │
  │    python3 tac2026.py --laptop                                  │
  │                                                                 │
  │  CONDITION 2 — Jetson connected to Laptop via Ethernet          │
  │    On LAPTOP  : python3 tac2026.py --receive                    │
  │    On JETSON  : python3 tac2026.py --transmit --host <LAPTOP_IP>│
  └─────────────────────────────────────────────────────────────────┘

OPTIONS:
  --laptop              Laptop-only mode  (runs TX + RX together on laptop)
  --receive             Laptop receiver only  (use when Jetson transmits)
  --transmit            Jetson transmitter only  (run this ON the Jetson)
  --host   <ip>         Laptop ethernet IP  (required with --transmit)
  --dev0   <device>     Camera 0 device  (default: /dev/video0)
  --dev1   <device>     Camera 1 device  (default: /dev/video2)
  --bitrate <kbps>      H264 bitrate  (default: 5000)

LIGHTING PRESET (optional):
  python3 tac2026.py --laptop bright
  python3 tac2026.py --receive night

KEYBOARD (on camera window):
  Q  quit    1 BRIGHT   2 MIDDAY   3 EVENING   4 NIGHT   A AUTO
  T  TAC     [ EXP-     ] EXP+     - GAMMA-    = GAMMA+  R RESET
"""

import threading
import subprocess
import queue
import time
import sys
import os
import argparse
import signal

import cv2
import cv2.aruco as aruco
import numpy as np

_ROS_OK = False
try:
    import rclpy
    from rclpy.node import Node
    _ROS_OK = True
except ImportError:
    pass

from underwater_enhance import enhance_for_aruco, CAMERA_MATRICES, DIST_COEFFS


# ════════════════════════════════════════════════════════════════
#  ARG PARSING
# ════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="TAC 2026 All-in-One")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--laptop",   action="store_true", help="Laptop-only mode (TX+RX on laptop)")
    grp.add_argument("--receive",  action="store_true", help="Receive only (laptop side when Jetson transmits)")
    grp.add_argument("--transmit", action="store_true", help="Transmit only (run on Jetson)")
    p.add_argument("--host",    default=None,          help="Laptop IP (required for --transmit)")
    p.add_argument("--dev0",    default="/dev/video0",  help="Camera 0 device (default: /dev/video0)")
    p.add_argument("--dev1",    default="/dev/video2",  help="Camera 1 device (default: /dev/video2)")
    p.add_argument("--bitrate", default=5000, type=int, help="H264 bitrate kbps (default: 5000)")
    p.add_argument("lighting",  nargs="?", default=None,
                   choices=["bright","midday","evening","night","auto"])
    return p.parse_args()


# ════════════════════════════════════════════════════════════════
#  LIGHTING PROFILES
# ════════════════════════════════════════════════════════════════
LIGHTING_MODES = {
    "BRIGHT":  {"label":"BRIGHT",  "gamma":0.7,  "red_gain":1.1, "deblur_s":0.35,
                "clahe_clip":2.0,  "clahe_tile":8,  "wb_strength":0.6,
                "sharpen_thr_lo":400,"sharpen_thr_hi":600,
                "auto_lo":190,"auto_hi":255,"color_balance":(-10,0,0),
                "btn_color":(0,200,255),"btn_active":(0,230,255)},
    "MIDDAY":  {"label":"MIDDAY",  "gamma":1.0,  "red_gain":1.3, "deblur_s":0.50,
                "clahe_clip":3.0,  "clahe_tile":8,  "wb_strength":0.85,
                "sharpen_thr_lo":300,"sharpen_thr_hi":500,
                "auto_lo":120,"auto_hi":190,"color_balance":(0,0,0),
                "btn_color":(60,200,60),"btn_active":(80,240,80)},
    "EVENING": {"label":"EVENING", "gamma":1.3,  "red_gain":1.5, "deblur_s":0.60,
                "clahe_clip":3.5,  "clahe_tile":6,  "wb_strength":1.0,
                "sharpen_thr_lo":200,"sharpen_thr_hi":400,
                "auto_lo":70,"auto_hi":120,"color_balance":(5,0,-5),
                "btn_color":(40,120,255),"btn_active":(60,160,255)},
    "NIGHT":   {"label":"NIGHT",   "gamma":1.8,  "red_gain":1.7, "deblur_s":0.70,
                "clahe_clip":5.0,  "clahe_tile":4,  "wb_strength":1.0,
                "sharpen_thr_lo":80,"sharpen_thr_hi":250,
                "auto_lo":0,"auto_hi":70,"color_balance":(0,0,0),
                "btn_color":(180,80,220),"btn_active":(210,110,255)},
}
MODE_ORDER = ["BRIGHT","MIDDAY","EVENING","NIGHT"]

# ════════════════════════════════════════════════════════════════
#  COLOURS
# ════════════════════════════════════════════════════════════════
C_ACCENT=(180,160,100); C_TEXT=(200,200,200); C_DIM=(120,120,120)
C_OK=(80,180,80);       C_WARN=(60,130,200);  C_BAD=(70,70,180)
C_CROSS=(160,160,160);  C_PIPE=(200,200,160); C_PANEL_BG=(18,18,18)
C_TITLE=(160,140,90);   C_TAC=(0,200,255)

# ════════════════════════════════════════════════════════════════
#  LIGHTING STATE CLASS
# ════════════════════════════════════════════════════════════════
class LightingState:
    def __init__(self, initial="AUTO"):
        self._mode=initial.upper(); self._tac=False
        self._overrides={m:{"gamma":0.0,"exposure_shift":0} for m in MODE_ORDER}
        self._lock=threading.Lock()
    def set_mode(self,m):
        with self._lock: self._mode=m.upper()
    def get_mode(self):
        with self._lock: return self._mode
    def toggle_tac(self):
        with self._lock: self._tac=not self._tac
        return self._tac
    def tac_on(self):
        with self._lock: return self._tac
    def adjust_gamma(self,d,r):
        with self._lock: self._overrides[r]["gamma"]=round(self._overrides[r]["gamma"]+d,2)
    def adjust_exposure(self,d,r):
        with self._lock: self._overrides[r]["exposure_shift"]+=d
    def reset_mode(self,r):
        with self._lock: self._overrides[r]={"gamma":0.0,"exposure_shift":0}
    def resolve(self,brt):
        with self._lock: m=self._mode
        if m!="AUTO": return m
        for n in MODE_ORDER:
            p=LIGHTING_MODES[n]
            if p["auto_lo"]<=brt<=p["auto_hi"]: return n
        return "MIDDAY"
    def profile(self,r):
        with self._lock: ov=dict(self._overrides[r])
        base=LIGHTING_MODES[r]; p=dict(base)
        p["gamma"]=max(0.3,min(3.0,base["gamma"]+ov["gamma"]))
        p["exposure_shift"]=ov["exposure_shift"]
        return p

# ════════════════════════════════════════════════════════════════
#  CAMERA MATRICES
# ════════════════════════════════════════════════════════════════
CAM0_MATRIX=np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]],dtype=np.float32)
CAM0_DIST  =np.array([-0.30,0.10,0.,0.,0.],dtype=np.float32)
_nm0,_=cv2.getOptimalNewCameraMatrix(CAM0_MATRIX,CAM0_DIST,(640,480),0)
_M1_0,_M2_0=cv2.initUndistortRectifyMap(CAM0_MATRIX,CAM0_DIST,None,_nm0,(640,480),cv2.CV_16SC2)

CAM1_MATRIX=np.array([[554.256,0.,320.],[0.,554.256,240.],[0.,0.,1.]],dtype=np.float32)
CAM1_DIST  =np.array([-0.38,0.15,0.,0.,-0.035],dtype=np.float32)
_nm1,_=cv2.getOptimalNewCameraMatrix(CAM1_MATRIX,CAM1_DIST,(640,480),0)
_M1_1,_M2_1=cv2.initUndistortRectifyMap(CAM1_MATRIX,CAM1_DIST,None,_nm1,(640,480),cv2.CV_16SC2)

UNDISTORT_MAPS={0:(_M1_0,_M2_0), 1:(_M1_1,_M2_1)}

# ════════════════════════════════════════════════════════════════
#  ARUCO SETUP
# ════════════════════════════════════════════════════════════════
PIPELINE_TAGS={
    56:{"order":1,"name":"PIPELINE ENTRY","msg":"Pipeline detected — move to next section"},
    5: {"order":2,"name":"SECTION 2",     "msg":"Section 2 reached — proceed forward"},
    20:{"order":3,"name":"SECTION 3",     "msg":"Section 3 reached — scan surroundings"},
    32:{"order":4,"name":"PIPELINE EXIT", "msg":"Pipeline exit — inspection complete"},
}
ARUCO_DICTS        =[aruco.DICT_4X4_1000,aruco.DICT_5X5_1000,aruco.DICT_6X6_1000,aruco.DICT_7X7_1000]
ARUCO_DICT_NAMES   =["4x4","5x5","6x6","7x7"]
ARUCO_DICT_PRIORITY=[1,2,3,4]
try:    aruco_params=aruco.DetectorParameters_create()
except: aruco_params=aruco.DetectorParameters_create()
aruco_params.minMarkerPerimeterRate=0.03; aruco_params.maxMarkerPerimeterRate=10.0
aruco_params.adaptiveThreshWinSizeMin=3;  aruco_params.adaptiveThreshWinSizeMax=53
aruco_params.adaptiveThreshWinSizeStep=4; aruco_params.adaptiveThreshConstant=7
aruco_params.minCornerDistanceRate=0.05;  aruco_params.polygonalApproxAccuracyRate=0.03
aruco_params.errorCorrectionRate=0.6;     aruco_params.minMarkerDistanceRate=0.05
aruco_params.cornerRefinementMethod=aruco.CORNER_REFINE_SUBPIX
aruco_dicts=[aruco.getPredefinedDictionary(d) for d in ARUCO_DICTS]

CENTER_TOLERANCE=60; MAX_LOST=8; MIN_MARKER_AREA=500
MARKER_RESULT_TTL=0.10; SPATIAL_MERGE_DIST=80; DETECT_SCALE=0.5
PIPELINE_MSG_COOLDOWN=3.0; OVERLAY_MSG_DURATION=3.0

PIPE_HSV_RANGES=[
    (np.array([18,90,70],dtype=np.uint8),  np.array([40,255,255],dtype=np.uint8)),
    (np.array([5,100,60],dtype=np.uint8),  np.array([18,255,255],dtype=np.uint8)),
]
PIPE_MIN_AREA=800; PIPE_MIN_ASPECT=2.2; PIPE_MIN_FILL_RATIO=0.45
PIPE_MIN_SOLIDITY=0.65; PIPE_MIN_LENGTH=80

# ════════════════════════════════════════════════════════════════
#  ENHANCEMENT HELPERS
# ════════════════════════════════════════════════════════════════
_CLAHE_CACHE={}
def _get_clahe(clip,tile):
    k=(round(clip,1),tile)
    if k not in _CLAHE_CACHE: _CLAHE_CACHE[k]=cv2.createCLAHE(clipLimit=clip,tileGridSize=(tile,tile))
    return _CLAHE_CACHE[k]

def _build_gamma_lut(g):
    return np.array([(i/255.0)**(1.0/max(g,0.01))*255 for i in range(256)],dtype=np.uint8)

def _dehaze(frame,omega=0.80):
    img=frame.astype(np.float32)/255.0; patch=7
    dark=cv2.erode(np.min(img,axis=2),np.ones((patch,patch),np.uint8))
    flat=dark.flatten(); n=max(1,int(flat.size*0.001))
    idx=np.argpartition(flat,-n)[-n:]
    A=np.clip(np.max(img.reshape(-1,3)[idx],axis=0),0.5,1.0)
    nm=cv2.erode(np.min(img/A[np.newaxis,np.newaxis,:],axis=2),np.ones((patch,patch),np.uint8))
    t=np.clip(1.0-omega*nm,0.15,1.0); t3=np.stack([t]*3,axis=2)
    return (np.clip((img-A)/t3+A,0,1)*255).astype(np.uint8)

def enhance_adaptive(frame,profile,tac_mode,maps):
    p=profile; M1,M2=maps
    out=cv2.remap(frame,M1,M2,cv2.INTER_LINEAR)
    out=cv2.medianBlur(out,3)
    es=p.get("exposure_shift",0)
    if es: out=np.clip(out.astype(np.int16)+es,0,255).astype(np.uint8)
    f=out.astype(np.float32)
    bm,gm,rm=np.mean(f[:,:,0]),np.mean(f[:,:,1]),np.mean(f[:,:,2])
    gv=(bm+gm+rm)/3.0; st=p["wb_strength"]
    f[:,:,0]=np.clip(f[:,:,0]*(1+st*(gv/(bm+1e-6)-1)),0,255)
    f[:,:,1]=np.clip(f[:,:,1]*(1+st*(gv/(gm+1e-6)-1)),0,255)
    f[:,:,2]=np.clip(f[:,:,2]*(1+st*(gv/(rm+1e-6)*p["red_gain"]-1)),0,255)
    out=f.astype(np.uint8)
    cb=p.get("color_balance",(0,0,0))
    if any(c!=0 for c in cb):
        out=np.clip(out.astype(np.int16)+np.array(cb,dtype=np.int16)[np.newaxis,np.newaxis,:],0,255).astype(np.uint8)
    out=cv2.LUT(out,_build_gamma_lut(p["gamma"]))
    lab=cv2.cvtColor(out,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    out=cv2.cvtColor(cv2.merge([_get_clahe(p["clahe_clip"],p["clahe_tile"]).apply(l),a,b]),cv2.COLOR_LAB2BGR)
    if tac_mode:
        out=_dehaze(out)
        lab=cv2.cvtColor(out,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        out=cv2.cvtColor(cv2.merge([_get_clahe(5.0,4).apply(l),a,b]),cv2.COLOR_LAB2BGR)
    s=p["deblur_s"]; k=np.array([[0,-s,0],[-s,1+4*s,-s],[0,-s,0]],dtype=np.float32)
    out=np.clip(cv2.filter2D(out,-1,k),0,255).astype(np.uint8)
    gray=cv2.cvtColor(out,cv2.COLOR_BGR2GRAY); score=cv2.Laplacian(gray,cv2.CV_64F).var()
    lo,hi=p["sharpen_thr_lo"],p["sharpen_thr_hi"]
    ss=2.0 if score<lo else (1.3 if score<hi else 0.0)
    if ss>0:
        blurred=cv2.GaussianBlur(out,(3,3),0)
        out=np.clip(cv2.addWeighted(out,1+ss,blurred,-ss,0),0,255).astype(np.uint8)
    return out

# ════════════════════════════════════════════════════════════════
#  PIPELINE MASK
# ════════════════════════════════════════════════════════════════
def contour_score_for_pipe(c):
    area=cv2.contourArea(c)
    if area<PIPE_MIN_AREA: return -1,None
    rect=cv2.minAreaRect(c); rw,rh=rect[1]
    if rw<=1 or rh<=1: return -1,None
    ls,ss=max(rw,rh),min(rw,rh)
    if ss<=1 or ls/ss<PIPE_MIN_ASPECT or ls<PIPE_MIN_LENGTH: return -1,None
    if area/(rw*rh)<PIPE_MIN_FILL_RATIO: return -1,None
    ha=cv2.contourArea(cv2.convexHull(c))
    if ha<=1 or area/ha<PIPE_MIN_SOLIDITY: return -1,None
    return area*0.002+ls/ss*120+area/(rw*rh)*200+area/ha*150+ls*0.8,rect

def compute_pipeline_mask(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=np.zeros(hsv.shape[:2],dtype=np.uint8)
    for lo,hi in PIPE_HSV_RANGES: mask=cv2.bitwise_or(mask,cv2.inRange(hsv,lo,hi))
    mask=cv2.GaussianBlur(mask,(5,5),0)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(11,11)))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bc,br,bs=None,None,-1
    for c in cnts:
        score,rect=contour_score_for_pipe(c)
        if score>bs: bs,bc,br=score,c,rect
    mv=np.zeros_like(frame); pf,ang,cpx,cpy=False,0.0,0,0
    if bc is not None:
        pf=True; cv2.drawContours(mv,[bc],-1,(255,255,255),-1)
        rw,rh=br[1]; ang=br[2]+(90 if rw>rh else 0)
        M=cv2.moments(bc)
        if M["m00"]>0: cpx,cpy=int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])
    return mv,pf,ang,cpx,cpy

# ════════════════════════════════════════════════════════════════
#  ARUCO WORKERS
# ════════════════════════════════════════════════════════════════
def get_tag_role(aid):
    return ("PIPELINE",PIPELINE_TAGS[aid]) if aid in PIPELINE_TAGS else ("DOCKING",None)

def make_aruco_worker(idx, input_q, result_q):
    ad=aruco_dicts[idx]; pr=ARUCO_DICT_PRIORITY[idx]; dn=ARUCO_DICT_NAMES[idx]
    def worker():
        while True:
            try: eg,scale=input_q.get(timeout=1.0)
            except queue.Empty: continue
            corners,ids,_=aruco.detectMarkers(eg,ad,parameters=aruco_params)
            if ids is None: continue
            ts=time.monotonic()
            for i in range(len(ids)):
                c=corners[i][0]
                area=cv2.contourArea(c.astype(np.float32))
                if area<MIN_MARKER_AREA*scale**2: continue
                tx=int(c[:,0].mean()/scale); ty=int(c[:,1].mean()/scale)
                aid=int(ids[i][0]); role,_=get_tag_role(aid)
                try:
                    result_q.put_nowait(({
                        "type":"ARUCO","priority":pr,"area":area,
                        "tx":tx,"ty":ty,"label":f"ArUco {dn}:{aid}",
                        "aruco_id":aid,"role":role,
                        "corner":(corners[i]/scale).astype(np.float32),"aid":ids[i].copy()
                    },ts))
                except queue.Full: pass
    return worker

# ════════════════════════════════════════════════════════════════
#  UI HELPERS
# ════════════════════════════════════════════════════════════════
class ToastMessage:
    def __init__(self): self.text,self.sub,self.ts,self.active="","",0.0,False
    def show(self,text,sub=""):  self.text,self.sub,self.ts,self.active=text,sub,time.monotonic(),True
    def tick(self,now):
        if self.active and (now-self.ts)>OVERLAY_MSG_DURATION: self.active=False

class MissionState:
    def __init__(self):
        self.pipeline_found=False; self.pipeline_angle=0.0
        self.marker_sequence=[]; self.marker_count=0
        self.pinger_side="unknown"; self._seen=set()
    def register_tag(self,aid,role):
        if aid not in self._seen:
            self._seen.add(aid); self.marker_sequence.append(aid)
            self.marker_count=len(self.marker_sequence)
            if role=="PIPELINE": self.pipeline_found=True
    def update_pipeline_angle(self,a): self.pipeline_angle=a
    def update_pinger_side(self,s): self.pinger_side=s

def draw_panel(frame,x1,y1,x2,y2,color=C_PANEL_BG,alpha=0.65):
    ov=frame.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

def draw_row(frame,label,value,y,vc=C_TEXT):
    cv2.putText(frame,label,(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.44,C_DIM,1,cv2.LINE_AA)
    cv2.putText(frame,str(value),(160,y),cv2.FONT_HERSHEY_SIMPLEX,0.50,vc,1,cv2.LINE_AA)

def draw_crosshair(frame,cx,cy):
    cv2.circle(frame,(cx,cy),38,C_CROSS,1,cv2.LINE_AA)
    cv2.line(frame,(cx-22,cy),(cx+22,cy),C_CROSS,1)
    cv2.line(frame,(cx,cy-22),(cx,cy+22),C_CROSS,1)
    cv2.circle(frame,(cx,cy),3,C_CROSS,-1)

def draw_toast(frame,toast,now,w,h):
    if not toast.active: return
    toast.tick(now)
    if not toast.active: return
    age=now-toast.ts
    af=max(0.,min(1.,1.-max(0.,(age-(OVERLAY_MSG_DURATION-0.6))/0.6)))
    cw,ch=500,72; x1,y1=(w-cw)//2,h//2-ch//2-30; x2,y2=x1+cw,y1+ch
    ov=frame.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),(25,25,25),-1)
    cv2.addWeighted(ov,af*0.80,frame,1-af*0.80,0,frame)
    cv2.rectangle(frame,(x1,y1),(x2,y2),C_ACCENT,1,cv2.LINE_AA)
    cv2.putText(frame,toast.text,(x1+14,y1+26),cv2.FONT_HERSHEY_SIMPLEX,0.65,C_ACCENT,1,cv2.LINE_AA)
    if toast.sub:
        cv2.putText(frame,toast.sub,(x1+14,y1+54),cv2.FONT_HERSHEY_SIMPLEX,0.50,C_TEXT,1,cv2.LINE_AA)

def draw_button_bar(frame,w,h,resolved,tac,last_resolved,lighting,buttons,toast,quit_flag):
    buttons.clear()
    BAR_H=120; BTN_H=50; PAD=5; MARGIN=6
    ROW1_Y=h-BAR_H+4; ROW2_Y=ROW1_Y+BTN_H+PAD+2
    draw_panel(frame,0,h-BAR_H,w,h,(15,15,15),0.82)
    cv2.line(frame,(0,h-BAR_H),(w,h-BAR_H),(60,60,60),1)

    mode_labels=[("BRIGHT","BRIGHT"),("MIDDAY","MIDDAY"),("EVENING","EVENING"),("NIGHT","NIGHT"),("AUTO","AUTO")]
    btn_w1=(w-MARGIN*2-PAD*4)//5
    for i,(mode_key,display) in enumerate(mode_labels):
        x1=MARGIN+i*(btn_w1+PAD); y1=ROW1_Y; x2=x1+btn_w1; y2=y1+BTN_H
        is_active=(resolved==mode_key) or (mode_key=="AUTO" and lighting.get_mode()=="AUTO")
        if mode_key=="AUTO":
            bg=(80,80,80) if is_active else (35,35,35)
            tc=(220,220,220) if is_active else (140,140,140)
            bd=(160,160,160) if is_active else (60,60,60)
        else:
            mc=LIGHTING_MODES[mode_key]["btn_color"]
            bg=mc if is_active else (35,35,35)
            tc=(10,10,10) if is_active else mc
            bd=mc
        cv2.rectangle(frame,(x1,y1),(x2,y2),bg,-1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),bd,3 if is_active else 1)
        fs=0.55; tw=cv2.getTextSize(display,cv2.FONT_HERSHEY_SIMPLEX,fs,1)[0][0]
        cv2.putText(frame,display,(x1+(btn_w1-tw)//2,y1+BTN_H//2+8),cv2.FONT_HERSHEY_SIMPLEX,fs,tc,1,cv2.LINE_AA)
        if is_active: cv2.line(frame,(x1+4,y2-3),(x2-4,y2-3),bd,3)
        def _act(mk):
            def a(): lighting.set_mode(mk); toast.show(f"{mk} MODE","")
            return a
        buttons.append({"rect":(x1,y1,x2,y2),"action":_act(mode_key)})

    # Row 2 — RESET, TAC, QUIT only (EXP/GAM are now trackbars)
    row2=[
        ("RESET",  lambda: (lighting.reset_mode(last_resolved), toast.show(f"RESET {last_resolved}",""))),
        ("TAC:"+ ("ON" if tac else "OFF"), lambda: (lambda s: toast.show(f"TAC {'ON' if s else 'OFF'}",""))(lighting.toggle_tac())),
        ("QUIT",   None),
    ]
    n2=len(row2); btn_w2=(w-MARGIN*2-PAD*(n2-1))//n2
    for i,(label,action) in enumerate(row2):
        x1=MARGIN+i*(btn_w2+PAD); y1=ROW2_Y; x2=x1+btn_w2; y2=y1+BTN_H
        if label.startswith("TAC"):
            bg=C_TAC if tac else (40,40,40); tc=(10,10,10) if tac else C_TAC; bd=C_TAC
        elif label=="QUIT":
            bg=(40,30,70); tc=(180,80,220); bd=(180,80,220)
        elif label=="RESET":
            bg=(50,30,30); tc=(200,100,100); bd=(200,100,100)
        else:
            bg=(30,45,55); tc=(140,200,220); bd=(80,140,160)
        cv2.rectangle(frame,(x1,y1),(x2,y2),bg,-1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),bd,1)
        fs=0.50; tw=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,fs,1)[0][0]
        cv2.putText(frame,label,(x1+(btn_w2-tw)//2,y1+BTN_H//2+7),cv2.FONT_HERSHEY_SIMPLEX,fs,tc,1,cv2.LINE_AA)
        buttons.append({"rect":(x1,y1,x2,y2),"action":action,"is_quit":label=="QUIT"})

# ════════════════════════════════════════════════════════════════
#  RECEIVER LOOP  (one instance per camera)
# ════════════════════════════════════════════════════════════════
def run_receiver(cam_id, port, win_title, mask_title, maps, jetson_mode, initial_lighting, direct_device=None):
    lighting   = LightingState(initial_lighting or "AUTO")
    toast      = ToastMessage()
    mission    = MissionState()
    buttons    = []
    quit_flag  = [False]
    enhance_skip = 2
    pipeline_last_msg_ts = {}

    # ── DIRECT mode (laptop-only): read camera directly, no GStreamer UDP ──
    if direct_device is not None:
        print(f"[CAM{cam_id}] Direct capture from {direct_device}")
        dev_index = int(direct_device.replace("/dev/video","")) if isinstance(direct_device, str) else direct_device
        cap = cv2.VideoCapture(dev_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[CAM{cam_id}] ERROR: Cannot open camera {direct_device}"); return
    else:
        # ── UDP/GStreamer mode (Jetson via Ethernet) ──
        PORT = port
        pipeline_str = (
            f'udpsrc port={PORT} '
            f'caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '
            f'rtpjitterbuffer latency=200 ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! '
            f'video/x-raw, format=BGR ! '
            f'appsink drop=True max-buffers=1 sync=false'
        )
        cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"[CAM{cam_id}] ERROR: Cannot open UDP stream on port {PORT}"); return

    input_queues  = [queue.Queue(maxsize=2) for _ in ARUCO_DICTS]
    result_queue  = queue.Queue(maxsize=8)

    for i in range(len(ARUCO_DICTS)):
        threading.Thread(target=make_aruco_worker(i, input_queues[i], result_queue), daemon=True).start()

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        for btn in buttons:
            x1,y1,x2,y2 = btn["rect"]
            if x1<=x<=x2 and y1<=y<=y2:
                if btn.get("is_quit"): quit_flag[0]=True
                elif btn["action"]: btn["action"]()
                break

    WIN = win_title
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 760, 500)
    cv2.moveWindow(WIN, 0 if cam_id==0 else 760, 0)
    cv2.setMouseCallback(WIN, on_mouse)

    # ── EXP / GAM sliders (trackbars) ──────────────────────────────
    # EXP range: -100 … +100  →  trackbar 0…200, centre=100
    # GAM range:  0.3 …  3.0  →  trackbar 0…270, value = (gamma-0.3)*100
    EXP_OFFSET = 100   # trackbar centre = no shift
    GAM_OFFSET  = 30   # trackbar 0 = gamma 0.30
    cv2.createTrackbar("EXP", WIN, EXP_OFFSET, 200, lambda v: None)
    cv2.createTrackbar("GAM", WIN, 100,         270, lambda v: None)
    _prev_exp_tb = EXP_OFFSET
    _prev_gam_tb = 100

    last_marker_result=None; last_marker_ts=0; last_target=None
    lost_frames=0; prev_det_id=None; prev_time=time.time()
    _fc=0; _last_enh=None; _last_res="MIDDAY"
    _padded=None  # pre-allocated padded buffer to avoid reallocation flicker

    def force_put(q,item):
        while True:
            try: q.get_nowait()
            except queue.Empty: break
        try: q.put_nowait(item)
        except queue.Full: pass

    print(f"[CAM{cam_id}] Receiver started")

    while True:
        ret,raw=cap.read()
        if not ret: time.sleep(0.01); continue
        tac=lighting.tac_on(); _fc+=1
        gray_raw=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
        raw_bright=float(cv2.mean(gray_raw)[0])
        resolved=lighting.resolve(raw_bright); _last_res=resolved
        profile=lighting.profile(resolved); overrides=lighting._overrides[resolved]

        # ── Read EXP / GAM trackbars and apply deltas to lighting state ──
        exp_tb = cv2.getTrackbarPos("EXP", WIN)
        gam_tb = cv2.getTrackbarPos("GAM", WIN)
        if exp_tb != _prev_exp_tb:
            delta_exp = exp_tb - _prev_exp_tb
            lighting.adjust_exposure(delta_exp, resolved)
            _prev_exp_tb = exp_tb
        if gam_tb != _prev_gam_tb:
            delta_gam = round((gam_tb - _prev_gam_tb) * 0.01, 2)
            lighting.adjust_gamma(delta_gam, resolved)
            _prev_gam_tb = gam_tb

        # Always enhance every frame — reusing stale frames causes flicker
        frame=enhance_adaptive(raw,profile,tac,maps)
        _last_enh=frame

        h,w=frame.shape[:2]; cx,cy=w//2,h//2
        now=time.monotonic(); ct=time.time()
        fps=1.0/max(ct-prev_time,1e-6); prev_time=ct

        mv,pf,pa,pcx,pcy=compute_pipeline_mask(frame)
        if pf:
            mission.update_pipeline_angle(pa); mission.pipeline_found=True
            mission.update_pinger_side("right" if pcx>cx else "left")

        eg=enhance_for_aruco(raw,cam_id,tac_mode=tac)
        egs=cv2.resize(eg,(0,0),fx=DETECT_SCALE,fy=DETECT_SCALE)
        for q in input_queues: force_put(q,(egs,DETECT_SCALE))

        fresh=[]
        while True:
            try:
                r,ts=result_queue.get_nowait()
                if (now-ts)<MARKER_RESULT_TTL: fresh.append((r,ts))
            except queue.Empty: break

        if fresh:
            all_det=[(r["priority"],r["area"],r["tx"],r["ty"],r) for r,_ in fresh]
            used,groups=[False]*len(all_det),[]
            for i in range(len(all_det)):
                if used[i]: continue
                grp=[i]; used[i]=True
                for j in range(i+1,len(all_det)):
                    if not used[j] and ((all_det[i][2]-all_det[j][2])**2+(all_det[i][3]-all_det[j][3])**2)**0.5<SPATIAL_MERGE_DIST:
                        grp.append(j); used[j]=True
                groups.append(grp)
            best_r,best_s=None,(-1,-1)
            for grp in groups:
                for idx in grp:
                    if (all_det[idx][0],all_det[idx][1])>best_s:
                        best_s=(all_det[idx][0],all_det[idx][1]); best_r=all_det[idx][4]
            if best_r:
                last_marker_result,last_marker_ts=best_r,now
                aid,role=best_r["aruco_id"],best_r["role"]
                mission.register_tag(aid,role)
                if role=="PIPELINE":
                    info=PIPELINE_TAGS[aid]
                    if (now-pipeline_last_msg_ts.get(aid,0.0))>=PIPELINE_MSG_COOLDOWN:
                        pipeline_last_msg_ts[aid]=now
                        toast.show(f"[{info['order']}/4] {info['name']}",info['msg'])
                elif role=="DOCKING" and aid!=prev_det_id:
                    toast.show(f"DOCKING TAG {aid}",f"X:{best_r['tx']}  Y:{best_r['ty']}")
                prev_det_id=aid

        detected=None
        if last_marker_result and (now-last_marker_ts)<MARKER_RESULT_TTL:
            r=last_marker_result
            detected=(r["tx"],r["ty"],r["label"],r["role"],r["aruco_id"])
            aruco.drawDetectedMarkers(frame,[r["corner"]],np.array([[r["aruco_id"]]]))
        elif (now-last_marker_ts)>=MARKER_RESULT_TTL:
            last_marker_result=None

        if detected: last_target,lost_frames=detected,0
        else:
            lost_frames+=1
            if lost_frames>=MAX_LOST: last_target=last_marker_result=prev_det_id=None

        # ── HUD ──
        PLW,PLY1,PLY2=260,58,286
        draw_panel(frame,0,0,w,48)
        draw_panel(frame,8,PLY1,PLW,PLY2)
        auto_flag=(lighting.get_mode()=="AUTO")
        mode_col=LIGHTING_MODES[resolved]["btn_color"]
        title=f"CAM{cam_id}  |  {resolved}"+(" [AUTO]" if auto_flag else "")+("  [TAC]" if tac else "")
        cv2.putText(frame,title,(14,32),cv2.FONT_HERSHEY_SIMPLEX,0.55,mode_col,1,cv2.LINE_AA)
        cv2.putText(frame,f"FPS:{int(fps)}",(w-80,20),cv2.FONT_HERSHEY_SIMPLEX,0.44,C_OK,1,cv2.LINE_AA)
        cv2.putText(frame,"TELEMETRY",(18,PLY1+22),cv2.FONT_HERSHEY_SIMPLEX,0.52,C_TITLE,1,cv2.LINE_AA)
        cv2.line(frame,(18,PLY1+30),(PLW-14,PLY1+30),C_TITLE,1)
        RS,RG=PLY1+52,24
        if last_target and lost_frames<MAX_LOST:
            tx,ty,lb,role,aid=last_target
            dx,dy=tx-cx,ty-cy; cnt=abs(dx)<CENTER_TOLERANCE and abs(dy)<CENTER_TOLERANCE
            draw_row(frame,"TARGET ID",aid,RS,C_ACCENT)
            draw_row(frame,"ROLE",role,RS+RG,C_WARN)
            draw_row(frame,"DX",dx,RS+RG*2,C_TEXT)
            draw_row(frame,"DY",dy,RS+RG*3,C_TEXT)
            draw_row(frame,"PINGER",mission.pinger_side.upper(),RS+RG*4,C_TEXT)
            draw_row(frame,"PIPE ANG",f"{mission.pipeline_angle:.1f}d",RS+RG*5,C_TEXT)
            cv2.putText(frame,"CENTER LOCKED" if cnt else "ALIGNING",(18,RS+RG*6+6),cv2.FONT_HERSHEY_SIMPLEX,0.55,C_OK if cnt else C_BAD,1,cv2.LINE_AA)
            cv2.circle(frame,(tx,ty),6,C_OK,-1)
            cv2.rectangle(frame,(tx-22,ty-22),(tx+22,ty+22),C_OK,1)
            cv2.arrowedLine(frame,(cx,cy),(tx,ty),C_DIM,1,tipLength=0.12)
        else:
            for i,(l,v) in enumerate([("TARGET ID","--"),("ROLE","NONE"),("DX","--"),("DY","--"),("PINGER",mission.pinger_side.upper()),("PIPE ANG",f"{mission.pipeline_angle:.1f}d")]):
                draw_row(frame,l,v,RS+RG*i,C_BAD if i<2 else C_DIM)
            cv2.putText(frame,"NO TARGET",(18,RS+RG*6+6),cv2.FONT_HERSHEY_SIMPLEX,0.55,C_BAD,1,cv2.LINE_AA)

        draw_crosshair(frame,cx,cy)
        if pf:
            cv2.circle(frame,(pcx,pcy),5,C_PIPE,-1)
            cv2.putText(frame,"PIPE",(pcx+8,pcy-8),cv2.FONT_HERSHEY_SIMPLEX,0.40,C_PIPE,1,cv2.LINE_AA)

        BAR_H=120
        STATUS_Y=h-BAR_H-4
        draw_panel(frame,0,STATUS_Y-20,w,STATUS_Y+4,(15,15,15),0.70)
        cv2.putText(frame,f"STATUS: {'PIPELINE DETECTED' if mission.pipeline_found else 'SEARCHING'}",(14,STATUS_Y),cv2.FONT_HERSHEY_SIMPLEX,0.50,C_OK if mission.pipeline_found else C_BAD,1,cv2.LINE_AA)
        cv2.putText(frame,f"MARKERS: {mission.marker_count}",(w-160,STATUS_Y),cv2.FONT_HERSHEY_SIMPLEX,0.46,C_DIM,1,cv2.LINE_AA)

        draw_button_bar(frame,w,h,resolved,tac,_last_res,lighting,buttons,toast,quit_flag)
        draw_toast(frame,toast,now,w,h)

        # Reuse pre-allocated padded buffer — avoids memory realloc flicker every frame
        padded_h=frame.shape[0]+0
        if _padded is None or _padded.shape[0]!=padded_h or _padded.shape[1]!=frame.shape[1]:
            _padded=np.zeros((padded_h,frame.shape[1],3),dtype=np.uint8)
        _padded[:frame.shape[0],:]=frame
        cv2.imshow(WIN,_padded)
        cv2.imshow(mask_title,mv)

        key=cv2.waitKey(1)&0xFF
        if key==ord('q') or quit_flag[0]: break
        elif key==ord('1'): lighting.set_mode("BRIGHT");  toast.show("BRIGHT MODE","")
        elif key==ord('2'): lighting.set_mode("MIDDAY");  toast.show("MIDDAY MODE","")
        elif key==ord('3'): lighting.set_mode("EVENING"); toast.show("EVENING MODE","")
        elif key==ord('4'): lighting.set_mode("NIGHT");   toast.show("NIGHT MODE","")
        elif key in (ord('a'),ord('A')): lighting.set_mode("AUTO"); toast.show("AUTO MODE","")
        elif key==ord('t'):
            s=lighting.toggle_tac(); toast.show(f"TAC {'ON' if s else 'OFF'}","")
        elif key==ord('['): pass  # EXP now controlled by trackbar
        elif key==ord(']'): pass  # EXP now controlled by trackbar
        elif key==ord('-'): pass  # GAM now controlled by trackbar
        elif key in (ord('='),ord('+')): pass  # GAM now controlled by trackbar
        elif key in (ord('r'),ord('R')):
            lighting.reset_mode(_last_res)
            toast.show(f"RESET {_last_res}","")
            # Reset trackbars back to centre/default
            cv2.setTrackbarPos("EXP", WIN, EXP_OFFSET)
            cv2.setTrackbarPos("GAM", WIN, 100)
            _prev_exp_tb = EXP_OFFSET
            _prev_gam_tb = 100

    cap.release(); cv2.destroyAllWindows()

# ════════════════════════════════════════════════════════════════
#  TRANSMITTER  (GStreamer subprocess)
# ════════════════════════════════════════════════════════════════
def run_transmitter(cam_id, device, host, port, bitrate, brightness, contrast):
    cmd = (
        f"gst-launch-1.0 v4l2src device={device} do-timestamp=true "
        f"! video/x-raw,width=640,height=480,framerate=30/1 "
        f"! queue max-size-buffers=1 leaky=downstream "
        f"! videoconvert "
        f"! videobalance brightness={brightness} contrast={contrast} "
        f"! x264enc tune=zerolatency speed-preset=veryfast bitrate={bitrate} key-int-max=60 bframes=0 "
        f"! rtph264pay config-interval=1 pt=96 "
        f"! udpsink host={host} port={port} sync=false async=false"
    )
    print(f"[TX-CAM{cam_id}] Starting: {device} → {host}:{port}")
    print(f"[TX-CAM{cam_id}] CMD: {cmd}")
    try:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(f"[TX-CAM{cam_id}] {line}", end="", flush=True)
        proc.wait()
    except Exception as e:
        print(f"[TX-CAM{cam_id}] ERROR: {e}")

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    lighting_init = (args.lighting or "AUTO").upper()

    print("=" * 65)
    print("  TAC 2026 — All-in-One  |  Dual Camera")
    if args.laptop:   print("  MODE: LAPTOP ONLY  (TX + RX on this machine)")
    if args.receive:  print("  MODE: RECEIVE ONLY  (Jetson transmits via Ethernet)")
    if args.transmit: print(f"  MODE: TRANSMIT ONLY  (streaming to {args.host})")
    print("=" * 65)

    # ── TRANSMIT ONLY (run on Jetson) ───────────────────────────
    if args.transmit:
        if not args.host:
            print("[ERROR] --host <laptop_ip> is required with --transmit")
            sys.exit(1)
        threads = []
        t0 = threading.Thread(target=run_transmitter,
             args=(0, args.dev0, args.host, 5000, args.bitrate, 0.050, 1.100), daemon=True)
        t1 = threading.Thread(target=run_transmitter,
             args=(1, args.dev1, args.host, 5001, args.bitrate, 0.080, 1.150), daemon=True)
        t0.start(); t1.start()
        try: t0.join(); t1.join()
        except KeyboardInterrupt: print("\n[INFO] Transmitters stopped.")
        return

    # ── LAPTOP ONLY — direct camera capture, no GStreamer UDP ──
    if args.laptop:
        print("[INFO] Laptop mode — reading cameras directly (no GStreamer)")
        print("[INFO] Starting receivers...")
        rx1 = threading.Thread(
            target=run_receiver,
            args=(1, 5001,
                  "CAM1 Lenovo  |  Adaptive Lighting  [CLICK BUTTONS]",
                  "Cam1 Pipeline Mask",
                  UNDISTORT_MAPS[1], False, lighting_init),
            kwargs={"direct_device": args.dev1},
            daemon=True
        )
        rx1.start()
        time.sleep(0.5)
        run_receiver(0, 5000,
                     "CAM0 HP Wide  |  Adaptive Lighting  [CLICK BUTTONS]",
                     "Cam0 Pipeline Mask",
                     UNDISTORT_MAPS[0], False, lighting_init,
                     direct_device=args.dev0)
        rx1.join(timeout=2)
        print("[INFO] All done.")
        return

    jetson_mode = args.receive  # higher quality when receiving from Jetson

    # ── START RECEIVERS (Jetson/UDP mode) ───────────────────────
    print("[INFO] Starting receivers...")
    rx1 = threading.Thread(
        target=run_receiver,
        args=(1, 5001,
              "CAM1 Lenovo  |  Adaptive Lighting  [CLICK BUTTONS]",
              "Cam1 Pipeline Mask",
              UNDISTORT_MAPS[1], jetson_mode, lighting_init),
        daemon=True
    )
    rx1.start()
    time.sleep(0.3)

    # Run cam0 in main thread
    run_receiver(0, 5000,
                 "CAM0 HP Wide  |  Adaptive Lighting  [CLICK BUTTONS]",
                 "Cam0 Pipeline Mask",
                 UNDISTORT_MAPS[0], jetson_mode, lighting_init)

    rx1.join(timeout=2)
    print("[INFO] All done.")

if __name__ == "__main__":
    main()
