#!/usr/bin/env python3
"""
go2_dpg_app.py — DearPyGui controller & live video viewer for Unitree Go2

Revision highlights
-------------------
* **Full-res RGBA textures** (no tiling/compression artifacts)
* **StopMove** on zero input
* Video auto-scales with its window
* All earlier features remain (speed slider, mode selector, flips, hand/stand, emotes, WASD/QE control)

Run with:  `python go2_dpg_app.py`
Requires: dearpygui, opencv-python, numpy, go2_webrtc_driver
"""

from __future__ import annotations
import asyncio
import threading
import time
import cv2
import numpy as np
from queue import Queue
import dearpygui.dearpygui as dpg

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
ROT_SPEED = 1               # rad/s for Q/E
CMD_RATE_HZ = 10            # Move publish rate
PADDING = 16                # Inner padding when fitting video image
INITIAL_FRAME_TIMEOUT = 5.0 # seconds to wait for first frame

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class Go2Controller:
    """Handles WebRTC, video frames, and robot commands."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.frame_queue: "Queue[np.ndarray]" = Queue()
        self.conn: Go2WebRTCConnection | None = None
        self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._vel_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    # -- Public API --
    def start(self):
        self._thread.start()

    def pop_frame(self):
        return self.frame_queue.get() if not self.frame_queue.empty() else None

    def set_velocity_component(self, comp: str, val: float):
        with self._vel_lock:
            self.velocity[comp] = val

    def stop(self):
        # send StopMove immediately
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["StopMove"]}
        ))

    # -- RPC Helpers --
    def _schedule(self, coro):
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(coro))

    async def _publish(self, topic: str, payload: dict):
        await self.conn.datachannel.pub_sub.publish_request_new(topic, payload)

    def set_mode(self, name: str):
        self._schedule(self._publish(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": name}}
        ))

    def flip(self, direction: str):
        api = SPORT_CMD[{
            "front": "FrontFlip",
            "back":  "BackFlip",
            "left":  "LeftFlip",
            "right": "RightFlip"
        }[direction]]
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": api, "parameter": {"data": True}}
        ))

    def handstand(self, on: bool):
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["StandOut"], "parameter": {"data": on}}
        ))

    def stand(self, on: bool):
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["StandUp"], "parameter": {"data": on}}
        ))

    def emote_hello(self):
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Hello"]}
        ))

    # -- Async Loop & Streaming --
    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_setup())
        self.loop.run_forever()

    async def _async_setup(self):
        self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
        await self.conn.connect()
        self.conn.video.switchVideoChannel(True)
        self.conn.video.add_track_callback(self._on_track)
        self.loop.create_task(self._move_sender())

    async def _on_track(self, track):
        while True:
            frame = await track.recv()
            self.frame_queue.put(frame.to_ndarray(format="bgr24"))

    async def _move_sender(self):
        """Send Move if any velocity, else StopMove."""
        dt = 1.0 / CMD_RATE_HZ
        while True:
            await asyncio.sleep(dt)
            with self._vel_lock:
                vx, vy, vz = self.velocity.values()

            if abs(vx) + abs(vy) + abs(vz) > 1e-3:
                await self._publish(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["Move"],
                     "parameter": {"x": vx, "y": vy, "z": vz}}
                )
            else:
                await self._publish(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["StopMove"]}
                )

    def close(self):
        if self.conn:
            self._schedule(self.conn.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(2)


# ---------------------------------------------------------------------------
# DearPyGui UI
# ---------------------------------------------------------------------------
def launch_gui():
    ctl = Go2Controller()
    ctl.start()

    # — wait for first frame —
    first_frame = None
    t0 = time.time()
    while first_frame is None and (time.time() - t0) < INITIAL_FRAME_TIMEOUT:
        first_frame = ctl.pop_frame()
        time.sleep(0.01)
    if first_frame is None:
        raise RuntimeError("No video frame received within timeout")

    # convert to RGB→RGBA float32
    rgb      = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    tex_h, tex_w = rgb.shape[:2]
    rgb_f32  = rgb.astype(np.float32) / 255.0                  # (H, W, 3)
    alpha    = np.ones((tex_h, tex_w, 1), dtype=np.float32)    # (H, W, 1)
    rgba_f32 = np.concatenate([rgb_f32, alpha], axis=2).flatten()

    # setup DPG
    dpg.create_context()
    dpg.create_viewport(title="Go2 Controller", width=1320, height=860)

    # pre-allocate full-res RGBA dynamic texture
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(
            tex_w,
            tex_h,
            rgba_f32,
            tag="video_tex"
        )

    # Video window
    with dpg.window(label="Video",
                    tag="video_window",
                    pos=(10, 10),
                    width=960,
                    height=720,
                    no_scrollbar=True):
        dpg.add_image("video_tex", tag="video_img")

    # Telemetry
    with dpg.window(label="Telemetry", pos=(990, 10), width=300, height=260):
        dpg.add_text("Keyboard: WASD translate | Q/E yaw")
        dpg.add_spacer(height=6)
        dpg.add_text("Cmd Vel:")
        vel_lbl = dpg.add_text("x:0 y:0 z:0", tag="vel_lbl")
        dpg.add_spacer(height=8)
        dpg.add_text("Translation Speed:")
        dpg.add_slider_float(tag="speed_slider",
                             default_value=0.3,
                             min_value=0.0,
                             max_value=1.0,
                             width=220)

    # Actions
    with dpg.window(label="Actions", pos=(990, 290), width=300, height=480):
        dpg.add_text("Mode:")
        dpg.add_combo(("normal", "ai"),
                      default_value="normal",
                      width=120,
                      callback=lambda s, a: ctl.set_mode(a))
        dpg.add_separator()

        dpg.add_text("Flips:")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Front", callback=lambda: ctl.flip("front"))
            dpg.add_button(label="Back",  callback=lambda: ctl.flip("back"))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Left",  callback=lambda: ctl.flip("left"))
            dpg.add_button(label="Right", callback=lambda: ctl.flip("right"))
        dpg.add_separator()

        dpg.add_text("Handstand:")
        with dpg.group(horizontal=True):
            dpg.add_button(label="On",  callback=lambda: ctl.handstand(True))
            dpg.add_button(label="Off", callback=lambda: ctl.handstand(False))
        dpg.add_text("Stand:")
        with dpg.group(horizontal=True):
            dpg.add_button(label="On",  callback=lambda: ctl.stand(True))
            dpg.add_button(label="Off", callback=lambda: ctl.stand(False))
        dpg.add_separator()

        dpg.add_text("Emotes:")
        dpg.add_button(label="Hello", callback=ctl.emote_hello)

    # Key handlers + immediate StopMove on all-zero
    def key_press(_, key):
        speed = dpg.get_value("speed_slider")
        if   key == dpg.mvKey_W: ctl.set_velocity_component("x",  speed)
        elif key == dpg.mvKey_S: ctl.set_velocity_component("x", -speed)
        elif key == dpg.mvKey_A: ctl.set_velocity_component("y",  speed)
        elif key == dpg.mvKey_D: ctl.set_velocity_component("y", -speed)
        elif key == dpg.mvKey_Q: ctl.set_velocity_component("z",  ROT_SPEED)
        elif key == dpg.mvKey_E: ctl.set_velocity_component("z", -ROT_SPEED)

    def key_release(_, key):
        if key in (dpg.mvKey_W, dpg.mvKey_S):
            ctl.set_velocity_component("x", 0.0)
        if key in (dpg.mvKey_A, dpg.mvKey_D):
            ctl.set_velocity_component("y", 0.0)
        if key in (dpg.mvKey_Q, dpg.mvKey_E):
            ctl.set_velocity_component("z", 0.0)

        vx, vy, vz = ctl.velocity.values()
        if abs(vx) + abs(vy) + abs(vz) < 1e-6:
            ctl.stop()

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=key_press)
        dpg.add_key_release_handler(callback=key_release)

    # Per-frame UI update
    def update_ui():
        frame = ctl.pop_frame()
        if frame is not None:
            rgb2   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h2, w2 = rgb2.shape[:2]
            f32    = rgb2.astype(np.float32) / 255.0
            a2     = np.ones((h2, w2, 1), dtype=np.float32)
            rgba2  = np.concatenate([f32, a2], axis=2).flatten()
            dpg.set_value("video_tex", rgba2)

            # keep aspect-fit scaling
            win_w, win_h = dpg.get_item_rect_size("video_window")
            aw = max(0, win_w - PADDING)
            ah = max(0, win_h - PADDING - 20)
            asp = w2 / h2
            tw = aw
            th = tw / asp
            if th > ah:
                th = ah
                tw = th * asp
            dpg.configure_item("video_img", width=int(tw), height=int(th))

        vx, vy, vz = ctl.velocity.values()
        dpg.set_value("vel_lbl", f"x:{vx:.2f} y:{vy:.2f} z:{vz:.2f}")

    # Main loop
    dpg.set_viewport_vsync(True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        update_ui()
        dpg.render_dearpygui_frame()

    # Cleanup
    ctl.close()
    dpg.destroy_context()


if __name__ == "__main__":
    launch_gui()
