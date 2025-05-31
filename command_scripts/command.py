
from __future__ import annotations
import asyncio
import threading
import time
import cv2
import numpy as np
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD



CMD_RATE_HZ = 1           # Move publish rate


class DogController: 
    def __init__(self): 
        self.loop = asyncio.new_event_loop()
        self.conn: Go2WebRTCConnection | None = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)


    # ------- PUBLIC 
    def start(self): 
        self._thread.start()

    def set_mode(self, name: str):
        self._schedule(self._publish(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": name}}
        ))

    def handstand(self, on: bool):
        self._schedule(self._publish(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["StandOut"], "parameter": {"data": on}}
        ))


    async def move(self): 
        dt = 1.0 / CMD_RATE_HZ

        for _ in range(2): 
            await asyncio.sleep(dt)
            await self._publish(
                RTC_TOPIC["RT_MOD"],
                {"api_id": SPORT_CMD["Move"],
                 "parameter": {"x": 1, "y": 0, "z": 0}}
            )

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


    def close(self):
        if self.conn:
            self._schedule(self.conn.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(2)

    # ------ PRIVATE 

    def _run_loop(self): 
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_setup())
        self.loop.run_forever()

    def _schedule(self, coro):
        """ Create a task for the thread to run"""
        self.loop.call_soon_threadsafe(lambda: asyncio.create_task(coro))

    async def _publish(self, topic: str, payload: dict):
        """ Send the command to the dog!!!"""
        await self.conn.datachannel.pub_sub.publish_request_new(topic, payload)

    async def _async_setup(self):
        """ Connect ot the dog """
        self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
        await self.conn.connect()






