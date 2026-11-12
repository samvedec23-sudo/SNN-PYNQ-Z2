#!/usr/bin/env python3
"""
fpga_ee.py - driver for the ee_axil register block (see rtl/ee_axil.v), plus a software stand-in for tests.

Works with BOTH bitstreams:
  * v2 (what you have now)   : decision unit only
  * v3 (LED build)           : + time-to-first-spike register, + 4 user LEDs, + RGB LED state
`FpgaEE.has_extras` tells you which one is loaded (it reads a capability marker; old builds return 0 there).
"""
MAGIC = 0xEE01
REG_CTRL, REG_STATUS, REG_FIRST, REG_LED = 0x00, 0x04, 0x08, 0x0C
CAP_MARK = 0xA5


class FpgaEE:
    def __init__(self, mmio, t_min, margin, t_max=32):
        self.m = mmio
        self.t_min, self.margin, self.t_max = t_min, margin, t_max
        self.cfg = (t_min << 12) | (margin << 18) | (t_max << 24)
        s = self.status()
        if (s >> 16) != MAGIC:
            raise RuntimeError("Wrong block / wrong address: STATUS read 0x%08x, expected 0xEE01xxxx. "
                               "Check the base address in Vivado's Address Editor (use --base)." % s)
        self.has_extras = ((self.m.read(REG_FIRST) >> 8) & 0xFF) == CAP_MARK

    def status(self):
        return self.m.read(REG_STATUS)

    def start(self):
        self.m.write(REG_CTRL, self.cfg | (1 << 11))

    def step(self, spikes10):
        word = int(sum(int(b) << c for c, b in enumerate(spikes10)))
        self.m.write(REG_CTRL, self.cfg | (1 << 10) | word)
        s = self.status()
        return bool(s & 1), (s >> 1) & 0xF, (s >> 5) & 0x3F

    # ---------------- v3 only ----------------
    def first_spike(self):
        """(seen, 1-based timestep of the first output spike). Hardware-measured. (False, 0) if none / not supported."""
        if not self.has_extras:
            return False, 0
        r = self.m.read(REG_FIRST)
        return bool(r & 0x40), r & 0x3F

    def set_leds(self, code4):
        if self.has_extras:
            self.m.write(REG_LED, int(code4) & 0xF)

    def rgb_state(self, done, steps_used):
        """Software mirror of the hardware-driven RGB LEDs (for the on-screen virtual LEDs)."""
        seen, _ = self.first_spike()
        return dict(rgb0_g=seen, rgb1_g=bool(done and steps_used != self.t_max), rgb1_r=bool(done and steps_used == self.t_max))


class MockMMIO:
    """Software stand-in with the same register protocol (for testing without a board).
    legacy=True behaves like the v2 bitstream (no first-spike / LED registers)."""
    def __init__(self, legacy=False):
        self.legacy = legacy
        self.cnt = [0] * 10; self.t = 0; self.done = 0; self.pred = 0; self.steps = 0
        self.stepcnt = 0; self.first_seen = 0; self.first_step = 0; self.led = 0; self.tmax = 32

    def write(self, off, val):
        if off == REG_LED:
            if not self.legacy:
                self.led = val & 0xF
            return
        assert off == REG_CTRL
        tmin, marg, tmax = (val >> 12) & 63, max((val >> 18) & 63, 1), (val >> 24) & 63
        self.tmax = tmax
        if val & (1 << 11):                                   # START
            self.cnt = [0] * 10; self.t = 0; self.done = 0; self.pred = 0; self.steps = 0
            self.stepcnt = 0; self.first_seen = 0; self.first_step = 0
        elif (val & (1 << 10)) and not self.done:             # STEP
            self.stepcnt += 1
            if (val & 0x3FF) and not self.first_seen:
                self.first_seen, self.first_step = 1, self.stepcnt
            for c in range(10):
                self.cnt[c] += (val >> c) & 1
            self.t += 1
            top = max(self.cnt); arg = self.cnt.index(top)
            second = max(v for i, v in enumerate(self.cnt) if i != arg)
            if (self.t >= tmin and top - second >= marg) or self.t >= tmax:
                self.done, self.pred, self.steps = 1, arg, self.t

    def read(self, off):
        if off == REG_STATUS:
            return (MAGIC << 16) | (self.steps << 5) | (self.pred << 1) | self.done
        if off == REG_FIRST:
            return 0 if self.legacy else (CAP_MARK << 8) | (self.first_seen << 6) | self.first_step
        if off == REG_LED:
            return 0 if self.legacy else self.led
        return 0


def open_overlay(bit, base=None):
    """Load the overlay on the board and return an MMIO for the ee_axil block."""
    from pynq import Overlay, MMIO
    ol = Overlay(bit)
    if base is None:
        cands = [k for k, v in ol.ip_dict.items() if "ee_axil" in (k + str(v.get("type", ""))).lower()]
        if not cands:
            raise SystemExit("Could not find ee_axil in the overlay. Run: print(list(ol.ip_dict)) and pass base=0x....")
        base = ol.ip_dict[cands[0]]["phys_addr"]
        print("found %s at 0x%08x" % (cands[0], base))
    return MMIO(base, 0x1000)
