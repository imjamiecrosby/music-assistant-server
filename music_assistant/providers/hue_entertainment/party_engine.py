"""
Section-aware "dance party" effects engine for Hue Entertainment.

Where the classic analyzer reacts to each moment on its own, this engine follows the
song's shape. It tracks how intense the music is relative to the track itself and
picks a scene to match:

- chill: a slow colour wave rolling around the room (intros, breakdowns)
- groove: a beat chase that walks a highlight from light to light
- build: lights fill from the floor up while beat flashes grow and whiten
- peak: high-contrast blasts on every beat, alternating sides of the room

Jumping into the peak scene from a quiet or rising passage fires a drop: a full-room
white hit and an immediate palette jump. Each light is limited to three flash
onsets in any one second so the show stays within common photosensitivity guidance.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hue_entertainment import LightColorCommand

from .analyzer import build_palette

if TYPE_CHECKING:
    from aiosendspin.models.visualizer import BeatTiming
    from hue_entertainment import LightChannel

SCENE_CHILL = "chill"
SCENE_GROOVE = "groove"
SCENE_BUILD = "build"
SCENE_PEAK = "peak"

# Spectrum handling matches the classic analyzer: gate the server's soft noise floor,
# lift with a square-root curve, and ignore the hissy top bins for loudness.
_NOISE_GATE = 0.02
_SPECTRUM_GAMMA = 0.5
_LOUDNESS_BINS = 11
_BASS_BINS = 2
_HIGH_BINS = (9, 15)

# Time constants (seconds) for the energy trackers. ``level`` follows loudness,
# ``ceiling``/``floor`` learn the track's own dynamic range (fast to widen, slow to
# narrow) so intensity is judged relative to the song rather than absolute volume.
_LEVEL_TAU = 0.5
_RANGE_ATTACK_TAU = 1.0
_RANGE_RELEASE_TAU = 40.0
_MIN_RANGE = 0.08
_BASS_BASE_TAU = 3.0
_BASS_DIP_TAU = 0.1
# Bin-to-bin noise alone lifts bass a little above its dip; ignore that much.
_BASS_DEADZONE = 0.03
_PUNCH_TAU = 1.5
_PUNCH_SCALE = 6.0
_INTENSITY_TAU = 0.6
_TENSION_FAST_TAU = 0.5
_TENSION_SLOW_TAU = 6.0
_INTENSITY_HISTORY_TAU = 4.0

# Scene thresholds on the 0-1 intensity, with hysteresis so scenes don't flicker.
_PEAK_ENTER = 0.68
_PEAK_EXIT = 0.52
_GROOVE_ENTER = 0.40
_GROOVE_EXIT = 0.32
_BUILD_RISE = 0.025
# Once building, hold the scene while tension is still climbing at all; slow builds
# rise too gently to stay above the entry threshold.
_BUILD_HOLD = 0.005
_BUILD_CONFIRM_US = 1_500_000
_BUILD_FULL_US = 12_000_000
_SCENE_MIN_DWELL_US = 1_500_000
_SCENE_FADE_US = 600_000
# A peak entered while the slow intensity history is below this counts as a drop.
_DROP_HISTORY_MAX = 0.55
# Tension jump that marks a drop (builds rise around 0.03-0.04, drops spike past 0.1),
# and the kick punch that has to come with it.
_DROP_JUMP = 0.08
_DROP_MIN_PUNCH = 0.15

# Flash limiter: at most this many flash onsets per light in any rolling window.
_FLASH_LIMIT = 3
_FLASH_WINDOW_US = 1_000_000
_ENVELOPE_CUTOFF = 0.01

# A beat that is already this far in the past when first seen is skipped instead of
# fired late, so a fresh schedule doesn't burst a backlog of flashes.
_BEAT_LATE_US = 100_000
_DEFAULT_BEAT_US = 500_000
_PEAK_MIN_STRENGTH = 0.1

_WHITE = (0.85, 0.85, 0.85)


@dataclass(slots=True)
class _Beat:
    timestamp_us: int
    is_downbeat: bool


@dataclass(slots=True)
class _Flash:
    """Per-light flash envelope plus the onsets used by the rate limiter."""

    amp: float = 0.0
    white: float = 0.0
    start_us: int = 0
    decay_us: int = 1
    onsets: deque[int] | None = None


class PartyEngine:
    """Render scene-based dance-party lighting from Sendspin visualizer features."""

    def __init__(
        self,
        channels: list[LightChannel],
        color_mode: str = "party",
        brightness: int = 100,
    ) -> None:
        """
        Initialize the engine.

        :param channels: Entertainment area channels with their room positions.
        :param color_mode: Accepted for interface compatibility with the classic analyzer.
        :param brightness: Overall brightness, 0-100.
        """
        self._channels = channels
        self._color_mode = color_mode
        self._brightness = max(0, min(100, brightness)) / 100.0
        self._server_palette: dict[str, tuple[int, int, int] | None] = {}
        self._palette = build_palette({})

        self._ring, self._height, self._side = _room_layout(channels)

        self._pending_spectrum: deque[tuple[int, list[int]]] = deque()
        self._pending_peaks: deque[tuple[int, float]] = deque()
        self._beats: deque[_Beat] = deque()
        self._last_fired_us: int | None = None
        self._beat_count = 0
        self._beat_period_us = _DEFAULT_BEAT_US
        self._last_beat_us: int | None = None

        self._last_spectrum_ts: int | None = None
        self._level = 0.0
        self._ceiling: float | None = None
        self._floor: float | None = None
        self._bass_base = 0.0
        self._bass_transient = 0.0
        self._punch = 0.0
        self._high = 0.0
        self._intensity = 0.0
        self._intensity_history = 0.0
        self._tension_fast = 0.0
        self._tension_slow = 0.0
        self._rising_since_us: int | None = None

        self._scene = SCENE_CHILL
        self._prev_scene = SCENE_CHILL
        self._scene_since_us = 0
        self._build_started_us: int | None = None

        self._palette_target = 0.0
        self._palette_pos = 0.0
        self._last_render_us: int | None = None

        self._flashes = [_Flash(onsets=deque()) for _ in channels]

    def update_settings(self, color_mode: str | None = None, brightness: int | None = None) -> None:
        """Update settings without resetting the show."""
        if color_mode is not None:
            self._color_mode = color_mode
        if brightness is not None:
            self._brightness = max(0, min(100, brightness)) / 100.0

    def debug_state(self) -> dict[str, float | str]:
        """Return the current scene and intensity for offline tuning."""
        return {"scene": self._scene, "intensity": round(self._intensity, 3)}

    # -- Input events --

    def apply_color_palette(self, update: dict[str, tuple[int, int, int] | None]) -> None:
        """Merge a color@v1 update and rebuild the palette."""
        self._server_palette.update(update)
        self._palette = build_palette(self._server_palette)

    def apply_spectrum(self, bins: list[int], timestamp_us: int) -> None:
        """Queue a spectrum frame until playback reaches its timestamp."""
        self._pending_spectrum.append((timestamp_us, bins))

    def apply_peak(self, strength: int, timestamp_us: int) -> None:
        """Queue an onset peak (strength 0-255) until playback reaches it."""
        normalized = max(0.0, min(1.0, strength / 255.0))
        if normalized >= _PEAK_MIN_STRENGTH:
            self._pending_peaks.append((timestamp_us, normalized))

    def push_beats(self, beats: list[BeatTiming]) -> None:
        """Append a segment of the server's beat schedule."""
        for beat in beats:
            if self._beats and beat.timestamp_us <= self._beats[-1].timestamp_us:
                continue
            self._beats.append(_Beat(beat.timestamp_us, beat.is_downbeat))

    def clear_beat_schedule(self) -> None:
        """Drop the beat schedule (track change); energy tracking carries on."""
        self._beats.clear()

    def clear_beats(self) -> None:
        """Drop every queued feature and running flash (seek or stream end)."""
        self.clear_beat_schedule()
        self._pending_spectrum.clear()
        self._pending_peaks.clear()
        self._last_spectrum_ts = None
        self._last_fired_us = None
        self._last_beat_us = None
        for flash in self._flashes:
            flash.amp = 0.0

    # -- Rendering --

    def render(self, now_us: int) -> list[LightColorCommand]:
        """Render every channel at server-clock time ``now_us``."""
        while self._pending_spectrum and self._pending_spectrum[0][0] <= now_us:
            ts, bins = self._pending_spectrum.popleft()
            self._consume_spectrum(bins, ts)
        self._update_scene(now_us)
        self._fire_beats(now_us)
        self._consume_peaks(now_us)
        self._advance_palette(now_us)
        if not self._channels:
            return []

        current = self._render_scene(self._scene, now_us)
        fade = (now_us - self._scene_since_us) / _SCENE_FADE_US
        if 0.0 <= fade < 1.0 and self._prev_scene != self._scene:
            previous = self._render_scene(self._prev_scene, now_us)
            current = [
                (_lerp(pc, cc, fade), pl + (cl - pl) * fade)
                for (pc, pl), (cc, cl) in zip(previous, current, strict=True)
            ]

        commands: list[LightColorCommand] = []
        for i, channel in enumerate(self._channels):
            color, level = current[i]
            amp, white = self._envelope(i, now_us)
            color = _lerp(color, _WHITE, min(1.0, white))
            scale = max(0.0, min(1.0, level + amp)) * self._brightness
            commands.append(_to_command(channel.channel_id, color, scale))
        return commands

    def _render_scene(
        self, scene: str, now_us: int
    ) -> list[tuple[tuple[float, float, float], float]]:
        """Return the base (color, level) per light for ``scene``, before flashes."""
        n = len(self._channels)
        pos = self._palette_pos
        if scene == SCENE_CHILL:
            # A slow wave rolling around the room, one cycle every two bars.
            bar_s = 4 * self._beat_period_us / 1_000_000
            cycle = now_us / 1_000_000 / (2 * bar_s)
            return [
                (
                    self._sample(pos + self._ring[i] * 0.6),
                    0.32 + 0.08 * math.sin(2 * math.pi * (cycle + self._ring[i])),
                )
                for i in range(n)
            ]
        if scene == SCENE_BUILD:
            progress = self._build_progress(now_us)
            color = _lerp(self._sample(pos), _WHITE, 0.6 * progress)
            fill = 0.15 + progress
            return [
                (color, 0.25 + 0.45 * progress if self._height[i] <= fill else 0.08)
                for i in range(n)
            ]
        if scene == SCENE_PEAK:
            return [(self._sample(pos + self._side[i]), 0.22) for i in range(n)]
        return [
            (self._sample(pos + self._side[i] + self._ring[i] * 0.25), 0.38 + 0.15 * self._punch)
            for i in range(n)
        ]

    # -- Feature tracking --

    def _consume_spectrum(self, bins: list[int], ts: int) -> None:
        """Update the energy trackers from one spectrum frame."""
        dt = 1 / 30 if self._last_spectrum_ts is None else (ts - self._last_spectrum_ts) / 1e6
        dt = max(0.0, min(0.2, dt))
        self._last_spectrum_ts = ts
        if not bins:
            return
        norm = [_lift(b) for b in bins]
        loud = _mean(norm[:_LOUDNESS_BINS])
        bass = _mean(norm[:_BASS_BINS])
        high = _mean(norm[_HIGH_BINS[0] : _HIGH_BINS[1]]) if len(norm) > _HIGH_BINS[0] else 0.0

        tension = 0.5 * loud + 0.5 * high
        if self._ceiling is None or self._floor is None:
            # Seed every tracker from the first frame so the opening seconds don't
            # read as a huge jump from silence.
            self._level = self._ceiling = self._floor = loud
            self._bass_base = bass
            self._tension_fast = self._tension_slow = tension
        self._level = _ema(self._level, loud, dt, _LEVEL_TAU)
        up_tau = _RANGE_ATTACK_TAU if self._level > self._ceiling else _RANGE_RELEASE_TAU
        self._ceiling = _ema(self._ceiling, self._level, dt, up_tau)
        down_tau = _RANGE_ATTACK_TAU if self._level < self._floor else _RANGE_RELEASE_TAU
        self._floor = _ema(self._floor, self._level, dt, down_tau)
        span = self._ceiling - self._floor
        relative = (self._level - self._floor) / span if span >= _MIN_RANGE else 0.5

        # The bass base follows the dip between kicks (quick to fall, slow to
        # rise), so a steady four-on-the-floor still reads as strong punch.
        base_tau = _BASS_DIP_TAU if bass < self._bass_base else _BASS_BASE_TAU
        self._bass_base = _ema(self._bass_base, bass, dt, base_tau)
        self._bass_transient = max(0.0, bass - self._bass_base - _BASS_DEADZONE)
        self._punch = _ema(
            self._punch, min(1.0, self._bass_transient * _PUNCH_SCALE), dt, _PUNCH_TAU
        )
        self._high = high

        raw = 0.5 * max(0.0, min(1.0, relative)) + 0.5 * self._punch
        self._intensity = _ema(self._intensity, raw, dt, _INTENSITY_TAU)
        self._intensity_history = _ema(
            self._intensity_history, self._intensity, dt, _INTENSITY_HISTORY_TAU
        )
        self._tension_fast = _ema(self._tension_fast, tension, dt, _TENSION_FAST_TAU)
        self._tension_slow = _ema(self._tension_slow, tension, dt, _TENSION_SLOW_TAU)

    def _update_scene(self, now_us: int) -> None:
        """Pick the scene for this frame and fire a drop on a quiet-to-peak jump."""
        rise = self._tension_fast - self._tension_slow
        if rise > _BUILD_RISE:
            if self._rising_since_us is None:
                self._rising_since_us = now_us
        else:
            self._rising_since_us = None
        building = (
            self._rising_since_us is not None
            and now_us - self._rising_since_us >= _BUILD_CONFIRM_US
        )

        current = self._scene
        intensity = self._intensity
        # A sharp jump in tension with kicks behind it is a drop, even while the
        # relative intensity is still catching up.
        jump = rise > _DROP_JUMP and self._punch > _DROP_MIN_PUNCH
        if (
            intensity >= _PEAK_ENTER
            or (jump and intensity >= _GROOVE_ENTER)
            or (current == SCENE_PEAK and intensity >= _PEAK_EXIT)
        ):
            target = SCENE_PEAK
        elif building or (current == SCENE_BUILD and rise > _BUILD_HOLD):
            target = SCENE_BUILD
        elif intensity >= _GROOVE_ENTER or (
            current in (SCENE_GROOVE, SCENE_PEAK) and intensity >= _GROOVE_EXIT
        ):
            target = SCENE_GROOVE
        else:
            target = SCENE_CHILL

        if target == current:
            return
        is_drop = target == SCENE_PEAK and (
            current == SCENE_BUILD or self._intensity_history < _DROP_HISTORY_MAX
        )
        # A drop lands immediately; any other change waits out the minimum dwell.
        if not is_drop and now_us - self._scene_since_us < _SCENE_MIN_DWELL_US:
            return
        if is_drop:
            self._drop(now_us)
        if target == SCENE_BUILD:
            self._build_started_us = now_us
        self._prev_scene = current
        self._scene = target
        self._scene_since_us = now_us

    def _drop(self, now_us: int) -> None:
        """Hit every light with a white flash and jump the palette."""
        for i in range(len(self._channels)):
            self._trigger(i, now_us, amp=1.0, decay_us=500_000, white=1.0)
        self._palette_target = math.floor(self._palette_target) + 2.0
        self._palette_pos = self._palette_target

    def _fire_beats(self, now_us: int) -> None:
        """Fire every scheduled beat the playhead has passed since the last frame."""
        while self._beats and self._beats[0].timestamp_us <= now_us:
            beat = self._beats.popleft()
            already_fired = (
                self._last_fired_us is not None and beat.timestamp_us <= self._last_fired_us
            )
            if already_fired or now_us - beat.timestamp_us > _BEAT_LATE_US:
                continue
            self._on_beat(beat.timestamp_us, beat.is_downbeat, now_us)
            self._last_fired_us = beat.timestamp_us

    def _consume_peaks(self, now_us: int) -> None:
        """Use onsets as beats when no schedule is active, and as sparkles at the peak."""
        while self._pending_peaks and self._pending_peaks[0][0] <= now_us:
            ts, strength = self._pending_peaks.popleft()
            if now_us - ts > _BEAT_LATE_US:
                continue
            if not self._beats and (
                self._last_beat_us is None or ts - self._last_beat_us > self._beat_period_us // 2
            ):
                self._on_beat(ts, self._beat_count % 4 == 0, now_us)
            elif self._scene == SCENE_PEAK and strength > 0.6:
                top = max(range(len(self._channels)), key=lambda i: self._height[i], default=None)
                if top is not None:
                    self._trigger(top, now_us, amp=0.4 * strength, decay_us=150_000, white=0.8)

    def _on_beat(self, ts: int, is_downbeat: bool, now_us: int) -> None:
        """Advance the beat counters and trigger the current scene's beat flashes."""
        if self._last_beat_us is not None:
            interval = ts - self._last_beat_us
            if 200_000 <= interval <= 1_500_000:
                self._beat_period_us = int(0.8 * self._beat_period_us + 0.2 * interval)
        self._last_beat_us = ts
        self._beat_count += 1
        n = len(self._channels)
        if n == 0:
            return

        if self._scene == SCENE_CHILL:
            if is_downbeat:
                self._palette_target += 0.25
            return
        if self._scene == SCENE_GROOVE:
            if is_downbeat:
                self._palette_target += 1.0
            order = sorted(range(n), key=lambda i: self._ring[i])
            lead = order[self._beat_count % n]
            self._trigger(lead, now_us, amp=0.55, decay_us=300_000, white=0.1)
            return
        if self._scene == SCENE_BUILD:
            progress = self._build_progress(now_us)
            fill = 0.15 + progress
            for i in range(n):
                if self._height[i] <= fill:
                    self._trigger(
                        i,
                        now_us,
                        amp=0.25 + 0.5 * progress,
                        decay_us=int(300_000 - 150_000 * progress),
                        white=0.3 + 0.5 * progress,
                    )
            return
        if is_downbeat:
            self._palette_target += 1.0
            for i in range(n):
                self._trigger(i, now_us, amp=0.75, decay_us=320_000, white=0.25)
            return
        group = self._beat_count % 2
        for i in range(n):
            if (self._side[i] >= 0.5) == bool(group):
                self._trigger(i, now_us, amp=0.75, decay_us=220_000, white=0.1)

    def _advance_palette(self, now_us: int) -> None:
        """Ease the palette position toward its target at a scene-dependent speed."""
        dt = 0.0 if self._last_render_us is None else (now_us - self._last_render_us) / 1e6
        self._last_render_us = now_us
        dt = max(0.0, min(0.2, dt))
        if self._scene == SCENE_CHILL:
            self._palette_target += 0.04 * dt
        tau = {SCENE_CHILL: 2.0, SCENE_GROOVE: 0.3, SCENE_BUILD: 1.0}.get(self._scene, 0.08)
        self._palette_pos = _ema(self._palette_pos, self._palette_target, dt, tau)

    def _build_progress(self, now_us: int) -> float:
        """Return 0-1 progress through the current build-up."""
        if self._build_started_us is None:
            return 0.0
        return max(0.0, min(1.0, (now_us - self._build_started_us) / _BUILD_FULL_US))

    # -- Flashes --

    def _trigger(self, index: int, now_us: int, *, amp: float, decay_us: int, white: float) -> None:
        """Start a flash on one light unless it would break the flash-rate limit."""
        flash = self._flashes[index]
        onsets = flash.onsets
        assert onsets is not None
        while onsets and now_us - onsets[0] >= _FLASH_WINDOW_US:
            onsets.popleft()
        if len(onsets) >= _FLASH_LIMIT:
            return
        onsets.append(now_us)
        flash.amp = amp
        flash.white = white
        flash.start_us = now_us
        flash.decay_us = max(1, decay_us)

    def _envelope(self, index: int, now_us: int) -> tuple[float, float]:
        """Return the (brightness boost, whiteness) of a light's running flash."""
        flash = self._flashes[index]
        if flash.amp <= 0.0:
            return 0.0, 0.0
        elapsed = max(0, now_us - flash.start_us)
        decay = math.exp(-elapsed / flash.decay_us)
        if flash.amp * decay < _ENVELOPE_CUTOFF:
            flash.amp = 0.0
            return 0.0, 0.0
        return flash.amp * decay, flash.white * decay

    def _sample(self, position: float) -> tuple[float, float, float]:
        """Sample the palette at a fractional, wrapping position."""
        palette = self._palette
        if len(palette) == 1:
            return palette[0]
        position %= len(palette)
        low = int(position)
        return _lerp(palette[low], palette[(low + 1) % len(palette)], position - low)


def _room_layout(
    channels: list[LightChannel],
) -> tuple[list[float], list[float], list[float]]:
    """
    Derive ring order, height and side for every light from its position.

    ``ring`` is the light's angle around the room centre as a 0-1 fraction, ``height``
    its normalised z (0 = lowest), and ``side`` 0 or 1 for left/right of centre. Lights
    without a usable layout fall back to their channel order.
    """
    n = len(channels)
    if n == 0:
        return [], [], []
    xs = [c.position[0] for c in channels]
    ys = [c.position[1] for c in channels]
    zs = [c.position[2] for c in channels]
    cx, cy = _mean(xs), _mean(ys)
    if max(xs) - min(xs) < 0.2 and max(ys) - min(ys) < 0.2:
        ring = [i / n for i in range(n)]
    else:
        angles = [math.atan2(y - cy, x - cx) for x, y in zip(xs, ys, strict=True)]
        order = sorted(range(n), key=lambda i: angles[i])
        ring = [0.0] * n
        for rank, idx in enumerate(order):
            ring[idx] = rank / n
    z_lo, z_hi = min(zs), max(zs)
    height = [(z - z_lo) / (z_hi - z_lo) for z in zs] if z_hi - z_lo >= 0.2 else [0.0 for _ in zs]
    if max(xs) - min(xs) >= 0.2:
        side = [1.0 if x > cx else 0.0 for x in xs]
    else:
        side = [float(i % 2) for i in range(n)]
    return ring, height, side


def _ema(value: float, target: float, dt: float, tau: float) -> float:
    """Step an exponential moving average by ``dt`` seconds with time constant ``tau``."""
    return value + (target - value) * (1.0 - math.exp(-dt / tau))


def _lift(raw: int) -> float:
    """Normalise a uint16 spectrum bin, gate the noise floor and lift it perceptually."""
    norm = max(0, min(65535, raw)) / 65535.0
    return 0.0 if norm < _NOISE_GATE else norm**_SPECTRUM_GAMMA


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _lerp(
    a: tuple[float, float, float], b: tuple[float, float, float], t: float
) -> tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _to_command(
    channel_id: int, color: tuple[float, float, float], scale: float
) -> LightColorCommand:
    return LightColorCommand(
        channel_id=channel_id,
        red=int(max(0.0, min(1.0, color[0] * scale)) * 65535),
        green=int(max(0.0, min(1.0, color[1] * scale)) * 65535),
        blue=int(max(0.0, min(1.0, color[2] * scale)) * 65535),
    )
