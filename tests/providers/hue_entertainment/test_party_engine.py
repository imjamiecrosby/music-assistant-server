"""Tests for PartyEngine (section-aware dance party effects)."""

from __future__ import annotations

import math

import pytest
from aiosendspin.models.visualizer import BeatTiming
from hue_entertainment import LightChannel

from music_assistant.providers.hue_entertainment.party_engine import (
    _FLASH_LIMIT,
    _HAT_BAND,
    _KICK_BAND,
    _PALETTE_PEAK,
    _PARTY_FALLBACK,
    _SNARE_BAND,
    SCENE_BUILD,
    SCENE_CHILL,
    SCENE_PEAK,
    PartyEngine,
    _light_band,
    _party_palette,
    _Reactor,
    _room_layout,
)

_FRAME_US = 1_000_000 // 30
_BEAT_US = 500_000


def _make_channels(
    positions: list[tuple[float, float, float]] | None = None, count: int = 4
) -> list[LightChannel]:
    """Create test light channels, optionally at given room positions."""
    if positions is None:
        positions = [(0.0, 0.0, 0.0)] * count
    return [
        LightChannel(channel_id=i, service_id=f"svc_{i}", name=f"Light {i}", position=pos)
        for i, pos in enumerate(positions)
    ]


def _feed(
    engine: PartyEngine,
    start_us: int,
    seconds: float,
    *,
    level: float,
    kick: float = 0.0,
    ramp_to: float | None = None,
) -> tuple[int, set[str]]:
    """
    Feed synthetic spectrum frames and render each one.

    :param engine: Engine under test.
    :param start_us: Playback time of the first frame.
    :param seconds: Duration to feed.
    :param level: Base loudness (0-1) across the bins.
    :param kick: Height of a kick on every beat in the bass bins.
    :param ramp_to: Loudness to reach linearly by the end, for builds.
    """
    scenes: set[str] = set()
    frames = int(seconds * 1_000_000 / _FRAME_US)
    ts = start_us
    for frame in range(frames):
        progress = frame / frames
        base = level if ramp_to is None else level + (ramp_to - level) * progress
        since_beat = (ts % _BEAT_US) / _BEAT_US
        kick_env = kick * math.exp(-since_beat * 9)
        bins = [int(min(1.0, base + kick_env) * 65535)] * 2 + [int(base * 65535)] * 15
        engine.apply_spectrum(bins, ts)
        engine.render(ts)
        scenes.add(str(engine.debug_state()["scene"]))
        ts += _FRAME_US
    return ts, scenes


def _spy_drops(engine: PartyEngine, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the time of every drop the engine fires, keeping its normal behaviour."""
    drops: list[int] = []
    original = engine._drop

    def _spy(now_us: int) -> None:
        drops.append(now_us)
        original(now_us)

    monkeypatch.setattr(engine, "_drop", _spy)
    return drops


class TestRendering:
    """Tests for the render output shape and global settings."""

    def test_one_command_per_channel(self) -> None:
        """Every channel gets exactly one command."""
        engine = PartyEngine(_make_channels(count=5))
        commands = engine.render(1_000_000)
        assert [c.channel_id for c in commands] == [0, 1, 2, 3, 4]

    def test_no_channels(self) -> None:
        """An area without channels renders nothing."""
        assert PartyEngine([]).render(1_000_000) == []

    def test_zero_brightness_is_dark(self) -> None:
        """Brightness 0 turns every light off regardless of the music."""
        engine = PartyEngine(_make_channels(count=3), brightness=0)
        _feed(engine, 1_000_000, 2.0, level=0.8, kick=0.5)
        for cmd in engine.render(4_000_000):
            assert (cmd.red, cmd.green, cmd.blue) == (0, 0, 0)

    def test_update_settings_clamps_brightness(self) -> None:
        """Live brightness updates are clamped to 0-100."""
        engine = PartyEngine(_make_channels(count=1))
        engine.update_settings(brightness=150)
        assert engine._brightness == 1.0
        engine.update_settings(brightness=40)
        assert engine._brightness == pytest.approx(0.4)

    def test_palette_update_changes_colors(self) -> None:
        """A color@v1 update replaces the neutral default palette."""
        engine = PartyEngine(_make_channels(count=1))
        neutral = list(engine._palette)
        engine.apply_color_palette({"primary": (230, 30, 40), "accent": (30, 60, 230)})
        assert engine._palette != neutral


class TestScenes:
    """Tests for following the song's structure."""

    def test_song_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Quiet intro stays chill, a rising build is detected, and the drop hits peak."""
        engine = PartyEngine(_make_channels(count=4))
        drops = _spy_drops(engine, monkeypatch)

        ts, intro = _feed(engine, 1_000_000, 10.0, level=0.05)
        assert intro == {SCENE_CHILL}

        ts, build = _feed(engine, ts, 12.0, level=0.08, ramp_to=0.3, kick=0.1)
        assert SCENE_BUILD in build

        drop_start = ts
        ts, drop = _feed(engine, ts, 15.0, level=0.35, kick=0.6)
        assert engine.debug_state()["scene"] == SCENE_PEAK
        assert SCENE_PEAK in drop
        assert len(drops) == 1
        assert drops[0] - drop_start < 3_000_000

        _feed(engine, ts, 10.0, level=0.05)
        assert engine.debug_state()["scene"] == SCENE_CHILL

    def test_steady_music_never_drops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Music that stays at one level never fires a drop."""
        engine = PartyEngine(_make_channels(count=2))
        drops = _spy_drops(engine, monkeypatch)
        _feed(engine, 1_000_000, 30.0, level=0.1)
        assert drops == []


class TestReactivity:
    """Tests for the continuous per-light spectrum reaction."""

    def test_lights_react_to_hits_in_chill(self) -> None:
        """A sudden bass hit lifts the bass light even in the calm scene."""
        engine = PartyEngine(
            _make_channels([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]),
        )
        ts, _ = _feed(engine, 1_000_000, 5.0, level=0.05)
        assert engine.debug_state()["scene"] == SCENE_CHILL
        before = engine.render(ts)[0]
        engine.apply_spectrum([int(0.6 * 65535)] * 2 + [int(0.05 * 65535)] * 15, ts)
        after = engine.render(ts)[0]
        assert max(after.red, after.green, after.blue) > max(before.red, before.green, before.blue)

    def test_bass_goes_to_lowest_light(self) -> None:
        """The lowest light gets the first (bass) band regardless of channel order."""
        engine = PartyEngine(
            _make_channels([(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]),
        )
        assert engine._band_slot == [2, 0, 1]

    def test_light_bands_skip_vocal_mids(self) -> None:
        """Two lights follow the kick and the hi-hats; a middle light the snare body."""
        assert _light_band(0, 2) == _KICK_BAND
        assert _light_band(1, 2) == _HAT_BAND
        assert _light_band(1, 3) == _SNARE_BAND
        for lo, hi in (_KICK_BAND, _SNARE_BAND, _HAT_BAND):
            assert hi <= 6 or lo >= 9

    def test_vocal_swell_does_not_drive_lights(self) -> None:
        """A rise in the vocal-range bins alone barely moves the lights."""
        engine = PartyEngine(_make_channels(count=2))
        ts, _ = _feed(engine, 1_000_000, 5.0, level=0.1)
        before = [max(c.red, c.green, c.blue) for c in engine.render(ts)]
        vocal = [int(0.1 * 65535)] * 4 + [int(0.9 * 65535)] * 5 + [int(0.1 * 65535)] * 8
        for _ in range(3):
            ts += _FRAME_US
            engine.apply_spectrum(vocal, ts)
        after = [max(c.red, c.green, c.blue) for c in engine.render(ts)]
        for b, a in zip(before, after, strict=True):
            assert a - b < 0.1 * 65535

    def test_reactor_auto_gain(self) -> None:
        """Quiet and loud material both reach a full reaction on their own hits."""
        for scale in (0.05, 0.5):
            reactor = _Reactor()
            reactor.update(scale * 0.2, 1 / 30, seed=True)
            value = 0.0
            for _ in range(60):
                for frame in range(15):
                    energy = scale * (1.0 if frame == 0 else 0.2)
                    value = max(value, reactor.update(energy, 1 / 30, seed=False))
            assert value > 0.9


class TestPalette:
    """Tests for the party palette."""

    def test_colors_are_never_dimmed(self) -> None:
        """Every palette colour runs at the same full peak."""
        palette = _party_palette(
            {"primary": (230, 30, 40), "accent": (220, 40, 50), "on_dark": (30, 60, 230)}
        )
        for color in palette:
            assert max(color) == pytest.approx(_PALETTE_PEAK)

    def test_grey_cover_gets_vivid_fallback(self) -> None:
        """A cover with no real colour uses the party fallback, not tinted whites."""
        assert _party_palette({"primary": (200, 200, 200)}) == list(_PARTY_FALLBACK)
        assert _party_palette({}) == list(_PARTY_FALLBACK)


class TestBeats:
    """Tests for beat and onset handling."""

    def test_late_beats_are_skipped(self) -> None:
        """Beats already well in the past when first seen don't fire a backlog."""
        engine = PartyEngine(_make_channels(count=2))
        engine.push_beats(
            [BeatTiming(timestamp_us=ts, is_downbeat=False) for ts in (100_000, 600_000)]
        )
        engine.render(2_000_000)
        assert engine._beat_count == 0
        assert not engine._beats

    def test_due_beats_fire_once(self) -> None:
        """A beat fires when the playhead reaches it and never again."""
        engine = PartyEngine(_make_channels(count=2))
        engine.push_beats([BeatTiming(timestamp_us=1_000_000, is_downbeat=True)])
        engine.render(999_000)
        assert engine._beat_count == 0
        engine.render(1_010_000)
        engine.render(1_040_000)
        assert engine._beat_count == 1

    def test_onsets_stand_in_for_missing_schedule(self) -> None:
        """Without a beat schedule, onset peaks drive the beat counter."""
        engine = PartyEngine(_make_channels(count=2))
        for i in range(4):
            ts = 1_000_000 + i * _BEAT_US
            engine.apply_peak(200, ts)
            engine.render(ts)
        assert engine._beat_count == 4

    def test_weak_onsets_ignored(self) -> None:
        """Low-confidence onsets are dropped on arrival."""
        engine = PartyEngine(_make_channels(count=1))
        engine.apply_peak(10, 1_000_000)
        assert not engine._pending_peaks

    def test_clear_beats_drops_queued_features(self) -> None:
        """A seek drops queued spectrum, peaks and beats."""
        engine = PartyEngine(_make_channels(count=1))
        engine.apply_spectrum([1000] * 17, 5_000_000)
        engine.apply_peak(200, 5_000_000)
        engine.push_beats([BeatTiming(timestamp_us=5_000_000, is_downbeat=True)])
        engine.clear_beats()
        assert not engine._pending_spectrum
        assert not engine._pending_peaks
        assert not engine._beats


class TestFlashLimit:
    """Tests for the photosensitivity flash limiter."""

    def test_flash_rate_capped_per_light(self) -> None:
        """No light starts more than the limit of flashes in one second."""
        engine = PartyEngine(_make_channels(count=1))
        for i in range(10):
            engine._trigger(0, 1_000_000 + i * 50_000, amp=1.0, decay_us=100_000, white=0.0)
        assert len(engine._flashes[0].onsets or []) == _FLASH_LIMIT

    def test_flash_allowed_after_window(self) -> None:
        """The limiter frees up once the oldest flash leaves the window."""
        engine = PartyEngine(_make_channels(count=1))
        for i in range(_FLASH_LIMIT):
            engine._trigger(0, 1_000_000 + i * 10_000, amp=1.0, decay_us=100_000, white=0.0)
        engine._trigger(0, 2_100_000, amp=0.5, decay_us=100_000, white=0.0)
        assert engine._flashes[0].start_us == 2_100_000


class TestRoomLayout:
    """Tests for deriving spatial roles from light positions."""

    def test_layout_from_positions(self) -> None:
        """Ring order follows angle, height follows z, side follows x."""
        ring, height, side = _room_layout(
            _make_channels([(-1.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.5)])
        )
        assert len(set(ring)) == 4
        assert height == [0.0, 1.0, 0.0, 0.5]
        assert side == [0.0, 0.0, 1.0, 0.0]

    def test_colocated_lights_fall_back_to_channel_order(self) -> None:
        """Lights at one point still get distinct ring slots and alternating sides."""
        ring, height, side = _room_layout(_make_channels(count=4))
        assert ring == [0.0, 0.25, 0.5, 0.75]
        assert height == [0.0] * 4
        assert side == [0.0, 1.0, 0.0, 1.0]
