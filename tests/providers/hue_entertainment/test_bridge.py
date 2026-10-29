"""Tests for the Hue bridge's effects engine selection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from aiosendspin.models.visualizer import BeatTiming
from hue_entertainment import LightChannel

from music_assistant.providers.hue_entertainment.analyzer import HueAudioAnalyzer
from music_assistant.providers.hue_entertainment.bridge import (
    HueEntertainmentBridge,
    _create_engine,
)
from music_assistant.providers.hue_entertainment.party_engine import PartyEngine
from music_assistant.providers.hue_entertainment.recorder import RECORD_DIR_ENV

_NOW_US = 10_000_000


@pytest.fixture
def area() -> MagicMock:
    """Entertainment area with two lights."""
    area = MagicMock()
    area.id = "area-1"
    area.name = "Living room"
    area.channels = [
        LightChannel(channel_id=i, service_id=f"svc_{i}", name=f"Light {i}") for i in range(2)
    ]
    return area


@pytest.fixture
def bridge(area: MagicMock, monkeypatch: pytest.MonkeyPatch) -> HueEntertainmentBridge:
    """Bridge running the classic analyzer, with a fixed server clock."""
    monkeypatch.delenv(RECORD_DIR_ENV, raising=False)
    sendspin_server = MagicMock()
    sendspin_server.clock.now_us.return_value = _NOW_US
    provider = MagicMock()
    provider.config.get_value.return_value = None
    bridge = HueEntertainmentBridge(provider, area, sendspin_server)
    bridge._analyzer = _create_engine(area, "smooth", 100)
    return bridge


@pytest.mark.parametrize(
    ("mode", "engine_cls"),
    [("party", PartyEngine), ("smooth", HueAudioAnalyzer), ("energetic", HueAudioAnalyzer)],
)
def test_create_engine_by_mode(area: MagicMock, mode: str, engine_cls: type) -> None:
    """The party mode gets the party engine; every other mode the classic analyzer."""
    assert isinstance(_create_engine(area, mode, 80), engine_cls)


def test_switch_to_party_carries_palette_and_beats(bridge: HueEntertainmentBridge) -> None:
    """Switching mid-song to party starts the new engine with the album colors and beats."""
    bridge._palette = {"primary": (230, 30, 40)}
    bridge._on_beats([BeatTiming(timestamp_us=_NOW_US + 500_000, is_downbeat=True)])

    bridge.update_settings(color_mode="party", brightness=60, hue_latency_ms=50)

    engine = bridge._analyzer
    assert isinstance(engine, PartyEngine)
    assert engine._server_palette == {"primary": (230, 30, 40)}
    assert [b.timestamp_us for b in engine._beats] == [_NOW_US + 500_000]
    assert engine._brightness == pytest.approx(0.6)
    assert bridge._hue_latency_us == 50_000


def test_switch_within_classic_keeps_engine(bridge: HueEntertainmentBridge) -> None:
    """Changing between classic modes updates the analyzer in place."""
    engine = bridge._analyzer
    bridge.update_settings(color_mode="energetic", brightness=40)
    assert bridge._analyzer is engine
    assert isinstance(engine, HueAudioAnalyzer)
    assert engine._color_mode == "energetic"


def test_switch_back_to_classic(bridge: HueEntertainmentBridge) -> None:
    """Leaving party mode swaps back to the classic analyzer."""
    bridge.update_settings(color_mode="party", brightness=100)
    bridge.update_settings(color_mode="ambient", brightness=100)
    assert isinstance(bridge._analyzer, HueAudioAnalyzer)


def test_beat_schedule_tracking(bridge: HueEntertainmentBridge) -> None:
    """Past beats are pruned on each push and a track change clears the schedule."""
    bridge._on_beats([BeatTiming(timestamp_us=_NOW_US - 1_000_000, is_downbeat=False)])
    bridge._on_beats([BeatTiming(timestamp_us=_NOW_US + 1_000_000, is_downbeat=False)])
    assert [b.timestamp_us for b in bridge._beat_schedule] == [_NOW_US + 1_000_000]
    bridge._on_beats_clear()
    assert bridge._beat_schedule == []
