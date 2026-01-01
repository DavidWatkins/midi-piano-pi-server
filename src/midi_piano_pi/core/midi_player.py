"""MIDI File Player for network MIDI interface."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import mido
from mido import MetaMessage

from .midi_controller import MIDIController, get_midi_controller

logger = logging.getLogger(__name__)

# GM Piano family program numbers (0-7)
# 0: Acoustic Grand Piano, 1: Bright Acoustic Piano, 2: Electric Grand Piano
# 3: Honky-tonk Piano, 4: Electric Piano 1, 5: Electric Piano 2
# 6: Harpsichord, 7: Clavinet
PIANO_PROGRAMS = {0, 1, 2, 3, 4, 5, 6, 7}

# Drum channel (excluded from piano playback)
DRUM_CHANNEL = 9


class PlaybackState(str, Enum):
    """Playback state enumeration."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class PlaybackStatus:
    """Current playback status."""
    state: PlaybackState = PlaybackState.STOPPED
    file_name: Optional[str] = None
    position_ms: int = 0
    duration_ms: int = 0
    tempo_percent: int = 100
    current_tick: int = 0
    total_ticks: int = 0
    play_all_channels: bool = False
    piano_channels: list = field(default_factory=list)


@dataclass
class MIDIFileInfo:
    """Information about a MIDI file."""
    path: Path
    name: str
    duration_ms: int
    total_ticks: int
    track_count: int
    has_lyrics: bool = False
    piano_channels: list = field(default_factory=list)
    all_channels: list = field(default_factory=list)


class MIDIPlayer:
    """
    Asynchronous MIDI file player.

    Plays MIDI and KAR files through the MIDI interface.
    Automatically detects piano channels and remaps them to channel 0.
    """

    def __init__(
        self,
        midi_controller: Optional[MIDIController] = None,
        on_status_change: Optional[Callable[[PlaybackStatus], None]] = None,
    ):
        self._midi = midi_controller or get_midi_controller()
        self._on_status_change = on_status_change

        self._status = PlaybackStatus()
        self._midi_file: Optional[mido.MidiFile] = None
        self._file_info: Optional[MIDIFileInfo] = None

        self._playback_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default

        self._tempo_factor = 1.0
        self._start_tick = 0

        # Channel filtering
        self._piano_channels: set[int] = set()
        self._channel_programs: dict[int, int] = {}  # channel -> program number
        self._play_all_channels = False  # Toggle for non-piano instruments

    @property
    def status(self) -> PlaybackStatus:
        """Get current playback status."""
        return self._status

    @property
    def file_info(self) -> Optional[MIDIFileInfo]:
        """Get loaded file info."""
        return self._file_info

    async def load_async(self, file_path: Path) -> MIDIFileInfo:
        """
        Load a MIDI or KAR file asynchronously, stopping current playback first.

        Args:
            file_path: Path to the MIDI file

        Returns:
            MIDIFileInfo with file details
        """
        # Stop any current playback synchronously
        if self._status.state != PlaybackState.STOPPED:
            await self.stop()

        return self._load_file(file_path)

    def load(self, file_path: Path) -> MIDIFileInfo:
        """
        Load a MIDI or KAR file (sync version).

        Args:
            file_path: Path to the MIDI file

        Returns:
            MIDIFileInfo with file details
        """
        # Stop any current playback
        if self._status.state != PlaybackState.STOPPED:
            # Force stop - set events and clear task
            self._stop_event.set()
            self._pause_event.set()
            if self._playback_task:
                self._playback_task.cancel()
                self._playback_task = None
            self._midi.all_notes_off()
            self._status.state = PlaybackState.STOPPED
            self._status.position_ms = 0
            self._status.current_tick = 0

        return self._load_file(file_path)

    def _load_file(self, file_path: Path) -> MIDIFileInfo:
        """
        Internal method to load a MIDI file.

        Args:
            file_path: Path to the MIDI file

        Returns:
            MIDIFileInfo with file details
        """

        logger.info("Loading MIDI file: %s", file_path)

        try:
            self._midi_file = mido.MidiFile(file_path)
        except Exception as e:
            logger.error("Failed to load MIDI file: %s", e)
            raise ValueError(f"Invalid MIDI file: {e}")

        # Calculate duration
        duration_ms = int(self._midi_file.length * 1000)

        # Count total ticks
        total_ticks = 0
        for track in self._midi_file.tracks:
            track_ticks = sum(msg.time for msg in track)
            total_ticks = max(total_ticks, track_ticks)

        # Check for lyrics (KAR files)
        has_lyrics = any(
            msg.type == 'lyrics' or (msg.type == 'text' and hasattr(msg, 'text'))
            for track in self._midi_file.tracks
            for msg in track
        )

        # Analyze channels and detect piano tracks
        self._channel_programs.clear()
        self._piano_channels.clear()
        all_channels: set[int] = set()

        for track in self._midi_file.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    channel = msg.channel
                    program = msg.program
                    self._channel_programs[channel] = program
                    all_channels.add(channel)

                    # Check if this is a piano program (0-7) and not drums
                    if program in PIANO_PROGRAMS and channel != DRUM_CHANNEL:
                        self._piano_channels.add(channel)
                        logger.debug("Piano detected: channel %d, program %d", channel, program)

                elif msg.type in ('note_on', 'note_off'):
                    all_channels.add(msg.channel)

        # If no piano channels detected, use all non-drum channels
        # (allows files without explicit program changes to still play)
        if not self._piano_channels:
            self._piano_channels = {ch for ch in all_channels if ch != DRUM_CHANNEL}
            logger.info("No piano program detected, using all non-drum channels: %s",
                       sorted(self._piano_channels))

        piano_channels_list = sorted(self._piano_channels)
        all_channels_list = sorted(all_channels)

        logger.info("Piano channels: %s", piano_channels_list)
        logger.info("All channels: %s", all_channels_list)
        logger.info("Channel programs: %s", self._channel_programs)

        self._file_info = MIDIFileInfo(
            path=file_path,
            name=file_path.name,
            duration_ms=duration_ms,
            total_ticks=total_ticks,
            track_count=len(self._midi_file.tracks),
            has_lyrics=has_lyrics,
            piano_channels=piano_channels_list,
            all_channels=all_channels_list,
        )

        # Update status
        self._status = PlaybackStatus(
            state=PlaybackState.STOPPED,
            file_name=file_path.name,
            duration_ms=duration_ms,
            total_ticks=total_ticks,
            play_all_channels=self._play_all_channels,
            piano_channels=piano_channels_list,
        )
        self._notify_status_change()

        logger.info(
            "Loaded: %s (duration: %dms, tracks: %d, lyrics: %s, piano_ch: %s)",
            file_path.name, duration_ms, len(self._midi_file.tracks),
            has_lyrics, piano_channels_list
        )

        return self._file_info

    def set_play_all_channels(self, enabled: bool) -> None:
        """
        Toggle playing all channels vs piano-only.

        Args:
            enabled: True to play all channels, False for piano only
        """
        self._play_all_channels = enabled
        self._status.play_all_channels = enabled
        logger.info("Play all channels: %s", enabled)
        self._notify_status_change()

    def _should_send_to_interface(self, channel: int) -> bool:
        """
        Check if a message on this channel should be sent to the MIDI interface.

        Args:
            channel: MIDI channel (0-15)

        Returns:
            True if the message should be sent
        """
        # Never send drums to piano
        if channel == DRUM_CHANNEL:
            return False

        # If play all channels is enabled, send everything except drums
        if self._play_all_channels:
            return True

        # Otherwise only send piano channels
        return channel in self._piano_channels

    async def play(self, from_tick: int = 0) -> None:
        """
        Start or resume playback.

        Args:
            from_tick: Starting tick position (0 = beginning)
        """
        if self._midi_file is None:
            raise ValueError("No MIDI file loaded")

        if self._status.state == PlaybackState.PAUSED:
            # Resume from pause
            self._pause_event.set()
            self._status.state = PlaybackState.PLAYING
            self._notify_status_change()
            return

        if self._status.state == PlaybackState.PLAYING:
            return  # Already playing

        # Start new playback
        self._start_tick = from_tick
        self._stop_event.clear()
        self._pause_event.set()

        self._status.state = PlaybackState.PLAYING
        self._status.current_tick = from_tick
        self._notify_status_change()

        # Start playback task
        self._playback_task = asyncio.create_task(self._playback_loop())

    async def pause(self) -> None:
        """Pause playback."""
        if self._status.state != PlaybackState.PLAYING:
            return

        self._pause_event.clear()
        self._status.state = PlaybackState.PAUSED
        self._notify_status_change()

    async def stop(self) -> None:
        """Stop playback and reset position."""
        if self._status.state == PlaybackState.STOPPED:
            return

        # Signal stop
        self._stop_event.set()
        self._pause_event.set()  # Unpause to allow loop to exit

        # Wait for playback task to finish
        if self._playback_task:
            try:
                await asyncio.wait_for(self._playback_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._playback_task.cancel()
            self._playback_task = None

        # Send all notes off
        self._midi.all_notes_off()

        # Reset status
        self._status.state = PlaybackState.STOPPED
        self._status.position_ms = 0
        self._status.current_tick = 0
        self._notify_status_change()

    async def seek(self, position_ms: int) -> None:
        """
        Seek to a position in the file.

        Args:
            position_ms: Position in milliseconds
        """
        if self._midi_file is None:
            return

        was_playing = self._status.state == PlaybackState.PLAYING

        # Stop current playback
        await self.stop()

        # Calculate tick position from ms
        # This is approximate - proper implementation would iterate through tempo changes
        if self._status.duration_ms > 0:
            tick_ratio = position_ms / self._status.duration_ms
            target_tick = int(tick_ratio * self._status.total_ticks)
        else:
            target_tick = 0

        self._status.position_ms = position_ms
        self._status.current_tick = target_tick

        if was_playing:
            await self.play(from_tick=target_tick)

    def set_tempo(self, percent: int) -> None:
        """
        Set playback tempo as percentage of original.

        Args:
            percent: Tempo percentage (50 = half speed, 200 = double speed)
        """
        self._tempo_factor = percent / 100.0
        self._status.tempo_percent = percent
        self._notify_status_change()

    async def _playback_loop(self) -> None:
        """Main playback loop."""
        if self._midi_file is None:
            return

        logger.info("Starting playback")

        try:
            start_time = time.time()
            current_tick = self._start_tick

            # Iterate directly over the MIDI file (not using play() which has blocking sleep)
            # This yields messages with msg.time as delta time in seconds
            for msg in self._midi_file:
                # Check for stop
                if self._stop_event.is_set():
                    break

                # Handle pause
                await self._pause_event.wait()

                if self._stop_event.is_set():
                    break

                # Apply tempo adjustment to timing using non-blocking async sleep
                # IMPORTANT: Do this BEFORE checking message type, because MetaMessages
                # (like time_signature, set_tempo) can have timing too, and we need to
                # preserve that delay even when skipping the message itself
                if msg.time > 0:
                    adjusted_time = msg.time / self._tempo_factor
                    await asyncio.sleep(adjusted_time)

                # Skip meta messages (tempo changes are already handled by mido's iteration)
                if isinstance(msg, MetaMessage):
                    continue

                # Filter and remap MIDI messages
                # All messages are sent on channel 0 for piano compatibility
                if msg.type == 'note_on':
                    if self._should_send_to_interface(msg.channel):
                        # Remap to channel 0 for piano
                        self._midi.note_on(msg.note, msg.velocity, channel=0)
                elif msg.type == 'note_off':
                    if self._should_send_to_interface(msg.channel):
                        self._midi.note_off(msg.note, channel=0)
                elif msg.type == 'control_change':
                    if self._should_send_to_interface(msg.channel):
                        self._midi.control_change(msg.control, msg.value, channel=0)
                elif msg.type == 'pitchwheel':
                    if self._should_send_to_interface(msg.channel):
                        self._midi.pitch_bend(msg.pitch + 8192, channel=0)
                # Skip program_change - piano is always piano

                # Update position
                elapsed = time.time() - start_time
                self._status.position_ms = int(elapsed * 1000 * self._tempo_factor)
                self._status.current_tick = current_tick

        except asyncio.CancelledError:
            logger.info("Playback cancelled")
        except Exception as e:
            logger.exception("Playback error: %s", e)
        finally:
            # Playback finished
            if not self._stop_event.is_set():
                # Natural end of file
                self._status.state = PlaybackState.STOPPED
                self._status.position_ms = self._status.duration_ms
                self._notify_status_change()
                logger.info("Playback finished")

            # Ensure all notes are off
            self._midi.all_notes_off()

    def _notify_status_change(self) -> None:
        """Notify listeners of status change."""
        if self._on_status_change:
            self._on_status_change(self._status)


# Global singleton instance
_player: Optional[MIDIPlayer] = None
_auto_queue_enabled: bool = True


def set_auto_queue(enabled: bool) -> None:
    """Enable or disable auto-queue advancement."""
    global _auto_queue_enabled
    _auto_queue_enabled = enabled


def is_auto_queue_enabled() -> bool:
    """Check if auto-queue is enabled."""
    return _auto_queue_enabled


def get_midi_player() -> MIDIPlayer:
    """Get the global MIDI player instance."""
    global _player
    if _player is None:
        _player = MIDIPlayer(on_status_change=_handle_playback_status_change)
    return _player


def _handle_playback_status_change(status: PlaybackStatus) -> None:
    """Handle playback status changes for auto-queue."""
    global _auto_queue_enabled

    # Check if song finished naturally (not manually stopped)
    # Natural finish: state is STOPPED and position equals duration
    if (status.state == PlaybackState.STOPPED and
        status.duration_ms > 0 and
        status.position_ms >= status.duration_ms - 100 and  # Allow small tolerance
        _auto_queue_enabled):

        logger.info("Auto-queue: Song finished, checking for next in queue")

        # Schedule playing next song (avoid blocking the callback)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_play_next_from_queue())
        except RuntimeError:
            # No running event loop - try to get the default loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(_play_next_from_queue())
                else:
                    logger.warning("Auto-queue: Event loop not running")
            except RuntimeError:
                logger.warning("Auto-queue: No event loop available")


async def _play_next_from_queue() -> None:
    """Play the next song from queue if available."""
    import json
    from pathlib import Path

    QUEUE_FILE = Path("/var/lib/midi-piano-pi/queue.json")

    # Small delay to ensure clean state
    await asyncio.sleep(0.5)

    # Load queue
    if not QUEUE_FILE.exists():
        return

    try:
        queue = json.loads(QUEUE_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return

    if not queue:
        return

    # Pop next item
    next_item = queue.pop(0)
    QUEUE_FILE.write_text(json.dumps(queue))

    # Find the file in catalog directories using id_to_path
    from .config import get_settings
    settings = get_settings()

    def id_to_path(file_id: str, base: Path) -> Path:
        """Convert a URL-safe ID back to a path."""
        rel_path = file_id.replace("__", "/")
        return base / rel_path

    file_path = None
    for dir_path in settings.catalog.directories:
        catalog_dir = Path(dir_path).expanduser()
        if not catalog_dir.exists():
            continue
        try:
            potential_path = id_to_path(next_item["id"], catalog_dir)
            if potential_path.exists() and potential_path.is_file():
                file_path = potential_path
                break
        except (ValueError, KeyError):
            continue

    if not file_path:
        logger.warning("Auto-queue: File not found: %s", next_item.get("name", next_item["id"]))
        return

    # Play the file
    player = get_midi_player()
    try:
        await player.load_async(file_path)
        await player.play()
        logger.info("Auto-queue: Now playing %s", next_item.get("name", file_path.name))
    except Exception as e:
        logger.error("Auto-queue: Failed to play %s: %s", next_item.get("name"), e)
