"""Playback control endpoints."""

import json
import random
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...core.config import get_settings
from ...core.midi_player import MIDIPlayer, PlaybackState, get_midi_player
from .catalog import id_to_path

router = APIRouter(prefix="/api/v1/playback", tags=["playback"])

# Server-side queue storage
QUEUE_FILE = Path("/var/lib/midi-piano-pi/queue.json")


def load_queue() -> list[dict]:
    """Load queue from file."""
    if QUEUE_FILE.exists():
        try:
            return json.loads(QUEUE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_queue(queue: list[dict]) -> None:
    """Save queue to file."""
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(queue))


class PlaybackStatusResponse(BaseModel):
    """Current playback status."""
    state: str
    file_name: Optional[str]
    position_ms: int
    duration_ms: int
    tempo_percent: int
    play_all_channels: bool
    piano_channels: list[int]


class LoadRequest(BaseModel):
    """Request to load a file for playback."""
    file_id: str


class SeekRequest(BaseModel):
    """Request to seek to a position."""
    position_ms: int = Field(..., ge=0)


class TempoRequest(BaseModel):
    """Request to change tempo."""
    percent: int = Field(..., ge=25, le=400)


class ChannelModeRequest(BaseModel):
    """Request to change channel playback mode."""
    play_all: bool = Field(..., description="True to play all channels, False for piano only")


def find_file_by_id(file_id: str) -> Path:
    """Find a file by its ID in the upload directory."""
    settings = get_settings()
    upload_dir = Path(settings.uploads.directory)

    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Upload directory not found")

    for file_path in upload_dir.iterdir():
        if file_path.stem == file_id:
            return file_path

    raise HTTPException(status_code=404, detail=f"File not found: {file_id}")


@router.get("", response_model=PlaybackStatusResponse)
async def get_playback_status(player: MIDIPlayer = Depends(get_midi_player)):
    """Get current playback status."""
    status = player.status
    return PlaybackStatusResponse(
        state=status.state.value,
        file_name=status.file_name,
        position_ms=status.position_ms,
        duration_ms=status.duration_ms,
        tempo_percent=status.tempo_percent,
        play_all_channels=status.play_all_channels,
        piano_channels=status.piano_channels,
    )


@router.post("/load")
async def load_file(request: LoadRequest, player: MIDIPlayer = Depends(get_midi_player)):
    """Load a MIDI file for playback."""
    file_path = find_file_by_id(request.file_id)

    try:
        file_info = player.load(file_path)
        return {
            "success": True,
            "file": {
                "name": file_info.name,
                "duration_ms": file_info.duration_ms,
                "track_count": file_info.track_count,
                "has_lyrics": file_info.has_lyrics,
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/play")
async def play(player: MIDIPlayer = Depends(get_midi_player)):
    """Start or resume playback."""
    if player.file_info is None:
        raise HTTPException(status_code=400, detail="No file loaded")

    await player.play()
    return {"success": True, "state": player.status.state.value}


@router.post("/pause")
async def pause(player: MIDIPlayer = Depends(get_midi_player)):
    """Pause playback."""
    await player.pause()
    return {"success": True, "state": player.status.state.value}


@router.post("/stop")
async def stop(player: MIDIPlayer = Depends(get_midi_player)):
    """Stop playback and reset to beginning."""
    await player.stop()
    return {"success": True, "state": player.status.state.value}


@router.post("/seek")
async def seek(request: SeekRequest, player: MIDIPlayer = Depends(get_midi_player)):
    """Seek to a position in the file."""
    if player.file_info is None:
        raise HTTPException(status_code=400, detail="No file loaded")

    await player.seek(request.position_ms)
    return {
        "success": True,
        "position_ms": player.status.position_ms,
    }


@router.put("/tempo")
async def set_tempo(request: TempoRequest, player: MIDIPlayer = Depends(get_midi_player)):
    """Set playback tempo as percentage of original."""
    player.set_tempo(request.percent)
    return {
        "success": True,
        "tempo_percent": player.status.tempo_percent,
    }


@router.put("/channels")
async def set_channel_mode(
    request: ChannelModeRequest, player: MIDIPlayer = Depends(get_midi_player)
):
    """
    Set channel playback mode.

    When play_all is False (default), only piano channels (GM programs 0-7) are sent
    to the MIDI interface. When True, all channels except drums are played.
    """
    player.set_play_all_channels(request.play_all)
    return {
        "success": True,
        "play_all_channels": player.status.play_all_channels,
        "piano_channels": player.status.piano_channels,
    }


# ============================================
# Queue Endpoints
# ============================================

class QueueItem(BaseModel):
    """A queue item."""
    id: str
    name: str


class AddToQueueRequest(BaseModel):
    """Request to add item to queue."""
    id: str
    name: str


@router.get("/queue")
async def get_queue():
    """Get the current queue."""
    queue = load_queue()
    return {"queue": queue}


@router.post("/queue")
async def add_to_queue(request: AddToQueueRequest):
    """Add an item to the queue."""
    queue = load_queue()

    # Avoid duplicates
    if any(item["id"] == request.id for item in queue):
        return {"success": False, "message": "Already in queue", "queue": queue}

    queue.append({"id": request.id, "name": request.name})
    save_queue(queue)
    return {"success": True, "queue": queue}


@router.delete("/queue/{index}")
async def remove_from_queue(index: int):
    """Remove an item from the queue by index."""
    queue = load_queue()

    if index < 0 or index >= len(queue):
        raise HTTPException(status_code=404, detail="Invalid queue index")

    removed = queue.pop(index)
    save_queue(queue)
    return {"success": True, "removed": removed, "queue": queue}


@router.post("/queue/shuffle")
async def shuffle_queue():
    """Shuffle the queue."""
    queue = load_queue()
    random.shuffle(queue)
    save_queue(queue)
    return {"success": True, "queue": queue}


@router.delete("/queue")
async def clear_queue():
    """Clear the entire queue."""
    save_queue([])
    return {"success": True, "queue": []}


@router.post("/queue/next")
async def play_next(player: MIDIPlayer = Depends(get_midi_player)):
    """Pop the next item from queue and play it."""
    queue = load_queue()

    if not queue:
        return {"success": False, "message": "Queue is empty"}

    next_item = queue.pop(0)
    save_queue(queue)

    # Find and play the file using id_to_path
    settings = get_settings()

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
        return {"success": False, "message": f"File not found: {next_item['name']}", "queue": queue}

    try:
        await player.load_async(file_path)
        await player.play()
        return {
            "success": True,
            "playing": next_item,
            "queue": queue,
        }
    except Exception as e:
        return {"success": False, "message": str(e), "queue": queue}
