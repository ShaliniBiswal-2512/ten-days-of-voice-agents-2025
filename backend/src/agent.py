# src/agent.py -- Day 10: Voice Improv Battle Agent

import os
import json
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, deepgram, noise_cancellation, openai as lk_openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

logger = logging.getLogger("day10_improv")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ------------------------------
# USER SCENE STATE
# ------------------------------

@dataclass
class Userdata:
    scene_id: Optional[str] = None
    scene_title: Optional[str] = None
    scene_style: Optional[str] = None          # e.g. "comedy", "thriller"
    location: Optional[str] = None             # e.g. "Mumbai local train"
    characters: List[str] = field(default_factory=list)  # short names like "you", "me", "boss"
    beat_history: List[str] = field(default_factory=list)  # short notes of key moments
    scene_active: bool = False
    turns_in_scene: int = 0
    greeted: bool = False                      # LLM should treat this as "you have already greeted"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ------------------------------
# Pydantic models for tools
# ------------------------------

class StartSceneInput(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="Short title for the scene, e.g. 'Late Night Study Session in Hostel'",
    )
    style: Optional[str] = Field(
        default=None,
        description="Mood or genre, e.g. 'comedy', 'drama', 'thriller', 'wholesome'",
    )
    location: Optional[str] = Field(
        default=None,
        description="Where the scene is happening, e.g. 'Mumbai cafe', 'college classroom'",
    )
    characters: Optional[str] = Field(
        default=None,
        description="Short description of characters, e.g. 'you as a student, me as strict teacher'",
    )


class RecordBeatInput(BaseModel):
    beat_text: str = Field(
        description="Short note of what just happened in the story, from the agent's point of view.",
    )


# ------------------------------
# TOOLS
# ------------------------------

@function_tool
async def start_scene(
    ctx: RunContext[Userdata],
    data: StartSceneInput,
) -> str:
    """
    Start a new improv scene with title, style, location and characters.
    Used by the LLM when the user says 'let's start a new scene' or similar.
    """
    ud = ctx.userdata
    ud.scene_id = f"scene-{uuid.uuid4().hex[:6]}"
    ud.scene_title = data.title or "Untitled Improv Scene"
    ud.scene_style = data.style or "light comedy"
    ud.location = data.location or "a casual everyday place"
    ud.characters = []

    if data.characters:
        # Very simple split on commas
        chars = [c.strip() for c in data.characters.split(",") if c.strip()]
        ud.characters.extend(chars)

    ud.beat_history = []
    ud.scene_active = True
    ud.turns_in_scene = 0

    description_parts = [
        f"Starting a new improv scene: {ud.scene_title}.",
        f"Style: {ud.scene_style}.",
        f"Location: {ud.location}.",
    ]
    if ud.characters:
        description_parts.append(f"Characters: {', '.join(ud.characters)}.")

    description_parts.append("You can begin the scene by saying your first line.")
    return " ".join(description_parts)


@function_tool
async def record_beat(
    ctx: RunContext[Userdata],
    data: RecordBeatInput,
) -> str:
    """
    Store a short 'beat' describing what just happened in the story.
    The LLM should call this after important turns to maintain continuity.
    """
    ud = ctx.userdata
    if not ud.scene_active:
        return "No active scene to record this beat in."

    text = data.beat_text.strip()
    if not text:
        return "Beat was empty, nothing stored."

    ud.beat_history.append(text)
    # keep only last 15 beats
    if len(ud.beat_history) > 15:
        ud.beat_history = ud.beat_history[-15:]

    ud.turns_in_scene += 1
    return f"Stored beat #{len(ud.beat_history)} for scene {ud.scene_id}."


@function_tool
async def get_scene_summary(ctx: RunContext[Userdata]) -> str:
    """
    Return a short recap of the current scene from stored beats.
    The assistant can use this for 'Previously on...' style callbacks.
    """
    ud = ctx.userdata
    if not ud.scene_active or not ud.beat_history:
        return "There is no active scene or nothing important has happened yet."

    lines = [f"Scene recap for '{ud.scene_title}' ({ud.scene_style}) at {ud.location}:"]
    for idx, b in enumerate(ud.beat_history[-5:], start=1):
        lines.append(f"{idx}. {b}")
    return "\n".join(lines)


@function_tool
async def end_scene(ctx: RunContext[Userdata]) -> str:
    """
    End the current improv scene. Returns a brief closing summary.
    """
    ud = ctx.userdata
    if not ud.scene_active:
        return "There is no active scene to end."

    title = ud.scene_title or "your improv scene"
    beats = ud.beat_history[-5:] if ud.beat_history else []

    ud.scene_active = False

    lines = [f"Ending {title}."]
    if beats:
        lines.append("Quick recap of how it went:")
        for idx, b in enumerate(beats, start=1):
            lines.append(f"{idx}. {b}")
    else:
        lines.append("We didn't record any key beats, but thank you for playing!")

    lines.append("You can start a fresh scene any time.")
    return "\n".join(lines)


# ------------------------------
# AGENT
# ------------------------------

class ImprovAgent(Agent):
    def __init__(self):
        instructions = """
You are 'Riya', a playful Indian improv partner for a Voice Improv Battle.

HIGH-LEVEL BEHAVIOUR:
- You act with the user like two actors doing improv together.
- You always respond in simple, clear English that sounds natural when spoken aloud.
- You are warm, encouraging, and a little dramatic or funny.

SPECIAL WELCOME BEHAVIOUR:
- When the user greets you at the start (for example: "hi", "hello", "hey", "namaste",
  "good morning", "good evening"), you MUST give a warm welcome intro.
- Your welcome intro should be similar to:
  "Hey! Riya here — your improv buddy for today’s Voice Battle!
   Ready to create something dramatic, funny, or totally filmy with me?
   What kind of scene shall we begin — comedy, thriller, romantic, or something else?"
- Do this only the FIRST time the user greets you in a session.
  After that, give normal answers and do not repeat the full welcome speech.

SCENE MANAGEMENT:
- When there is no active scene yet:
  - First, handle any greeting with the special welcome intro.
  - Then, ask what kind of scene the user wants (comedy, drama, thriller, romantic, etc.).
  - Once they answer, use the `start_scene` tool to properly set up the scene.
- During a scene:
  - Stay in character.
  - Build on what the user says ("yes, and" improv style).
  - Speak in short, vivid sentences and always end with a question like
    "What do you say next?" or "How do you respond now?"
  - After important moments, call `record_beat` using a short description
    like "The train starts moving", "You confess your secret", etc.
- If the user asks "what happened so far" or "recap", call `get_scene_summary`
  and then continue the scene.
- When the user says they want to stop or end, call `end_scene`, read the recap,
  and close the scene politely. Then you can offer to start a new one.

SAFETY:
- Keep everything safe, light, and non-violent.
- No offensive or harmful content.
- This is only a fun demo, not real advice.

REMEMBER:
- Always end each response with a question so the user can continue.
"""
        super().__init__(
            instructions=instructions,
            tools=[start_scene, record_beat, get_scene_summary, end_scene],
        )


# ------------------------------
# ENTRYPOINT + PREWARM
# ------------------------------

def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("VAD prewarmed successfully.")
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")


async def entrypoint(ctx: JobContext):
    userdata = Userdata()

    logger.info("Starting Day 10 Improv Agent session...")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=lk_openai.LLM(model="gpt-4o-mini"),
        tts=murf.TTS(
            voice="en-US-matthew",   # keep a known working voice; swap later if needed
            style="Conversation",
            text_pacing=True,
        ),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=ImprovAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    logger.info("Improv agent is ready and listening.")
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
