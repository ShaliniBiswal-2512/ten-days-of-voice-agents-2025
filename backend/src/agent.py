import logging
import json
import random
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    metrics,
    tokenize,
    function_tool,
    cli,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("sunnyvale-puppy-agent")

load_dotenv(".env.local")


class GameMasterAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly and cheerful Game Master running a simple, light-hearted adventure set in the cozy town of Sunnyvale.

TONE RULES:
- Speak in easy, friendly, playful tone
- Keep descriptions simple and vivid, using easy English
- Avoid complex fantasy words or heavy lore
- Always end with: "What would you like to do next?"

STORY THEME: "The Lost Puppy of Sunnyvale"
- The player is a young helper in the peaceful town of Sunnyvale
- The town baker, Mrs. Willow, has lost her puppy named Milo
- The player must help find Milo by exploring different places in town

GAME MASTER GUIDELINES:
1. Begin the story in Sunnyvale Town Square in the morning
2. Introduce Mrs. Willow and her missing puppy Milo
3. Let the player explore Sunnyvale: the town park, the market, and the riverside
4. Include 2–3 small challenges (talking to townsfolk, following paw prints, simple clues)
5. Give the player choices that affect the progress of the search
6. End with finding Milo and returning him to Mrs. Willow
7. Always remember what the player did earlier and keep continuity

STORY BEATS:
- Mrs. Willow asks for help finding Milo
- First clue in or near the town park
- A clue like paw prints, a dropped ribbon, or barking sound near the riverside
- A helpful NPC (gardener, child, or shopkeeper) gives a hint
- Milo is found stuck or hiding somewhere safe but hard to spot
- Heartwarming ending with Mrs. Willow and Milo

You are the player's guide. Keep it simple, warm, and fun.
Always, always end your turn with: "What would you like to do next?"
"""
        )
        self.world_state = self._initialize_world_state()
        self.game_active = True

    def _initialize_world_state(self) -> Dict[str, Any]:
        """Initialize the game world state for the Sunnyvale puppy story."""
        return {
            "player": {
                "name": "Helper",
                "health": 100,
                "max_health": 100,
                "inventory": ["notebook", "pencil", "small backpack"],
                "gold": 0,
                "location": "sunnyvale_square",
                "reputation": 0,
            },
            "npcs": {
                "mrs_willow": {
                    "name": "Mrs. Willow",
                    "role": "Town Baker",
                    "location": "sunnyvale_square",
                    "attitude": "friendly",
                    "quest_given": False,
                },
                "gardener": {
                    "name": "Mr. Green",
                    "role": "Park Gardener",
                    "location": "town_park",
                    "attitude": "friendly",
                    "spoken_to": False,
                },
                "shopkeeper": {
                    "name": "Lena",
                    "role": "Market Shopkeeper",
                    "location": "market",
                    "attitude": "friendly",
                    "spoken_to": False,
                },
            },
            "locations": {
                "sunnyvale_square": {
                    "name": "Sunnyvale Town Square",
                    "description": "The center of Sunnyvale, with a small fountain, shops, and the bakery. Morning sun makes everything bright and warm.",
                    "visited": False,
                },
                "town_park": {
                    "name": "Sunnyvale Park",
                    "description": "A green park with soft grass, tall trees, and children playing nearby.",
                    "visited": False,
                },
                "riverside": {
                    "name": "Riverside Path",
                    "description": "A quiet path beside a gentle river, where birds sing and water flows softly.",
                    "visited": False,
                },
                "market": {
                    "name": "Sunnyvale Market",
                    "description": "A small market with fresh fruits, toys, and kind townsfolk.",
                    "visited": False,
                },
            },
            "quests": {
                "find_milo": {
                    "name": "Find Milo the Puppy",
                    "status": "active",  # active, completed, failed
                    "description": "Help Mrs. Willow find her lost puppy Milo somewhere in Sunnyvale.",
                }
            },
            "events": {
                "met_mrs_willow": False,
                "went_to_park": False,
                "saw_pawprints": False,
                "spoke_to_gardener": False,
                "spoke_to_shopkeeper": False,
                "went_to_riverside": False,
                "heard_barking": False,
                "found_milo": False,
            },
            "game_state": {
                "turn_count": 0,
                "current_chapter": 1,
            },
        }

    def _dice_roll(self, sides: int = 20, modifier: int = 0) -> int:
        """Make a dice roll for game mechanics."""
        return random.randint(1, sides) + modifier

    def _update_location(self, new_location: str):
        """Update player location and trigger location-based events."""
        old_location = self.world_state["player"]["location"]
        self.world_state["player"]["location"] = new_location

        # Mark location as visited
        if new_location in self.world_state["locations"]:
            self.world_state["locations"][new_location]["visited"] = True

        # Trigger simple story-related flags
        if new_location == "town_park":
            self.world_state["events"]["went_to_park"] = True
        elif new_location == "riverside":
            self.world_state["events"]["went_to_riverside"] = True

    @function_tool()
    async def roll_dice(self, context: RunContext, sides: int = 20, modifier: int = 0) -> str:
        """Roll dice for simple chance checks (e.g., noticing clues)."""
        roll = self._dice_roll(sides, modifier)
        return f"Dice roll: {roll} (d{sides} + {modifier})"

    @function_tool()
    async def check_inventory(self, context: RunContext) -> str:
        """Check the player's current inventory and status."""
        player = self.world_state["player"]
        inventory_list = ", ".join(player["inventory"]) if player["inventory"] else "Empty"

        return (
            f"Health: {player['health']}/{player['max_health']} | "
            f"Gold: {player['gold']} | Inventory: {inventory_list}"
        )

    @function_tool()
    async def add_to_inventory(self, context: RunContext, item: str) -> str:
        """Add an item to player's inventory."""
        self.world_state["player"]["inventory"].append(item)
        status = await self.check_inventory(context)
        return f"Added {item} to inventory. {status}"

    @function_tool()
    async def remove_from_inventory(self, context: RunContext, item: str) -> str:
        """Remove an item from player's inventory."""
        if item in self.world_state["player"]["inventory"]:
            self.world_state["player"]["inventory"].remove(item)
            status = await self.check_inventory(context)
            return f"Removed {item} from inventory. {status}"
        return f"{item} not found in inventory."

    @function_tool()
    async def update_health(self, context: RunContext, change: int) -> str:
        """Update player's health (positive or negative)."""
        self.world_state["player"]["health"] = max(
            0,
            min(
                self.world_state["player"]["health"] + change,
                self.world_state["player"]["max_health"],
            ),
        )

        status = "gained" if change > 0 else "lost"
        current_health = self.world_state["player"]["health"]

        if current_health <= 0:
            self.game_active = False
            return (
                f"You have {status} {abs(change)} health. Current health: 0/100. "
                "You feel too tired and faint. The adventure ends for now."
            )

        return f"You have {status} {abs(change)} health. Current health: {current_health}/100."

    @function_tool()
    async def move_to_location(self, context: RunContext, location: str) -> str:
        """Move player to a new location in Sunnyvale."""
        valid_locations = list(self.world_state["locations"].keys())
        if location not in valid_locations:
            return f"Unknown location. Valid locations: {', '.join(valid_locations)}"

        self._update_location(location)
        loc_data = self.world_state["locations"][location]
        return f"You have moved to {loc_data['name']}. {loc_data['description']}"

    @function_tool()
    async def complete_quest(self, context: RunContext, quest_name: str) -> str:
        """Mark a quest as completed."""
        if quest_name in self.world_state["quests"]:
            self.world_state["quests"][quest_name]["status"] = "completed"
            self.world_state["player"]["reputation"] += 10
            return (
                f"Quest '{self.world_state['quests'][quest_name]['name']}' completed! "
                "People in Sunnyvale are grateful to you."
            )
        return f"Quest '{quest_name}' not found."

    @function_tool()
    async def get_world_state(self, context: RunContext) -> str:
        """Get current world state summary (for GM context)."""
        player = self.world_state["player"]
        current_loc = self.world_state["locations"][player["location"]]

        active_quests = [q for q in self.world_state["quests"].values() if q["status"] == "active"]
        quest_status = active_quests[0]["name"] if active_quests else "No active quests"

        return (
            f"Location: {current_loc['name']} | "
            f"Health: {player['health']}/100 | "
            f"Active Quest: {quest_status} | "
            f"Turn: {self.world_state['game_state']['turn_count']}"
        )

    @function_tool()
    async def restart_game(self, context: RunContext) -> str:
        """Restart the game with fresh world state."""
        self.world_state = self._initialize_world_state()
        self.game_active = True
        return (
            "The story begins again. You are back in Sunnyvale Town Square on a bright morning, "
            "ready to help with a new adventure. What would you like to do next?"
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize the game master agent
    agent = GameMasterAgent()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
