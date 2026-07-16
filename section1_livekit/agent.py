import logging

from livekit.agents import Agent, RunContext, function_tool

logger = logging.getLogger("airline_agent")

_FLIGHT_DB: dict[str, str] = {
    "BA123": "On time — scheduled departure 14:30 UTC from LHR, arrival 17:45 UTC at JFK.",
    "AA456": "Delayed by 45 minutes — new departure 09:15 UTC from LAX due to late inbound aircraft.",
    "LH789": "Boarding now — gate B12 at FRA, departure in 20 minutes.",
    "EK101": "Cancelled — please contact Emirates support to rebook.",
}


class AirlineAgent(Agent):
    """Airline booking support assistant."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful airline booking support assistant. "
                "You assist passengers with flight status enquiries, rebooking advice, "
                "and general travel questions. Always be polite and concise. "
                "When a passenger asks about a specific flight, use the get_flight_status tool "
                "to look it up — do not guess or invent flight information."
            )
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hello! Welcome to airline support. I can help you check flight status, "
            "rebooking options, and more. How can I assist you today?"
        )

    @function_tool
    async def get_flight_status(self, context: RunContext, flight_id: str) -> str:
        """Look up the current operational status of a flight.

        Use this tool whenever a passenger asks about a specific flight's status,
        delay, gate number, or departure and arrival times.

        Args:
            flight_id: The IATA flight identifier, e.g. 'BA123' or 'AA456'.
        """
        flight_id = flight_id.upper().strip()
        logger.info("[TOOL CALL] get_flight_status(flight_id=%r)", flight_id)

        result = _FLIGHT_DB.get(
            flight_id,
            f"Flight {flight_id} was not found in our system. "
            "Please verify the flight number and try again.",
        )

        logger.info("[TOOL RESULT] %r", result)
        return result
