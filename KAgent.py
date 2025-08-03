import re
import json
import random
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class MyExampleAgent(Agent):
    """
    A Gomoku AI agent that uses a language model to make strategic moves.
    Inherits from the base Agent class provided by the Gomoku framework.
    """

    def _setup(self):
        """
        Initialize the agent by setting up the language model client.
        This method is called once when the agent is created.
        """
        self.llm = OpenAIGomokuClient(model="gemma2-9b-it")

    async def get_move(self, game_state):
        """
        Generate the next move for the current game state using an LLM.

        Args:
            game_state: Current state of the Gomoku game board

        Returns:
            tuple: (row, col) coordinates of the chosen move
        """
        player = self.player.value
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        board_str = game_state.format_board("standard")
        board_size = game_state.board_size

        # Enhanced engine-style prompt
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional Gomoku (5-in-a-row) player acting as a tactical engine. "
                    f"You play as {player} and the opponent plays as {rival}. "
                    f"Evaluate the board using this priority order:\n"
                    f"1. If there is a move that wins immediately, choose it.\n"
                    f"2. If the opponent has a move that wins next turn, block it.\n"
                    f"3. If there is a move that creates an open 4, choose it.\n"
                    f"4. If there is a move that blocks opponent’s open 4, choose it.\n"
                    f"5. If there is a move that creates an open 3, choose it.\n"
                    f"6. Otherwise, play the strongest positional move that advances your attack or limits theirs.\n"
                    f"Output only the coordinates of the chosen move."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"""Here is the current board. The grid is {board_size}x{board_size}, with row and column indices labeled.
Cells contain:
- "." for empty
- "{player}" for your stones
- "{rival}" for opponent's stones

{board_str}

Return ONLY a JSON object with two integers: row and col. 
No extra text, no explanation, no code block.

Example:
{{ "row": 3, "col": 4 }}"""
                ),
            },
        ]

        content = await self.llm.complete(messages)

        # Parse the LLM response
        try:
            match = re.search(r"\{\s*\"row\"\s*:\s*\d+\s*,\s*\"col\"\s*:\s*\d+\s*\}", content)
            if match:
                move = json.loads(match.group(0))
                row, col = int(move.get("row", -1)), int(move.get("col", -1))

                if 0 <= row < board_size and 0 <= col < board_size:
                    if game_state.is_valid_move(row, col):
                        return (row, col)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback to random valid move
        legal_moves = game_state.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)

        return None
