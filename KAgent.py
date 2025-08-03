import re
import json
import random
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class MyExampleAgent(Agent):
    """
    A Gomoku AI agent with tactical pre-check (detect win/block moves) 
    and LLM strategy fallback.
    """

    def _setup(self):
        self.llm = OpenAIGomokuClient(model="gemma2-9b-it")

    def _find_immediate_win_or_block(self, game_state, player_symbol):
        """
        Scan board for immediate winning move or block opponent’s win.
        Checks horizontal, vertical, and both diagonals.
        """
        board = game_state.board
        size = game_state.board_size
        opponent_symbol = "O" if player_symbol == "X" else "X"

        def check_sequence(seq, base_r, base_c, dr, dc, symbol):
            """
            Check a sequence of 5 cells.
            If 4 are 'symbol' and 1 is '.', return the empty cell coordinates.
            """
            if seq.count(symbol) == 4 and seq.count(".") == 1:
                idx = seq.index(".")
                return (base_r + dr * idx, base_c + dc * idx)
            return None

        # Check all possible starting positions for 5-in-a-row
        for r in range(size):
            for c in range(size):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    end_r = r + dr * 4
                    end_c = c + dc * 4
                    if 0 <= end_r < size and 0 <= end_c < size:
                        # Extract sequence of 5 cells
                        seq = [board[r + dr * i][c + dc * i] for i in range(5)]

                        # Check for player's win
                        move = check_sequence(seq, r, c, dr, dc, player_symbol)
                        if move and game_state.is_valid_move(*move):
                            return move

                        # Check for blocking opponent's win
                        move = check_sequence(seq, r, c, dr, dc, opponent_symbol)
                        if move and game_state.is_valid_move(*move):
                            return move
        return None

    async def get_move(self, game_state):
        player_symbol = self.player.value
        rival_symbol = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value
        board_size = game_state.board_size

        # 1️⃣ Immediate win/block
        urgent_move = self._find_immediate_win_or_block(game_state, player_symbol)
        if urgent_move:
            return urgent_move

        # 2️⃣ LLM strategy
        board_str = game_state.format_board("standard")
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional Gomoku player. You play as {player_symbol}, opponent as {rival_symbol}. "
                    f"Priority:\n"
                    f"1. Create a win.\n"
                    f"2. Block opponent win.\n"
                    f"3. Create open 4.\n"
                    f"4. Block opponent open 4.\n"
                    f"5. Create open 3.\n"
                    f"6. Otherwise choose the strongest positional move."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"""Current board ({board_size}x{board_size}):
{board_str}

Return ONLY JSON:
{{ "row": <row>, "col": <col> }}"""
                ),
            },
        ]

        content = await self.llm.complete(messages)

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

        # 3️⃣ Fallback random
        legal_moves = game_state.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else None
