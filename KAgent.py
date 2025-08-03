import re
import json
import random
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class MyExampleAgent(Agent):
    def _setup(self):
        self.llm = OpenAIGomokuClient(model="gemma2-9b-it")
        self.debug = True  # Turn on/off logging

    def log(self, message):
        """Helper to print logs when debug mode is on."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def _check_sequence(self, seq, symbol, empty_required):
        return seq.count(symbol) == (5 - empty_required) and seq.count(".") == empty_required

    def _find_threat_move(self, game_state, symbol, empty_required, reason=""):
        board = game_state.board
        size = game_state.board_size

        for r in range(size):
            for c in range(size):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    end_r = r + dr * 4
                    end_c = c + dc * 4
                    if 0 <= end_r < size and 0 <= end_c < size:
                        seq = [board[r + dr * i][c + dc * i] for i in range(5)]
                        if self._check_sequence(seq, symbol, empty_required):
                            idx = seq.index(".")
                            move = (r + dr * idx, c + dc * idx)
                            if game_state.is_valid_move(*move):
                                self.log(f"{reason} found at {move}")
                                return move
        return None

    def _score_move(self, game_state, move, player_symbol, rival_symbol):
        r, c = move
        size = game_state.board_size
        board = game_state.board

        score = 0
        center = size // 2
        score += max(0, (center - abs(r - center)) + (center - abs(c - center)))

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if board[nr][nc] in [player_symbol, rival_symbol]:
                        score += 2

        board[r][c] = player_symbol
        if self._find_threat_move(game_state, player_symbol, 1):
            score += 50
        if self._find_threat_move(game_state, rival_symbol, 1):
            score += 45
        if self._find_threat_move(game_state, player_symbol, 2):
            score += 20
        if self._find_threat_move(game_state, rival_symbol, 2):
            score += 15
        board[r][c] = "."
        return score

    def _lookahead(self, game_state, player_symbol, rival_symbol):
        legal_moves = game_state.get_legal_moves()
        best_move = None
        best_score = float("-inf")

        for move in legal_moves:
            r, c = move
            game_state.board[r][c] = player_symbol
            ai_score = self._score_move(game_state, move, player_symbol, rival_symbol)

            opp_moves = game_state.get_legal_moves()
            worst_reply_score = float("inf")
            for opp_move in opp_moves:
                orr, occ = opp_move
                game_state.board[orr][occ] = rival_symbol
                reply_score = self._score_move(game_state, opp_move, rival_symbol, player_symbol)
                game_state.board[orr][occ] = "."
                worst_reply_score = min(worst_reply_score, reply_score)

            game_state.board[r][c] = "."
            total_score = ai_score - worst_reply_score

            if total_score > best_score:
                best_score = total_score
                best_move = move

        self.log(f"Lookahead selected move {best_move} with score {best_score}")
        return best_move

    async def get_move(self, game_state):
        player_symbol = self.player.value
        rival_symbol = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # 1️⃣ Immediate win/block
        move = self._find_threat_move(game_state, player_symbol, 1, reason="Immediate Win")
        if move:
            return move
        move = self._find_threat_move(game_state, rival_symbol, 1, reason="Immediate Block")
        if move:
            return move

        # 2️⃣ Lookahead
        move = self._lookahead(game_state, player_symbol, rival_symbol)
        if move:
            return move

        # 3️⃣ LLM fallback
        self.log("No tactical move found — using LLM")
        board_str = game_state.format_board("standard")
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional Gomoku player. You play as {player_symbol}, opponent as {rival_symbol}. "
                    f"Choose the best move according to win/block priorities."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Board:\n{board_str}\nReturn JSON: {{\"row\": <row>, \"col\": <col>}}"
                ),
            },
        ]
        self.log(f"LLM Prompt:\n{messages}")

        content = await self.llm.complete(messages)
        self.log(f"LLM Raw Response: {content}")

        try:
            match = re.search(r"\{\s*\"row\"\s*:\s*\d+\s*,\s*\"col\"\s*:\s*\d+\s*\}", content)
            if match:
                move = json.loads(match.group(0))
                row, col = int(move.get("row", -1)), int(move.get("col", -1))
                if game_state.is_valid_move(row, col):
                    self.log(f"LLM chose move {(row, col)}")
                    return (row, col)
        except Exception as e:
            self.log(f"LLM parsing failed: {e}")

        # 4️⃣ Fallback random
        legal_moves = game_state.get_legal_moves()
        fallback = random.choice(legal_moves) if legal_moves else None
        self.log(f"Fallback random move {fallback}")
        return fallback
