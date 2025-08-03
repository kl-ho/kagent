import re
import json
import random
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class MyExampleAgent(Agent):
    def _setup(self):
        self.llm = OpenAIGomokuClient(model="gemma2-9b-it")
        self.debug = True

    def log(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

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
                                self.log(f"{reason} at {move}")
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

    async def get_move(self, game_state):
        player_symbol = self.player.value
        rival_symbol = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # 1️⃣ Immediate win/block
        move = self._find_threat_move(game_state, player_symbol, 1, "Immediate Win")
        if move:
            return move
        move = self._find_threat_move(game_state, rival_symbol, 1, "Immediate Block")
        if move:
            return move

        # 2️⃣ Score moves & pick top 3 for LLM
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            return None

        scored_moves = [(m, self._score_move(game_state, m, player_symbol, rival_symbol)) 
                        for m in legal_moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        top_moves = scored_moves[:3]
        self.log(f"Top 3 candidate moves for LLM: {top_moves}")

        # 3️⃣ Ask LLM to choose among candidates
        board_str = game_state.format_board("standard")
        moves_str = "\n".join([f"{i+1}. {m[0]} (score {m[1]})" for i, m in enumerate(top_moves)])
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional Gomoku player as {player_symbol}. "
                    f"Opponent is {rival_symbol}. Select the best move from the candidate list."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Board:\n{board_str}\n\n"
                    f"Candidate moves:\n{moves_str}\n\n"
                    f"Return ONLY JSON: {{\"row\": <row>, \"col\": <col>}}"
                ),
            },
        ]
        self.log("Sending prompt to LLM...")
        self.log(messages)

        content = await self.llm.complete(messages)
        self.log(f"LLM Raw Response: {content}")

        try:
            match = re.search(r"\{\s*\"row\"\s*:\s*\d+\s*,\s*\"col\"\s*:\s*\d+\s*\}", content)
            if match:
                move = json.loads(match.group(0))
                chosen_move = (int(move["row"]), int(move["col"]))
                if chosen_move in [m[0] for m in top_moves]:
                    self.log(f"LLM chose move {chosen_move}")
                    return chosen_move
        except Exception as e:
            self.log(f"LLM parsing failed: {e}")

        # 4️⃣ Fallback: highest scored move
        fallback = top_moves[0][0]
        self.log(f"Fallback to top scored move {fallback}")
        return fallback
