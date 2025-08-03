import re
import json
import random
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class MyExampleAgent(Agent):
    def _setup(self):
        # 🔹 Updated model to Qwen3-8B
        self.llm = OpenAIGomokuClient(model="Qwen/Qwen3-8B")
        self.debug = True

    def log(self, msg):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def _check_pattern(self, seq, symbol, count_stones, count_empty):
        return seq.count(symbol) == count_stones and seq.count(".") == count_empty

    def _find_threat_move(self, game_state, symbol, stones_needed, empty_needed, reason=""):
        board = game_state.board
        size = game_state.board_size
        for r in range(size):
            for c in range(size):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    end_r = r + dr * 4
                    end_c = c + dc * 4
                    if 0 <= end_r < size and 0 <= end_c < size:
                        seq = [board[r + dr * i][c + dc * i] for i in range(5)]
                        if self._check_pattern(seq, symbol, stones_needed, empty_needed):
                            idx = seq.index(".")
                            move = (r + dr * idx, c + dc * idx)
                            if game_state.is_valid_move(*move):
                                if reason:
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
        if self._find_threat_move(game_state, player_symbol, 4, 1):
            score += 50
        if self._find_threat_move(game_state, rival_symbol, 4, 1):
            score += 45
        if self._find_threat_move(game_state, player_symbol, 3, 2):
            score += 20
        if self._find_threat_move(game_state, rival_symbol, 3, 2):
            score += 15
        board[r][c] = "."
        return score

    async def get_move(self, game_state):
        player_symbol = self.player.value
        rival_symbol = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # Immediate win/block
        move = self._find_threat_move(game_state, player_symbol, 4, 1, "Immediate Win")
        if move: return move
        move = self._find_threat_move(game_state, rival_symbol, 4, 1, "Immediate Block")
        if move: return move
        move = self._find_threat_move(game_state, rival_symbol, 3, 2, "Block 3-in-a-row")
        if move: return move

        # Score moves
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            return None
        scored_moves = [(m, self._score_move(game_state, m, player_symbol, rival_symbol)) for m in legal_moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        top_moves = scored_moves[:3]
        self.log(f"Top 3 moves for LLM: {top_moves}")

        # Threat summary
        threat_summary = []
        opp4 = self._find_threat_move(game_state, rival_symbol, 4, 1)
        if opp4:
            threat_summary.append(f"Opponent has open-4 at {opp4}, blocking is critical.")
        my4 = self._find_threat_move(game_state, player_symbol, 4, 1)
        if my4:
            threat_summary.append(f"You have open-4 at {my4}, winning move possible.")
        opp3 = self._find_threat_move(game_state, rival_symbol, 3, 2)
        if opp3:
            threat_summary.append(f"Opponent has open-3 at {opp3}, blocking is urgent.")
        my3 = self._find_threat_move(game_state, player_symbol, 3, 2)
        if my3:
            threat_summary.append(f"You have open-3 at {my3}, attack opportunity.")

        threat_text = "\n".join(threat_summary) if threat_summary else "No immediate urgent threats detected."
        board_str = game_state.format_board("standard")
        moves_str = "\n".join([f"{i+1}. {m[0]} (score {m[1]})" for i, m in enumerate(top_moves)])

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional Gomoku player playing as {player_symbol}. "
                    f"Opponent is {rival_symbol}.\n"
                    f"Think deeply but output only JSON.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Board:\n{board_str}\n\n"
                    f"Threat Analysis:\n{threat_text}\n\n"
                    f"Candidate Moves:\n{moves_str}\n\n"
                    f"Output ONLY JSON: {{\"row\": <row>, \"col\": <col>}}"
                ),
            },
        ]
        self.log("Sending to LLM...")
        self.log(messages)

        content = await self.llm.complete(messages)
        self.log(f"LLM Response: {content}")

        try:
            match = re.search(r"\{\s*\"row\"\s*:\s*\d+\s*,\s*\"col\"\s*:\s*\d+\s*\}", content)
            if match:
                move = json.loads(match.group(0))
                return (int(move["row"]), int(move["col"]))
        except Exception as e:
            self.log(f"LLM parsing failed: {e}")

        return top_moves[0][0]
