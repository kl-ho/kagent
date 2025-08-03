import re
import json
import random
from gomoku import Agent
from gomoku.llm import OpenAIGomokuClient
from gomoku.core.models import Player


class MyExampleAgent(Agent):
    """
    A Gomoku AI agent with:
    - Immediate tactical checks
    - Anticipatory block/setup for open 3/4
    - Threat-based scoring
    - Two-move lookahead (light minimax)
    """

    def _setup(self):
        self.llm = OpenAIGomokuClient(model="gemma2-9b-it")

    def _check_sequence(self, seq, symbol, empty_required):
        """Check if a sequence of 5 cells contains stones and empty spaces for a threat."""
        return seq.count(symbol) == (5 - empty_required) and seq.count(".") == empty_required

    def _find_threat_move(self, game_state, symbol, empty_required):
        """Find a move that matches a specific threat pattern."""
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
                                return move
        return None

    def _score_move(self, game_state, move, player_symbol, rival_symbol):
        """Score a potential move based on threats and positioning."""
        r, c = move
        size = game_state.board_size
        board = game_state.board

        score = 0

        # Centrality bonus
        center = size // 2
        score += max(0, (center - abs(r - center)) + (center - abs(c - center)))

        # Adjacency bonus
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if board[nr][nc] in [player_symbol, rival_symbol]:
                        score += 2

        # Threat creation/blocking
        board[r][c] = player_symbol
        if self._find_threat_move(game_state, player_symbol, 1):
            score += 50  # Creates win
        if self._find_threat_move(game_state, rival_symbol, 1):
            score += 45  # Blocks opponent win
        if self._find_threat_move(game_state, player_symbol, 2):
            score += 20  # Creates open 3
        if self._find_threat_move(game_state, rival_symbol, 2):
            score += 15  # Blocks opponent open 3
        board[r][c] = "."

        return score

    def _lookahead(self, game_state, player_symbol, rival_symbol):
        """Evaluate moves considering opponent's best reply (depth=2)."""
        legal_moves = game_state.get_legal_moves()
        best_move = None
        best_score = float("-inf")

        for move in legal_moves:
            # Simulate AI move
            r, c = move
            game_state.board[r][c] = player_symbol
            ai_score = self._score_move(game_state, move, player_symbol, rival_symbol)

            # Opponent’s best counter
            opp_moves = game_state.get_legal_moves()
            worst_reply_score = float("inf")
            for opp_move in opp_moves:
                orr, occ = opp_move
                game_state.board[orr][occ] = rival_symbol
                reply_score = self._score_move(game_state, opp_move, rival_symbol, player_symbol)
                game_state.board[orr][occ] = "."
                worst_reply_score = min(worst_reply_score, reply_score)

            # Undo AI move
            game_state.board[r][c] = "."

            # Minimax: maximize AI score, minimize opponent’s best reply
            total_score = ai_score - worst_reply_score
            if total_score > best_score:
                best_score = total_score
                best_move = move

        return best_move

    async def get_move(self, game_state):
        player_symbol = self.player.value
        rival_symbol = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # 1️⃣ Immediate win/block
        move = self._find_threat_move(game_state, player_symbol, 1)
        if move:
            return move
        move = self._find_threat_move(game_state, rival_symbol, 1)
        if move:
            return move

        # 2️⃣ Two-move lookahead
        move = self._lookahead(game_state, player_symbol, rival_symbol)
        if move:
            return move

        # 3️⃣ Fallback random
        legal_moves = game_state.get_legal_moves()
        return random.choice(legal_moves) if legal_moves else None
