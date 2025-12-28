"""
Scopa Game GUI - Pygame-based graphical interface for playing vs AI.

Usage:
    python -m scopa.gui
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import pygame

from scopa.game import Card, Suit, ScopaEngine
from scopa.config import MODELS_DIR
from scopa.gui.card_renderer import CardRenderer
from scopa.gui.ai_player import AIPlayer


# Colors
GREEN_FELT = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 140, 0)


class ScopaGUI:
    """Main game GUI class."""
    
    def __init__(self, width: int = 1024, height: int = 768):
        pygame.init()
        pygame.display.set_caption("Scopa vs AI")
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 72)
        
        # Components
        self.card_renderer = CardRenderer()
        self.ai_player = AIPlayer(str(MODELS_DIR / "scopa_ai_latest"))
        self.engine = ScopaEngine()
        
        # Game state
        self.selected_card: Optional[Card] = None
        self.selected_card_index: Optional[int] = None
        self.game_over = False
        self.player_turn = True
        self.message = ""
        self.message_time = 0.0
        
        # Animation
        self.ai_thinking = False
        self.ai_think_start = 0.0
        
        # Capture selection
        self.capture_mode = False
        self.valid_captures: List[List[Card]] = []
        self.selected_table_cards: List[Card] = []
        self.card_to_play: Optional[Card] = None
    
    def start_new_game(self) -> None:
        """Start a new game."""
        self.engine.reset()
        self.selected_card = None
        self.selected_card_index = None
        self.game_over = False
        self.player_turn = True
        self.capture_mode = False
        self.valid_captures = []
        self.selected_table_cards = []
        self.card_to_play = None
        self.show_message("Partita iniziata! È il tuo turno.")
    
    def show_message(self, msg: str, duration: float = 2.0) -> None:
        """Show a temporary message."""
        self.message = msg
        self.message_time = time.time() + duration
    
    def _get_hand_positions(self, hand: List[Card], y: int) -> List[Tuple[int, int]]:
        """Calculate card positions for a hand."""
        if not hand:
            return []
        card_spacing = 100
        total = (len(hand) - 1) * card_spacing + self.card_renderer.card_width
        start_x = (self.width - total) // 2
        return [(start_x + i * card_spacing, y) for i in range(len(hand))]
    
    def _get_table_positions(self) -> List[Tuple[int, int]]:
        """Calculate positions for table cards."""
        table = self.engine.table
        if not table:
            return []
        
        per_row = min(5, len(table))
        rows = (len(table) + per_row - 1) // per_row
        spacing_x, spacing_y = 90, 130
        
        positions = []
        for i in range(len(table)):
            row, col = i // per_row, i % per_row
            cards_in_row = min(per_row, len(table) - row * per_row)
            total = (cards_in_row - 1) * spacing_x + self.card_renderer.card_width
            start_x = (self.width - total) // 2
            start_y = (self.height - rows * spacing_y) // 2
            positions.append((start_x + col * spacing_x, start_y + row * spacing_y))
        
        return positions
    
    def draw(self) -> None:
        """Draw the game screen."""
        self.screen.fill(GREEN_FELT)
        pygame.draw.rect(self.screen, DARK_GREEN, (0, 0, self.width, self.height), 10)
        
        # AI's hand (face down)
        for i, pos in enumerate(self._get_hand_positions(self.engine.hands[1], 30)):
            if i < len(self.engine.hands[1]):
                img = self.card_renderer.get_back_image()
                if img:
                    self.screen.blit(img, pos)
        
        # AI label
        ai_label = self.font.render(
            "🤖 AI" if self.ai_player.use_model else "⚠️ AI EURISTICA",
            True, WHITE if self.ai_player.use_model else ORANGE
        )
        self.screen.blit(ai_label, (self.width // 2 - ai_label.get_width() // 2, 5))
        
        # Table cards
        for i, pos in enumerate(self._get_table_positions()):
            if i < len(self.engine.table):
                card = self.engine.table[i]
                if self.capture_mode and card in self.selected_table_cards:
                    self.card_renderer.draw_card_highlighted(self.screen, card, pos[0], pos[1], GOLD)
                else:
                    self.card_renderer.draw_card(self.screen, card, pos[0], pos[1])
        
        # Player's hand
        for i, pos in enumerate(self._get_hand_positions(self.engine.hands[0], self.height - 150)):
            if i < len(self.engine.hands[0]):
                card = self.engine.hands[0][i]
                if self.selected_card_index == i:
                    self.card_renderer.draw_card_highlighted(self.screen, card, pos[0], pos[1], YELLOW)
                else:
                    self.card_renderer.draw_card(self.screen, card, pos[0], pos[1])
        
        # Player label
        label = self.font.render("🎮 Tu", True, WHITE)
        self.screen.blit(label, (self.width // 2 - label.get_width() // 2, self.height - 30))
        
        # Scores
        self._draw_scores()
        
        # Message
        if self.message and time.time() < self.message_time:
            surf = self.font.render(self.message, True, GOLD)
            rect = surf.get_rect(center=(self.width // 2, self.height // 2 - 180))
            pygame.draw.rect(self.screen, BLACK, rect.inflate(20, 10))
            self.screen.blit(surf, rect)
        
        # Instructions
        if self.capture_mode:
            instr = "Seleziona carte, INVIO per confermare (ESC annulla)"
        elif self.player_turn:
            instr = "Clicca una carta per giocarla"
        else:
            instr = "L'AI sta pensando..."
        self.screen.blit(self.small_font.render(instr, True, WHITE), (10, self.height - 60))
        
        if self.game_over:
            self._draw_game_over()
        
        pygame.display.flip()
    
    def _draw_scores(self) -> None:
        """Draw score display."""
        y = 200
        texts = [
            f"Scope: Tu {self.engine.scope[0]} - AI {self.engine.scope[1]}",
            f"Carte: Tu {len(self.engine.captured[0])} - AI {len(self.engine.captured[1])}",
            f"Mazzo: {len(self.engine.deck)} carte"
        ]
        for i, txt in enumerate(texts):
            self.screen.blit(self.small_font.render(txt, True, WHITE), (20, y + i * 25))
    
    def _draw_game_over(self) -> None:
        """Draw game over screen."""
        overlay = pygame.Surface((self.width, self.height))
        overlay.fill(BLACK)
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        self.engine.finalize_game()
        scores = self.engine.calculate_score()
        
        title = self.large_font.render("PARTITA FINITA!", True, GOLD)
        self.screen.blit(title, title.get_rect(center=(self.width // 2, 200)))
        
        score_txt = self.font.render(f"Tu: {scores[0]} - AI: {scores[1]}", True, WHITE)
        self.screen.blit(score_txt, score_txt.get_rect(center=(self.width // 2, 300)))
        
        if scores[0] > scores[1]:
            winner, color = "HAI VINTO! 🎉", GOLD
        elif scores[1] > scores[0]:
            winner, color = "L'AI ha vinto 🤖", RED
        else:
            winner, color = "Pareggio!", WHITE
        
        self.screen.blit(
            self.large_font.render(winner, True, color),
            self.large_font.render(winner, True, color).get_rect(center=(self.width // 2, 400))
        )
        
        restart = self.font.render("SPAZIO per rigiocare, ESC per uscire", True, WHITE)
        self.screen.blit(restart, restart.get_rect(center=(self.width // 2, 550)))
    
    def handle_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse click."""
        if self.game_over or not self.player_turn:
            return
        
        if self.capture_mode:
            for i, p in enumerate(self._get_table_positions()):
                if i < len(self.engine.table):
                    rect = pygame.Rect(p[0], p[1], self.card_renderer.card_width, self.card_renderer.card_height)
                    if rect.collidepoint(pos):
                        card = self.engine.table[i]
                        if card in self.selected_table_cards:
                            self.selected_table_cards.remove(card)
                        else:
                            self.selected_table_cards.append(card)
                        return
        else:
            positions = self._get_hand_positions(self.engine.hands[0], self.height - 150)
            for i, p in enumerate(positions):
                if i < len(self.engine.hands[0]):
                    rect = pygame.Rect(p[0], p[1], self.card_renderer.card_width, self.card_renderer.card_height)
                    if rect.collidepoint(pos):
                        self.selected_card_index = i
                        self.selected_card = self.engine.hands[0][i]
                        self._handle_card_selection()
                        return
    
    def _handle_card_selection(self) -> None:
        """Handle card selection."""
        if not self.selected_card:
            return
        
        moves = self.engine.get_legal_moves(0)
        valid = [(c, cap) for c, cap in moves if c.index == self.selected_card.index]
        
        if not valid:
            return
        
        captures = [cap for _, cap in valid if cap]
        
        if not captures:
            self._execute_move(self.selected_card, [])
        elif len(captures) == 1:
            self._execute_move(self.selected_card, captures[0])
        else:
            self.capture_mode = True
            self.valid_captures = captures
            self.card_to_play = self.selected_card
            self.selected_table_cards = []
            self.show_message("Seleziona le carte da prendere")
    
    def _confirm_capture(self) -> None:
        """Confirm capture selection."""
        if not self.capture_mode:
            return
        
        selected_set = set(c.index for c in self.selected_table_cards)
        
        for capture in self.valid_captures:
            if set(c.index for c in capture) == selected_set:
                self._execute_move(self.card_to_play, capture)
                self.capture_mode = False
                self.selected_table_cards = []
                self.card_to_play = None
                return
        
        self.show_message("Combinazione non valida!")
    
    def _execute_move(self, card: Card, capture: List[Card]) -> None:
        """Execute player's move."""
        scopa = self.engine.apply_move(0, card, capture)
        self.selected_card = None
        self.selected_card_index = None
        
        if capture:
            if scopa:
                self.show_message("SCOPA! 🎉")
            else:
                self.show_message(f"Hai preso {len(capture)} carte")
        else:
            self.show_message("Carta calata")
        
        self._check_deal_or_end()
        
        if not self.game_over:
            self.player_turn = False
            self.ai_thinking = True
            self.ai_think_start = time.time()
    
    def _do_ai_turn(self) -> None:
        """Execute AI's turn."""
        if not self.ai_thinking or self.game_over:
            return
        
        if time.time() - self.ai_think_start < 1.0:
            return
        
        self.ai_thinking = False
        
        card, capture = self.ai_player.get_move(self.engine, player_index=1)
        
        if card is None:
            self.player_turn = True
            return
        
        scopa = self.engine.apply_move(1, card, capture)
        
        if capture:
            if scopa:
                self.show_message("L'AI ha fatto SCOPA! 🤖")
            else:
                self.show_message(f"L'AI ha preso {len(capture)} carte")
        else:
            self.show_message("L'AI ha calato una carta")
        
        self._check_deal_or_end()
        
        if not self.game_over:
            self.player_turn = True
    
    def _check_deal_or_end(self) -> None:
        """Check if need to deal or game over."""
        if not self.engine.hands[0] and not self.engine.hands[1]:
            if self.engine.deck:
                self.engine.deal_new_hand()
                self.show_message("Nuove carte!")
            else:
                self.game_over = True
    
    def run(self) -> None:
        """Main game loop."""
        self.start_new_game()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.capture_mode:
                            self.capture_mode = False
                            self.selected_table_cards = []
                            self.card_to_play = None
                        else:
                            running = False
                    elif event.key == pygame.K_RETURN and self.capture_mode:
                        self._confirm_capture()
                    elif event.key == pygame.K_SPACE and self.game_over:
                        self.start_new_game()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
            
            if not self.player_turn and not self.game_over:
                self._do_ai_turn()
            
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Entry point."""
    print("=" * 50)
    print("  SCOPA vs AI")
    print("=" * 50)
    print("\nControlli:")
    print("  - Clicca carta per giocarla")
    print("  - INVIO per confermare presa")
    print("  - ESC per uscire")
    print("  - SPAZIO per nuova partita")
    print()
    
    game = ScopaGUI()
    game.run()


if __name__ == "__main__":
    main()
