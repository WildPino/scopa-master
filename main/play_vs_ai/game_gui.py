"""
Scopa Game GUI - Pygame-based graphical interface for playing vs AI
"""
import pygame
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scopa_engine import ScopaEngine, Card, Suit
from play_vs_ai.card_renderer import CardRenderer
from play_vs_ai.ai_player import AIPlayer


GREEN_FELT = (34, 139, 34)       # Forest green
DARK_GREEN = (0, 100, 0)         # Darker green for contrast
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 140, 0)           # Warning color


class ScopaGUI:
    """Main game GUI class"""
    
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
        
        # Paths
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cards_folder = os.path.join(base_dir, "carte_napoletane")
        model_path = os.path.join(base_dir, "main", "models", "scopa_ai_mixed_latest.zip")
        
        # Initialize components
        self.card_renderer = CardRenderer(cards_folder, card_width=80, card_height=120)
        self.ai_player = AIPlayer(model_path)
        self.engine = ScopaEngine()
        
        # Game state
        self.selected_card = None
        self.selected_card_index = None
        self.game_over = False
        self.player_turn = True  # True = human's turn, False = AI's turn
        self.message = ""
        self.message_time = 0
        
        # Animation state
        self.ai_thinking = False
        self.ai_think_start = 0
        
        # Capture selection state
        self.capture_mode = False
        self.valid_captures = []
        self.selected_table_cards = []
        self.card_to_play = None
        
    def start_new_game(self):
        """Start a new game"""
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
    
    def show_message(self, msg: str, duration: float = 2.0):
        """Show a temporary message"""
        self.message = msg
        self.message_time = time.time() + duration
    
    def get_player_hand_positions(self):
        """Calculate positions for player's hand cards (bottom of screen)"""
        hand = self.engine.hands[0]
        if not hand:
            return []
        
        card_spacing = 100
        total_width = (len(hand) - 1) * card_spacing + self.card_renderer.card_width
        start_x = (self.width - total_width) // 2
        y = self.height - 150
        
        return [(start_x + i * card_spacing, y) for i in range(len(hand))]
    
    def get_ai_hand_positions(self):
        """Calculate positions for AI's hand cards (top of screen)"""
        hand = self.engine.hands[1]
        if not hand:
            return []
        
        card_spacing = 100
        total_width = (len(hand) - 1) * card_spacing + self.card_renderer.card_width
        start_x = (self.width - total_width) // 2
        y = 30
        
        return [(start_x + i * card_spacing, y) for i in range(len(hand))]
    
    def get_table_positions(self):
        """Calculate positions for table cards (center of screen)"""
        table = self.engine.table
        if not table:
            return []
        
        # Arrange in a grid-like pattern
        cards_per_row = min(5, len(table))
        rows = (len(table) + cards_per_row - 1) // cards_per_row
        
        card_spacing_x = 90
        card_spacing_y = 130
        
        positions = []
        for i, card in enumerate(table):
            row = i // cards_per_row
            col = i % cards_per_row
            cards_in_row = min(cards_per_row, len(table) - row * cards_per_row)
            
            total_width = (cards_in_row - 1) * card_spacing_x + self.card_renderer.card_width
            start_x = (self.width - total_width) // 2
            start_y = (self.height - rows * card_spacing_y) // 2
            
            x = start_x + col * card_spacing_x
            y = start_y + row * card_spacing_y
            positions.append((x, y))
        
        return positions
    
    def draw(self):
        """Draw the entire game screen"""
        # Background
        self.screen.fill(GREEN_FELT)
        
        # Draw decorative border
        pygame.draw.rect(self.screen, DARK_GREEN, (0, 0, self.width, self.height), 10)
        
        # Draw AI's hand (face down)
        ai_positions = self.get_ai_hand_positions()
        for i, pos in enumerate(ai_positions):
            if i < len(self.engine.hands[1]):
                img = self.card_renderer.get_back_image()
                if img:
                    self.screen.blit(img, pos)
        
        # Draw label for AI (with warning if heuristic)
        if self.ai_player.use_model:
            ai_label = self.font.render("🤖 AI (Modello Addestrato)", True, WHITE)
        else:
            ai_label = self.font.render("⚠️ AI EURISTICA", True, ORANGE)
        self.screen.blit(ai_label, (self.width // 2 - ai_label.get_width() // 2, 5))
        
        # Draw table cards
        table_positions = self.get_table_positions()
        for i, pos in enumerate(table_positions):
            if i < len(self.engine.table):
                card = self.engine.table[i]
                # Highlight if in capture selection mode
                if self.capture_mode and card in self.selected_table_cards:
                    self.card_renderer.draw_card_highlighted(self.screen, card, pos[0], pos[1], GOLD)
                else:
                    self.card_renderer.draw_card(self.screen, card, pos[0], pos[1], face_up=True)
        
        # Draw player's hand (face up)
        player_positions = self.get_player_hand_positions()
        for i, pos in enumerate(player_positions):
            if i < len(self.engine.hands[0]):
                card = self.engine.hands[0][i]
                if self.selected_card_index == i:
                    self.card_renderer.draw_card_highlighted(self.screen, card, pos[0], pos[1], YELLOW)
                else:
                    self.card_renderer.draw_card(self.screen, card, pos[0], pos[1], face_up=True)
        
        # Draw label for player
        player_label = self.font.render("🎮 Tu", True, WHITE)
        self.screen.blit(player_label, (self.width // 2 - player_label.get_width() // 2, self.height - 30))
        
        # Draw scores
        self._draw_scores()
        
        # Draw message
        if self.message and time.time() < self.message_time:
            msg_surface = self.font.render(self.message, True, GOLD)
            msg_rect = msg_surface.get_rect(center=(self.width // 2, self.height // 2 - 180))
            pygame.draw.rect(self.screen, BLACK, msg_rect.inflate(20, 10))
            self.screen.blit(msg_surface, msg_rect)
        
        # Draw instructions
        if self.capture_mode:
            instr = self.small_font.render("Clicca le carte da prendere, poi INVIO per confermare (ESC per annullare)", True, WHITE)
        elif self.player_turn:
            instr = self.small_font.render("Clicca una carta per giocarla", True, WHITE)
        else:
            instr = self.small_font.render("L'AI sta pensando...", True, WHITE)
        self.screen.blit(instr, (10, self.height - 60))
        
        # Draw game over screen
        if self.game_over:
            self._draw_game_over()
        
        # Draw heuristic warning if applicable
        if not self.ai_player.use_model:
            self._draw_heuristic_warning()
        
        pygame.display.flip()
    
    def _draw_heuristic_warning(self):
        """Draw a prominent warning when using heuristic AI"""
        warning_font = pygame.font.Font(None, 28)
        
        # Warning banner at top-right
        warning_text = "⚠️ ATTENZIONE: Stai giocando contro l'AI EURISTICA"
        subtext = "Il modello addestrato non è stato caricato"
        
        text_surface = warning_font.render(warning_text, True, ORANGE)
        sub_surface = self.small_font.render(subtext, True, YELLOW)
        
        # Background box
        box_width = max(text_surface.get_width(), sub_surface.get_width()) + 20
        box_height = text_surface.get_height() + sub_surface.get_height() + 15
        box_x = self.width - box_width - 10
        box_y = 160
        
        pygame.draw.rect(self.screen, BLACK, (box_x, box_y, box_width, box_height))
        pygame.draw.rect(self.screen, ORANGE, (box_x, box_y, box_width, box_height), 2)
        
        self.screen.blit(text_surface, (box_x + 10, box_y + 5))
        self.screen.blit(sub_surface, (box_x + 10, box_y + text_surface.get_height() + 8))
    
    def _draw_scores(self):
        """Draw the score display"""
        # Left side - scores
        y = 200
        
        # Scope
        scope_text = self.small_font.render(f"Scope: Tu {self.engine.scope[0]} - AI {self.engine.scope[1]}", True, WHITE)
        self.screen.blit(scope_text, (20, y))
        
        # Cards captured
        cards_text = self.small_font.render(f"Carte: Tu {len(self.engine.captured[0])} - AI {len(self.engine.captured[1])}", True, WHITE)
        self.screen.blit(cards_text, (20, y + 25))
        
        # Deck remaining
        deck_text = self.small_font.render(f"Mazzo: {len(self.engine.deck)} carte", True, WHITE)
        self.screen.blit(deck_text, (20, y + 50))
    
    def _draw_game_over(self):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.fill(BLACK)
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))
        
        # Calculate final score
        self.engine.finalize_game()
        scores = self.engine.calculate_score()
        
        # Title
        title = self.large_font.render("PARTITA FINITA!", True, GOLD)
        title_rect = title.get_rect(center=(self.width // 2, 200))
        self.screen.blit(title, title_rect)
        
        # Scores
        score_text = self.font.render(f"Tu: {scores[0]} - AI: {scores[1]}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.width // 2, 300))
        self.screen.blit(score_text, score_rect)
        
        # Winner
        if scores[0] > scores[1]:
            winner = "HAI VINTO! 🎉"
            color = GOLD
        elif scores[1] > scores[0]:
            winner = "L'AI ha vinto 🤖"
            color = RED
        else:
            winner = "Pareggio!"
            color = WHITE
        
        winner_text = self.large_font.render(winner, True, color)
        winner_rect = winner_text.get_rect(center=(self.width // 2, 400))
        self.screen.blit(winner_text, winner_rect)
        
        # Restart instruction
        restart = self.font.render("Premi SPAZIO per rigiocare, ESC per uscire", True, WHITE)
        restart_rect = restart.get_rect(center=(self.width // 2, 550))
        self.screen.blit(restart, restart_rect)
    
    def handle_click(self, pos):
        """Handle mouse click"""
        if self.game_over or not self.player_turn:
            return
        
        if self.capture_mode:
            # Check if clicked on a table card
            table_positions = self.get_table_positions()
            for i, card_pos in enumerate(table_positions):
                if i < len(self.engine.table):
                    rect = pygame.Rect(card_pos[0], card_pos[1], 
                                      self.card_renderer.card_width, 
                                      self.card_renderer.card_height)
                    if rect.collidepoint(pos):
                        card = self.engine.table[i]
                        if card in self.selected_table_cards:
                            self.selected_table_cards.remove(card)
                        else:
                            self.selected_table_cards.append(card)
                        return
        else:
            # Check if clicked on a player's card
            player_positions = self.get_player_hand_positions()
            for i, card_pos in enumerate(player_positions):
                if i < len(self.engine.hands[0]):
                    rect = pygame.Rect(card_pos[0], card_pos[1], 
                                      self.card_renderer.card_width, 
                                      self.card_renderer.card_height)
                    if rect.collidepoint(pos):
                        self.selected_card_index = i
                        self.selected_card = self.engine.hands[0][i]
                        self._handle_card_selection()
                        return
    
    def _handle_card_selection(self):
        """Handle when player selects a card to play"""
        if self.selected_card is None:
            return
        
        # Get valid captures for this card
        legal_moves = self.engine.get_legal_moves(0)
        valid_for_card = [(card, capture) for card, capture in legal_moves 
                         if card.index == self.selected_card.index]
        
        if not valid_for_card:
            return
        
        # Check capture options
        captures = [capture for _, capture in valid_for_card]
        non_empty_captures = [c for c in captures if c]
        
        if not non_empty_captures:
            # No captures possible - just drop the card
            self._execute_player_move(self.selected_card, [])
        elif len(non_empty_captures) == 1:
            # Only one capture option
            self._execute_player_move(self.selected_card, non_empty_captures[0])
        else:
            # Multiple capture options - enter capture selection mode
            self.capture_mode = True
            self.valid_captures = non_empty_captures
            self.card_to_play = self.selected_card
            self.selected_table_cards = []
            self.show_message("Seleziona le carte da prendere")
    
    def _confirm_capture(self):
        """Confirm the selected capture in capture mode"""
        if not self.capture_mode:
            return
        
        # Check if selected cards form a valid capture
        selected_tuple = tuple(sorted([c.index for c in self.selected_table_cards]))
        
        for capture in self.valid_captures:
            capture_tuple = tuple(sorted([c.index for c in capture]))
            if selected_tuple == capture_tuple:
                self._execute_player_move(self.card_to_play, capture)
                self.capture_mode = False
                self.selected_table_cards = []
                self.card_to_play = None
                return
        
        self.show_message("Combinazione non valida!")
    
    def _cancel_capture(self):
        """Cancel capture mode"""
        self.capture_mode = False
        self.selected_table_cards = []
        self.card_to_play = None
        self.selected_card = None
        self.selected_card_index = None
    
    def _execute_player_move(self, card, capture):
        """Execute the player's move"""
        self.engine.apply_move(0, card, capture)
        self.selected_card = None
        self.selected_card_index = None
        
        if capture:
            if not self.engine.table and len(self.engine.deck) > 0:
                self.show_message("SCOPA! 🎉")
            else:
                self.show_message(f"Hai preso {len(capture)} carte")
        else:
            self.show_message("Carta calata")
        
        # Check for end of round
        self._check_deal_or_end()
        
        if not self.game_over:
            self.player_turn = False
            self.ai_thinking = True
            self.ai_think_start = time.time()
    
    def _do_ai_turn(self):
        """Execute AI's turn"""
        if not self.ai_thinking or self.game_over:
            return
        
        # Wait a bit for visual effect
        if time.time() - self.ai_think_start < 1.0:
            return
        
        self.ai_thinking = False
        
        # Get AI's move
        card, capture = self.ai_player.get_move(self.engine, player_index=1)
        
        if card is None:
            self.player_turn = True
            return
        
        # Execute the move
        self.engine.apply_move(1, card, capture)
        
        if capture:
            if not self.engine.table and len(self.engine.deck) > 0:
                self.show_message("L'AI ha fatto SCOPA! 🤖")
            else:
                self.show_message(f"L'AI ha preso {len(capture)} carte")
        else:
            self.show_message("L'AI ha calato una carta")
        
        # Check for end of round
        self._check_deal_or_end()
        
        if not self.game_over:
            self.player_turn = True
    
    def _check_deal_or_end(self):
        """Check if we need to deal new cards or if game is over"""
        # Check if both hands are empty
        if not self.engine.hands[0] and not self.engine.hands[1]:
            if self.engine.deck:
                # Deal new hands
                self.engine.deal_new_hand()
                self.show_message("Nuove carte distribuite!")
            else:
                # Game over
                self.game_over = True
    
    def run(self):
        """Main game loop"""
        self.start_new_game()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.capture_mode:
                            self._cancel_capture()
                        elif self.game_over:
                            running = False
                        else:
                            running = False
                    
                    elif event.key == pygame.K_RETURN:
                        if self.capture_mode:
                            self._confirm_capture()
                    
                    elif event.key == pygame.K_SPACE:
                        if self.game_over:
                            self.start_new_game()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
            
            # AI turn
            if not self.player_turn and not self.game_over:
                self._do_ai_turn()
            
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    game = ScopaGUI()
    game.run()


if __name__ == "__main__":
    main()
