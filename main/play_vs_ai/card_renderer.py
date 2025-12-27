"""
Card Renderer - Utility class for loading and rendering Scopa cards
"""
import pygame
import os

class CardRenderer:
    """Handles loading and rendering of card images"""
    
    def __init__(self, cards_folder: str, card_width: int = 80, card_height: int = 120):
        """
        Initialize the card renderer.
        
        Args:
            cards_folder: Path to the carte_napoletane folder
            card_width: Width to scale cards to
            card_height: Height to scale cards to
        """
        self.cards_folder = cards_folder
        self.card_width = card_width
        self.card_height = card_height
        self.card_images = {}
        self.back_image = None
        self._load_images()
    
    def _load_images(self):
        """Load all card images from the folder"""
        # Load card back
        back_path = os.path.join(self.cards_folder, "back.png")
        if os.path.exists(back_path):
            img = pygame.image.load(back_path).convert_alpha()
            self.back_image = pygame.transform.scale(img, (self.card_width, self.card_height))
        
        # Load all 40 cards (1.png to 40.png)
        for i in range(1, 41):
            card_path = os.path.join(self.cards_folder, f"{i}.png")
            if os.path.exists(card_path):
                img = pygame.image.load(card_path).convert_alpha()
                self.card_images[i] = pygame.transform.scale(img, (self.card_width, self.card_height))
    
    def get_card_image(self, card) -> pygame.Surface:
        """
        Get the image for a card.
        
        Args:
            card: Card object with .index attribute (0-39)
        
        Returns:
            pygame.Surface with the card image
        """
        # Card indices are 0-39, image files are 1-40
        image_index = card.index + 1
        return self.card_images.get(image_index, self.back_image)
    
    def get_back_image(self) -> pygame.Surface:
        """Get the card back image"""
        return self.back_image
    
    def draw_card(self, surface: pygame.Surface, card, x: int, y: int, face_up: bool = True):
        """
        Draw a card on the surface.
        
        Args:
            surface: Pygame surface to draw on
            card: Card object to draw
            x, y: Position to draw at
            face_up: Whether to show the card face or back
        """
        if face_up:
            img = self.get_card_image(card)
        else:
            img = self.back_image
        
        if img:
            surface.blit(img, (x, y))
    
    def draw_card_highlighted(self, surface: pygame.Surface, card, x: int, y: int, 
                               highlight_color=(255, 255, 0), border_width=3):
        """Draw a card with a highlight border"""
        # Draw highlight border
        pygame.draw.rect(surface, highlight_color, 
                        (x - border_width, y - border_width, 
                         self.card_width + 2*border_width, 
                         self.card_height + 2*border_width), 
                        border_width)
        # Draw the card
        self.draw_card(surface, card, x, y, face_up=True)
