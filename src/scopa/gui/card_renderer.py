"""
Card Renderer - Utility class for loading and rendering Scopa cards.
"""
from __future__ import annotations

import pygame
from pathlib import Path
from typing import Optional, Dict

from scopa.game import Card
from scopa.config import CARDS_DIR


class CardRenderer:
    """
    Handles loading and rendering of card images.
    
    Args:
        cards_folder: Path to carte_napoletane folder
        card_width: Width to scale cards to
        card_height: Height to scale cards to
    """
    
    def __init__(
        self,
        cards_folder: Optional[Path] = None,
        card_width: int = 80,
        card_height: int = 120
    ):
        self.cards_folder = Path(cards_folder) if cards_folder else CARDS_DIR
        self.card_width = card_width
        self.card_height = card_height
        self.card_images: Dict[int, pygame.Surface] = {}
        self.back_image: Optional[pygame.Surface] = None
        self._load_images()
    
    def _load_images(self) -> None:
        """Load all card images from the folder."""
        # Card back
        back_path = self.cards_folder / "back.png"
        if back_path.exists():
            img = pygame.image.load(str(back_path)).convert_alpha()
            self.back_image = pygame.transform.scale(img, (self.card_width, self.card_height))
        
        # 40 cards (1.png to 40.png)
        for i in range(1, 41):
            card_path = self.cards_folder / f"{i}.png"
            if card_path.exists():
                img = pygame.image.load(str(card_path)).convert_alpha()
                self.card_images[i] = pygame.transform.scale(img, (self.card_width, self.card_height))
    
    def get_card_image(self, card: Card) -> Optional[pygame.Surface]:
        """Get the image for a card."""
        # Card indices are 0-39, image files are 1-40
        return self.card_images.get(card.index + 1, self.back_image)
    
    def get_back_image(self) -> Optional[pygame.Surface]:
        """Get the card back image."""
        return self.back_image
    
    def draw_card(
        self,
        surface: pygame.Surface,
        card: Card,
        x: int,
        y: int,
        face_up: bool = True
    ) -> None:
        """Draw a card on the surface."""
        img = self.get_card_image(card) if face_up else self.back_image
        if img:
            surface.blit(img, (x, y))
    
    def draw_card_highlighted(
        self,
        surface: pygame.Surface,
        card: Card,
        x: int,
        y: int,
        highlight_color: tuple = (255, 255, 0),
        border_width: int = 3
    ) -> None:
        """Draw a card with a highlight border."""
        pygame.draw.rect(
            surface, highlight_color,
            (x - border_width, y - border_width,
             self.card_width + 2 * border_width,
             self.card_height + 2 * border_width),
            border_width
        )
        self.draw_card(surface, card, x, y, face_up=True)
