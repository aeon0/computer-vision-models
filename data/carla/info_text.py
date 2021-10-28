import carla
import pygame
from data.carla import World, Sensors


class InfoText():
    def __init__(self):
        self.font = pygame.font.SysFont("monospace", 12)
        self.info_text_width = 450 # in [px]

    def update_text(self, display, offset_x, world: World, sensors: Sensors):
        new_info = []
        new_info.append("Ego vel. (m/s): x={:.2f}, y={:.2f}, z={:.2f}".format(world.ego_vehicle.get_velocity().x, world.ego_vehicle.get_velocity().y, world.ego_vehicle.get_velocity().z))
        
        offset_y = 10
        for text in new_info:
            text_surface = self.font.render(text, False, (255, 255, 255))
            display.blit(text_surface, dest=(offset_x + 15, offset_y))
            offset_y = offset_y + 13
