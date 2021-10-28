from data.carla import Sensors, World, InfoText
import pygame
import carla
import numpy as np


class Game():
    def __init__(self):
        pygame.init()
        pygame.font.init()

        client: carla.Client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)

        self.world = World(client)
        self.sensors = Sensors(self.world.carla_world, self.world.ego_vehicle)
        self.info_text = InfoText()

    def loop(self):
        display = pygame.display.set_mode((self.sensors.display_width * 2 + self.info_text.info_text_width,
            self.sensors.display_height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
        clock = pygame.time.Clock()

        try:
            while True:
                clock.tick_busy_loop(100)
                
                surface_spectator = pygame.surfarray.make_surface(self.sensors.display_spectator_img.swapaxes(0, 1))
                display.blit(surface_spectator, (0, 0))

                surface_rgb_sensor = pygame.surfarray.make_surface(self.sensors.display_rgb_img.swapaxes(0, 1))
                display.blit(surface_rgb_sensor, (self.sensors.display_width, 0))

                surface_depth_sensor = pygame.surfarray.make_surface(self.sensors.display_depth_img.swapaxes(0, 1))
                display.blit(surface_depth_sensor, (0, self.sensors.display_height))

                surface_semseg_sensor = pygame.surfarray.make_surface(self.sensors.display_semseg_img.swapaxes(0, 1))
                display.blit(surface_semseg_sensor, (self.sensors.display_width, self.sensors.display_height))

                # fill text surface
                filler = np.zeros((self.info_text.info_text_width, self.sensors.display_height * 2, 3))
                surface_info_text = pygame.surfarray.make_surface(filler)
                display.blit(surface_info_text, (self.sensors.display_width * 2, 0))
                # add info text
                self.info_text.update_text(display, self.sensors.display_width * 2, self.world, self.sensors)

                pygame.display.flip()
        finally:
            self.sensors.destroy()
            self.world.destroy()
            pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.loop()
