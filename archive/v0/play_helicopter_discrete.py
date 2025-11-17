import pygame
import numpy as np
from tasks import DiscretePredictiveInferenceEnv

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
BUCKET_WIDTH = 100
BUCKET_HEIGHT = 50
BAG_SIZE = 30
FPS = 30
BAG_DELAY = 1500  # Delay in milliseconds before showing the bag
DISPLAY_DURATION = 2500  # Total display time for bucket and bag

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Positions
positions = [0, 1, 2, 3, 4]
position_coords = {0: 100, 1: 250, 2: 400, 3: 550, 4: 700}

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Discrete Predictive Inference Task")
clock = pygame.time.Clock()

# Game loop using the gym environment
def main():
    env = DiscretePredictiveInferenceEnv(condition="change-point")
    obs = env.reset()
    trial_start = True
    waiting_for_space = False
    action_made = False
    bucket_drawn = False
    bag_drawn = False
    reward = 0
    action_timer_start = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if waiting_for_space and event.key == pygame.K_SPACE:
                    waiting_for_space = False
                    trial_start = True
                    action_made = False
                    bucket_drawn = False
                    bag_drawn = False
                elif not waiting_for_space and not action_made:
                    if event.key == pygame.K_1:
                        action = 0  # Action 1 corresponds to position 0
                    elif event.key == pygame.K_2:
                        action = 1  # Action 2 corresponds to position 1
                    elif event.key == pygame.K_3:
                        action = 2  # Action 3 corresponds to position 2
                    elif event.key == pygame.K_4:
                        action = 3  # Action 4 corresponds to position 3
                    elif event.key == pygame.K_5:
                        action = 4  # Action 5 corresponds to position 4
                    else:
                        continue

                    # Execute action in the environment
                    obs, reward, done, _ = env.step(action)
                    action_timer_start = pygame.time.get_ticks()
                    action_made = True
                    bucket_drawn = True

                    if done:
                        obs = env.reset()
                        trial_start = True
                        action_made = False
                        bucket_drawn = False
                        bag_drawn = False

        # Draw environment
        screen.fill(WHITE)
        
        if trial_start:
            font = pygame.font.SysFont(None, 55)
            text = font.render("New Trial", True, BLACK)
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
            pygame.display.flip()
            pygame.time.wait(1000)  # Wait for 1 second
            trial_start = False
        elif waiting_for_space:
            font = pygame.font.SysFont(None, 55)
            text = font.render("Press SPACE to continue", True, BLACK)
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
            reward_text = font.render(f"Reward: {int(reward)}", True, BLACK)
            screen.blit(reward_text, (SCREEN_WIDTH // 2 - reward_text.get_width() // 2, SCREEN_HEIGHT // 2 + text.get_height()))
            pygame.display.flip()
        else:
            if action_made and bucket_drawn:
                # Draw bucket at chosen position
                bucket_x = position_coords[int(obs[0])]
                pygame.draw.rect(screen, BLUE, (bucket_x - BUCKET_WIDTH // 2, SCREEN_HEIGHT - BUCKET_HEIGHT - 10, BUCKET_WIDTH, BUCKET_HEIGHT))

                # Draw the bag after the delay
                if pygame.time.get_ticks() - action_timer_start >= BAG_DELAY:
                    pygame.draw.circle(screen, RED, (position_coords[int(obs[1])], SCREEN_HEIGHT - BUCKET_HEIGHT - BAG_SIZE // 2 - 10), BAG_SIZE)
                    bag_drawn = True

                # End trial after total display duration
                if pygame.time.get_ticks() - action_timer_start >= DISPLAY_DURATION:
                    waiting_for_space = True

            pygame.display.flip()

        clock.tick(FPS)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
