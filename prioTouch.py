import pygame
import logging
import datetime
import os
import shutil
import time
import glob
from moviepy.editor import VideoFileClip

os.environ["SDL_MOUSE_TOUCH_EVENTS"] = "1"

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 480
BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (48, 127, 112)
FONT_SIZE = 30
LOGO_PATH = "assets/Priogen_logo.png"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TouchButton:
    def __init__(self, text, center, size=(200, 60), width_override=None):
        self.text = text
        w, h = size
        if width_override:
            w = width_override
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.center = center
        self.color = (230, 240, 250)
        self.text_color = (0, 0, 0)

    def draw(self, screen, font):
        pygame.draw.rect(screen, (245, 245, 245), self.rect.move(-2, -2), border_radius=10)
        pygame.draw.rect(screen, (180, 180, 180), self.rect.move(2, 2), border_radius=10)
        pygame.draw.rect(screen, self.color, self.rect, border_radius=10)
        txt_surface = font.render(self.text, True, self.text_color)
        txt_rect = txt_surface.get_rect(center=self.rect.center)
        screen.blit(txt_surface, txt_rect)

    def is_pressed(self, pos):
        return self.rect.collidepoint(pos)

def draw_logo(screen):
    try:
        logo = pygame.image.load(LOGO_PATH).convert_alpha()
        logo_rect = logo.get_rect()
        scale_factor = min(SCREEN_WIDTH / logo_rect.width, 200 / logo_rect.height)
        logo_scaled = pygame.transform.smoothscale(logo, (int(logo_rect.width * scale_factor), int(logo_rect.height * scale_factor)))
        logo_rect = logo_scaled.get_rect()
        logo_rect.topleft = ((SCREEN_WIDTH - logo_rect.width) // 2, 20)
        screen.blit(logo_scaled, logo_rect)
    except Exception as e:
        logger.error(f"Failed to load logo image: {e}")

def draw_datetime_bar(screen, font):
    now = datetime.datetime.now()
    time_surf = font.render(now.strftime("%I:%M:%S %p"), True, TEXT_COLOR)
    date_surf = font.render(now.strftime("%d %b %Y"), True, TEXT_COLOR)
    screen.blit(time_surf, (10, SCREEN_HEIGHT - time_surf.get_height() - 10))
    screen.blit(date_surf, (SCREEN_WIDTH - date_surf.get_width() - 10, SCREEN_HEIGHT - date_surf.get_height() - 10))

def draw_title(screen, text, font):
    shadow_surface = font.render(text, True, (0, 0, 0))
    title_surface = font.render(text, True, TEXT_COLOR)
    shadow_rect = shadow_surface.get_rect(center=(SCREEN_WIDTH // 2 + 2, 42))
    title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, 40))
    screen.blit(shadow_surface, shadow_rect)
    screen.blit(title_surface, title_rect)

def play_intro_video(screen: pygame.Surface, video_path: str = "assets/PriogenOpener.mp4") -> None:
    try:
        logger.info(f"Attempting to play video: {video_path}")
        clip = VideoFileClip(video_path)
        for frame in clip.iter_frames(fps=24, dtype="uint8"):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Playback interrupted by user.")
                    return
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            frame_surface = pygame.transform.smoothscale(frame_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
            rect = frame_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.fill((0, 0, 0))
            screen.blit(frame_surface, rect)
            pygame.display.update()
        clip.close()
        time.sleep(0.2)
        logger.info("Video playback complete.")
    except Exception as e:
        logger.error(f"Failed to play video: {e}")

def find_csv_files():
    csv_files = []
    mount_bases = glob.glob("/media/lars/*")
    for base in mount_bases:
        logger.info(f"Checking USB mount: {base}")
        for root, _, files in os.walk(base):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    logger.info(f"Found CSV: {full_path}")
                    csv_files.append(full_path)
    return csv_files

def format_elapsed(start_time):
    elapsed = datetime.datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02}:{seconds:02}"

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.mouse.set_visible(True)
    play_intro_video(screen)
    font = pygame.font.SysFont("Arial", FONT_SIZE)
    icon_font = pygame.font.SysFont("Arial", 35)
    clock = pygame.time.Clock()

    custom_button = TouchButton("Custom Run", (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 130))
    nano_button = TouchButton("Nano-QuIC", (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 + 40))
    rt_button = TouchButton("RT-QuIC", (3 * SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 + 40))
    back_arrow = TouchButton("←", (70, 40), (50, 50))
    start_nano_button = TouchButton("Start Nano-QuIC", (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 60), (300, 60))

    state = "home"
    file_buttons = []
    csv_file_list = []
    imported_filename = None
    show_start_button = False
    run_start_time = None

    while True:
        screen.fill(BACKGROUND_COLOR)

        if state == "home":
            draw_logo(screen)
            draw_datetime_bar(screen, font)
            custom_button.draw(screen, font)
            nano_button.draw(screen, font)
            rt_button.draw(screen, font)

        elif state == "nano":
            draw_title(screen, "Substrate Selection", font)
            TouchButton("PrP", (SCREEN_WIDTH * 0.3, SCREEN_HEIGHT // 2)).draw(screen, font)
            TouchButton("Alpha-Syn", (SCREEN_WIDTH * 0.7, SCREEN_HEIGHT // 2)).draw(screen, font)
            back_arrow.draw(screen, icon_font)

        elif state == "rt":
            draw_title(screen, "Substrate Selection", font)
            TouchButton("PrP", (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT // 2)).draw(screen, font)
            TouchButton("Alpha-Syn", (SCREEN_WIDTH * 0.5, SCREEN_HEIGHT // 2)).draw(screen, font)
            TouchButton("TDP-43", (SCREEN_WIDTH * 0.8, SCREEN_HEIGHT // 2)).draw(screen, font)
            back_arrow.draw(screen, icon_font)

        elif state == "prp_import":
            draw_title(screen, "Import Nano-QuIC PrP Run Information", font)
            back_arrow.draw(screen, icon_font)

            for btn in file_buttons:
                btn.draw(screen, font)

            if not file_buttons and not imported_filename:
                msg = font.render("No .csv files found on USB.", True, (150, 0, 0))
                screen.blit(msg, msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
            elif imported_filename:
                status = f"{imported_filename} successfully imported"
                msg = font.render(status, True, (0, 150, 0))
                screen.blit(msg, msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
                if show_start_button:
                    start_nano_button.draw(screen, font)

        elif state == "nano_run":
            draw_logo(screen)
            draw_datetime_bar(screen, font)
            if run_start_time:
                label = font.render("Nano-QuIC Run Started At", True, TEXT_COLOR)
                elapsed_time = font.render(format_elapsed(run_start_time), True, TEXT_COLOR)
                screen.blit(label, label.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)))
                screen.blit(elapsed_time, elapsed_time.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20)))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if state == "home":
                    if nano_button.is_pressed(pos):
                        state = "nano"
                    elif rt_button.is_pressed(pos):
                        state = "rt"
                    elif custom_button.is_pressed(pos):
                        print("Custom Run button pressed")

                elif state == "nano":
                    if back_arrow.is_pressed(pos):
                        state = "home"
                    elif TouchButton("PrP", (SCREEN_WIDTH * 0.3, SCREEN_HEIGHT // 2)).is_pressed(pos):
                        state = "prp_import"
                        csv_file_list = find_csv_files()
                        imported_filename = None
                        show_start_button = False
                        file_buttons = [
                            TouchButton(os.path.basename(path), (SCREEN_WIDTH // 2, 100 + i * 50), width_override=500)
                            for i, path in enumerate(csv_file_list)
                        ]

                elif state == "rt":
                    if back_arrow.is_pressed(pos):
                        state = "home"

                elif state == "prp_import":
                    if back_arrow.is_pressed(pos):
                        state = "nano"
                        file_buttons = []
                        imported_filename = None
                        show_start_button = False
                    elif show_start_button and start_nano_button.is_pressed(pos):
                        run_start_time = datetime.datetime.now()
                        state = "nano_run"
                    for btn in file_buttons:
                        if btn.is_pressed(pos):
                            selected = next((f for f in csv_file_list if os.path.basename(f) == btn.text), None)
                            if selected:
                                try:
                                    shutil.copy(selected, "assets/")
                                    imported_filename = os.path.basename(selected)
                                    show_start_button = True
                                    file_buttons = []
                                    print(f"Copied {selected} to assets/")
                                except Exception as e:
                                    print(f"Failed to copy file: {e}")

        clock.tick(30)

if __name__ == "__main__":
    main()
