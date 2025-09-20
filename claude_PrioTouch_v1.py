import pygame
import logging
import datetime
import os
import shutil
import time
import glob
import cv2
import numpy as np
import threading
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path
import contextlib

# Configuration
@dataclass
class Config:
    # Display settings
    DISPLAY_WIDTH: int = 1920
    DISPLAY_HEIGHT: int = 1080
    TARGET_WIDTH: int = 800
    TARGET_HEIGHT: int = 480
    
    # Colors
    BACKGROUND_COLOR: Tuple[int, int, int] = (255, 255, 255)
    TEXT_COLOR: Tuple[int, int, int] = (48, 127, 112)
    
    # UI settings
    FONT_SIZE: int = 30
    BUTTON_SIZE: Tuple[int, int] = (200, 60)
    
    # Paths
    LOGO_PATH: str = "assets/Priogen_logo.png"
    VIDEO_PATH: str = "assets/PriogenOpener.mp4"
    AUDIO_PATH: str = "assets/PriogenLogoSound.wav"
    BUTTON_SOUND_PATH: str = "assets/buttonClick.wav"
    ASSETS_DIR: str = "assets"
    
    # USB mount paths
    USB_MOUNT_PATTERN: str = "/media/lars/*"
    
    # Audio settings
    AUDIO_FREQUENCY: int = 44100
    AUDIO_BUFFER: int = 512

config = Config()

# Force SDL settings for fullscreen display
os.environ.update({
    "SDL_VIDEODRIVER": "x11",
    "DISPLAY": ":0",
    "SDL_MOUSE_TOUCH_EVENTS": "1",
    "SDL_VIDEO_WINDOW_POS": "0,0",
    "SDL_VIDEO_CENTERED": "0"
})

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioError(Exception):
    """Custom exception for audio-related errors"""
    pass

class VideoError(Exception):
    """Custom exception for video-related errors"""
    pass

class DisplayError(Exception):
    """Custom exception for display-related errors"""
    pass

class AudioController:
    """Optimized audio controller with better error handling and resource management"""
    
    def __init__(self):
        self.sounds = {}
        self.mixer_initialized = False
        self._lock = threading.Lock()
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize audio system with proper error handling"""
        try:
            pygame.mixer.pre_init(
                frequency=config.AUDIO_FREQUENCY,
                size=-16,
                channels=2,
                buffer=config.AUDIO_BUFFER
            )
            pygame.mixer.init()
            self.mixer_initialized = True
            logger.info("Audio mixer initialized successfully")
            self._load_all_sounds()
        except pygame.error as e:
            logger.error(f"Failed to initialize audio mixer: {e}")
            raise AudioError(f"Audio initialization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during audio initialization: {e}")
            self.mixer_initialized = False
    
    def _load_all_sounds(self) -> None:
        """Load all sound files with error handling"""
        sound_files = {
            'button_click': config.BUTTON_SOUND_PATH,
            'intro_sound': config.AUDIO_PATH
        }
        
        for sound_name, path in sound_files.items():
            try:
                if Path(path).exists():
                    self.sounds[sound_name] = pygame.mixer.Sound(path)
                    logger.info(f"Loaded sound: {sound_name}")
                else:
                    logger.warning(f"Sound file not found: {path}")
            except pygame.error as e:
                logger.warning(f"Failed to load {sound_name}: {e}")
        
        # Generate fallback sounds if needed
        self._generate_fallback_sounds()
    
    def _generate_fallback_sounds(self) -> None:
        """Generate programmatic sounds as fallbacks"""
        if not self.mixer_initialized:
            return
        
        try:
            sample_rate = config.AUDIO_FREQUENCY
            
            # Generate different tone patterns
            sound_patterns = {
                'startup': (0.3, [(800, 1200)]),  # Rising tone
                'success': (0.1, [(1000, 1000), (1200, 1200)]),  # Double beep
                'error': (0.2, [(300, 300)]),  # Low buzz
                'button_fallback': (0.05, [(800, 800)])  # Quick beep
            }
            
            for sound_name, (duration, frequencies) in sound_patterns.items():
                if sound_name.replace('_fallback', '') not in self.sounds:
                    sound = self._create_tone_sound(duration, frequencies, sample_rate)
                    if sound:
                        self.sounds[sound_name] = sound
                        
        except Exception as e:
            logger.warning(f"Failed to generate fallback sounds: {e}")
    
    def _create_tone_sound(self, duration: float, freq_pairs: List[Tuple[int, int]], 
                          sample_rate: int) -> Optional[pygame.mixer.Sound]:
        """Create a tone sound with given parameters"""
        try:
            frames = int(duration * sample_rate)
            arr = np.zeros(frames)
            
            for i, (start_freq, end_freq) in enumerate(freq_pairs):
                segment_frames = frames // len(freq_pairs)
                start_idx = i * segment_frames
                end_idx = min((i + 1) * segment_frames, frames)
                
                for j in range(start_idx, end_idx):
                    if start_freq == end_freq:
                        frequency = start_freq
                    else:
                        progress = (j - start_idx) / (end_idx - start_idx)
                        frequency = start_freq + (end_freq - start_freq) * progress
                    
                    arr[j] = np.sin(2 * np.pi * frequency * j / sample_rate) * 0.3
            
            # Convert to 16-bit stereo
            arr = (arr * 32767).astype(np.int16)
            arr = np.repeat(arr.reshape(frames, 1), 2, axis=1)
            return pygame.sndarray.make_sound(arr)
            
        except Exception as e:
            logger.warning(f"Failed to create tone sound: {e}")
            return None
    
    @contextlib.contextmanager
    def _audio_context(self):
        """Context manager for safe audio operations"""
        with self._lock:
            if not self.mixer_initialized:
                yield False
                return
            try:
                yield True
            except Exception as e:
                logger.warning(f"Audio operation failed: {e}")
    
    def play_sound(self, sound_name: str, fallback_name: Optional[str] = None) -> None:
        """Play a sound with fallback options"""
        with self._audio_context() as audio_available:
            if not audio_available:
                return
            
            sound = self.sounds.get(sound_name) or self.sounds.get(fallback_name or f"{sound_name}_fallback")
            if sound:
                try:
                    sound.play()
                    logger.debug(f"Played sound: {sound_name}")
                except pygame.error as e:
                    logger.warning(f"Failed to play sound {sound_name}: {e}")
    
    def play_button_click(self) -> None:
        """Play button click sound"""
        self.play_sound('button_click', 'button_fallback')
    
    def play_startup_sound(self) -> None:
        """Play startup sound in separate thread"""
        def play():
            self.play_sound('intro_sound', 'startup')
        
        threading.Thread(target=play, daemon=True).start()
    
    def play_success_sound(self) -> None:
        """Play success sound pattern"""
        def play():
            self.play_sound('success')
            time.sleep(0.15)
            self.play_sound('success')
        
        threading.Thread(target=play, daemon=True).start()
    
    def play_error_sound(self) -> None:
        """Play error sound"""
        self.play_sound('error')
    
    def stop_all_sounds(self) -> None:
        """Stop all playing sounds"""
        with self._audio_context() as audio_available:
            if audio_available:
                pygame.mixer.stop()
    
    def cleanup(self) -> None:
        """Clean up audio resources"""
        try:
            self.stop_all_sounds()
            if self.mixer_initialized:
                pygame.mixer.quit()
                logger.info("Audio cleanup completed")
        except Exception as e:
            logger.error(f"Audio cleanup failed: {e}")

class VideoPlayer:
    """Optimized video player with better performance and error handling"""
    
    def __init__(self, audio_controller: AudioController):
        self.audio_controller = audio_controller
        self.cap = None
        self._is_playing = False
    
    def play_video(self, screen: pygame.Surface, video_path: str) -> bool:
        """Play video with optimized performance and error handling"""
        try:
            if not Path(video_path).exists():
                logger.warning(f"Video file not found: {video_path}")
                return False
            
            # Start audio early
            self.audio_controller.play_startup_sound()
            
            # Initialize video capture with optimized settings
            self.cap = self._initialize_capture(video_path)
            if not self.cap:
                return False
            
            # Get video properties
            video_info = self._get_video_info()
            if not video_info:
                return False
            
            fps, frame_count = video_info['fps'], video_info['frame_count']
            logger.info(f"Playing video: {video_info}")
            
            # Play video with optimized frame timing
            return self._play_video_loop(screen, fps, frame_count)
            
        except Exception as e:
            logger.error(f"Video playback failed: {e}")
            return False
        finally:
            self._cleanup_video()
    
    def _initialize_capture(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """Initialize video capture with multiple backend attempts"""
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, _ = cap.read()
                    if ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                        logger.info(f"Video opened with backend: {backend}")
                        return cap
                    cap.release()
            except Exception as e:
                logger.debug(f"Backend {backend} failed: {e}")
                continue
        
        logger.error("All video backends failed")
        return None
    
    def _get_video_info(self) -> Optional[dict]:
        """Get video properties with validation"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        try:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validate and set defaults
            if fps <= 0 or fps > 120:
                fps = 24.0
                logger.warning("Invalid FPS detected, using default: 24")
            
            if frame_count <= 0:
                frame_count = fps * 10  # Assume 10 second video
                logger.warning("Invalid frame count, using estimate")
            
            return {
                'fps': fps,
                'frame_count': int(frame_count),
                'width': width,
                'height': height
            }
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None
    
    def _play_video_loop(self, screen: pygame.Surface, fps: float, frame_count: int) -> bool:
        """Optimized video playback loop"""
        frame_delay = 1000 / fps
        clock = pygame.time.Clock()
        frame_counter = 0
        screen_size = screen.get_size()
        self._is_playing = True
        
        # Pre-allocate surface for better performance
        frame_surface = None
        
        while self._is_playing and self.cap and self.cap.isOpened():
            # Handle events
            if self._check_user_input():
                logger.info("Video playback interrupted by user")
                return True
            
            # Read and process frame
            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Video completed after {frame_counter} frames")
                break
            
            frame_counter += 1
            
            try:
                # Optimize frame processing
                frame_surface = self._process_frame(frame, screen_size, frame_surface)
                if frame_surface:
                    screen.fill((0, 0, 0))
                    screen.blit(frame_surface, (0, 0))
                    pygame.display.flip()
                
            except Exception as e:
                logger.warning(f"Frame processing error: {e}")
                continue
            
            # Maintain frame rate
            clock.tick(fps)
            
            # Progress logging (less frequent)
            if frame_counter % 60 == 0:
                progress = (frame_counter / frame_count) * 100 if frame_count > 0 else 0
                logger.debug(f"Video progress: {progress:.1f}%")
        
        return True
    
    def _process_frame(self, frame: np.ndarray, screen_size: Tuple[int, int], 
                      reuse_surface: Optional[pygame.Surface] = None) -> Optional[pygame.Surface]:
        """Process video frame with optimization"""
        try:
            # Convert color space efficiently
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create or reuse surface
            if reuse_surface is None or reuse_surface.get_size() != screen_size:
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                frame_surface = pygame.transform.scale(frame_surface, screen_size)
            else:
                # Reuse existing surface for better performance
                pygame.surfarray.blit_array(reuse_surface, frame_rgb.swapaxes(0, 1))
                frame_surface = pygame.transform.scale(reuse_surface, screen_size)
            
            return frame_surface
            
        except Exception as e:
            logger.warning(f"Frame processing failed: {e}")
            return None
    
    def _check_user_input(self) -> bool:
        """Check for user input to skip video"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._is_playing = False
                return True
            elif event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return True
        return False
    
    def _cleanup_video(self) -> None:
        """Clean up video resources"""
        self._is_playing = False
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                logger.warning(f"Video cleanup error: {e}")
        
        # Stop audio
        self.audio_controller.stop_all_sounds()

class TouchButton:
    """Optimized touch button class"""
    
    def __init__(self, text: str, center: Tuple[int, int], 
                 size: Tuple[int, int] = None, width_override: Optional[int] = None):
        self.text = text
        size = size or config.BUTTON_SIZE
        w, h = size
        if width_override:
            w = width_override
        
        self.rect = pygame.Rect(0, 0, w, h)
        self.rect.center = center
        
        # Pre-calculate colors for better performance
        self.colors = {
            'shadow_dark': (245, 245, 245),
            'shadow_light': (180, 180, 180),
            'button': (230, 240, 250),
            'text': (0, 0, 0)
        }
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw button with optimized rendering"""
        # Draw shadow effects
        pygame.draw.rect(screen, self.colors['shadow_dark'], 
                        self.rect.move(-2, -2), border_radius=10)
        pygame.draw.rect(screen, self.colors['shadow_light'], 
                        self.rect.move(2, 2), border_radius=10)
        
        # Draw main button
        pygame.draw.rect(screen, self.colors['button'], self.rect, border_radius=10)
        
        # Draw text (cache rendered text for repeated draws)
        if not hasattr(self, '_cached_text') or self._cached_text[0] != self.text:
            self._cached_text = (self.text, font.render(self.text, True, self.colors['text']))
        
        txt_surface = self._cached_text[1]
        txt_rect = txt_surface.get_rect(center=self.rect.center)
        screen.blit(txt_surface, txt_rect)
    
    def is_pressed(self, pos: Tuple[int, int]) -> bool:
        """Check if button is pressed at given position"""
        return self.rect.collidepoint(pos)

class FileManager:
    """Optimized file management with better error handling"""
    
    @staticmethod
    def ensure_assets_dir() -> None:
        """Ensure assets directory exists"""
        try:
            Path(config.ASSETS_DIR).mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create assets directory: {e}")
            raise
    
    @staticmethod
    def find_csv_files() -> List[str]:
        """Find CSV files on USB drives with better error handling"""
        csv_files = []
        
        try:
            mount_bases = glob.glob(config.USB_MOUNT_PATTERN)
            logger.info(f"Checking {len(mount_bases)} USB mount points")
            
            for base in mount_bases:
                try:
                    base_path = Path(base)
                    if not base_path.exists() or not base_path.is_dir():
                        continue
                    
                    logger.info(f"Scanning USB mount: {base}")
                    
                    # Use pathlib for more efficient file searching
                    for csv_file in base_path.rglob("*.csv"):
                        if csv_file.is_file():
                            csv_files.append(str(csv_file))
                            logger.info(f"Found CSV: {csv_file}")
                            
                except PermissionError:
                    logger.warning(f"Permission denied accessing: {base}")
                except Exception as e:
                    logger.warning(f"Error scanning {base}: {e}")
        
        except Exception as e:
            logger.error(f"Error finding CSV files: {e}")
        
        return csv_files
    
    @staticmethod
    def copy_file_safely(source: str, destination_dir: str) -> bool:
        """Copy file with comprehensive error handling"""
        try:
            source_path = Path(source)
            dest_dir = Path(destination_dir)
            
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source}")
                return False
            
            if not source_path.is_file():
                logger.error(f"Source is not a file: {source}")
                return False
            
            # Create destination directory if needed
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            destination = dest_dir / source_path.name
            
            # Check available space (basic check)
            try:
                source_size = source_path.stat().st_size
                dest_stat = shutil.disk_usage(dest_dir)
                if dest_stat.free < source_size * 2:  # Safety margin
                    logger.error("Insufficient disk space for file copy")
                    return False
            except Exception as e:
                logger.warning(f"Could not check disk space: {e}")
            
            # Perform the copy
            shutil.copy2(source, destination)
            
            # Verify copy
            if destination.exists() and destination.stat().st_size == source_size:
                logger.info(f"Successfully copied {source} to {destination}")
                return True
            else:
                logger.error("File copy verification failed")
                return False
                
        except PermissionError as e:
            logger.error(f"Permission denied copying file: {e}")
            return False
        except shutil.Error as e:
            logger.error(f"File copy error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error copying file: {e}")
            return False

class DisplayManager:
    """Optimized display management with robust initialization"""
    
    def __init__(self):
        self.screen = None
        self.virtual_screen = None
        self.use_scaling = False
        self.scale_x = 1.0
        self.scale_y = 1.0
    
    def initialize_display(self) -> bool:
        """Initialize display with multiple fallback methods"""
        try:
            pygame.init()
            pygame.display.init()
            
            # Log display info
            self._log_display_info()
            
            # Try different display modes
            if self._try_display_modes():
                self._configure_display()
                self._setup_scaling()
                return True
            else:
                raise DisplayError("All display initialization methods failed")
                
        except Exception as e:
            logger.error(f"Display initialization failed: {e}")
            return False
    
    def _log_display_info(self) -> None:
        """Log display information for debugging"""
        try:
            driver = pygame.display.get_driver()
            logger.info(f"SDL video driver: {driver}")
        except pygame.error:
            logger.warning("Could not get video driver info")
        
        try:
            info = pygame.display.Info()
            logger.info(f"Display info: {info.current_w}x{info.current_h}")
        except pygame.error:
            logger.warning("Could not get display info")
    
    def _try_display_modes(self) -> bool:
        """Try different display modes in order of preference"""
        display_attempts = [
            ("Auto fullscreen", lambda: pygame.display.set_mode((0, 0), pygame.FULLSCREEN)),
            ("Target fullscreen", lambda: pygame.display.set_mode(
                (config.TARGET_WIDTH, config.TARGET_HEIGHT), pygame.FULLSCREEN)),
            ("Target windowed", lambda: pygame.display.set_mode(
                (config.TARGET_WIDTH, config.TARGET_HEIGHT))),
            ("Minimal window", lambda: pygame.display.set_mode((320, 240)))
        ]
        
        for mode_name, mode_func in display_attempts:
            try:
                logger.info(f"Attempting {mode_name}")
                self.screen = mode_func()
                logger.info(f"{mode_name} successful: {self.screen.get_size()}")
                return True
            except Exception as e:
                logger.warning(f"{mode_name} failed: {e}")
                continue
        
        return False
    
    def _configure_display(self) -> None:
        """Configure display settings"""
        pygame.mouse.set_visible(False)
        pygame.display.set_caption("Priogen Interface")
        self.screen.fill((0, 0, 0))
        pygame.display.flip()
    
    def _setup_scaling(self) -> None:
        """Setup scaling if needed"""
        actual_size = self.screen.get_size()
        target_size = (config.TARGET_WIDTH, config.TARGET_HEIGHT)
        
        if actual_size != target_size:
            logger.info(f"Setting up scaling: {actual_size} -> {target_size}")
            self.use_scaling = True
            self.scale_x = actual_size[0] / target_size[0]
            self.scale_y = actual_size[1] / target_size[1]
            self.virtual_screen = pygame.Surface(target_size)
        else:
            self.use_scaling = False
            self.virtual_screen = self.screen
    
    def get_scaled_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Convert screen coordinates to virtual coordinates"""
        if self.use_scaling:
            return (int(pos[0] / self.scale_x), int(pos[1] / self.scale_y))
        return pos
    
    def update_display(self) -> None:
        """Update display with scaling if needed"""
        if self.use_scaling:
            scaled_surface = pygame.transform.scale(self.virtual_screen, self.screen.get_size())
            self.screen.blit(scaled_surface, (0, 0))
        pygame.display.update()

class PriogenInterface:
    """Main application class with optimized structure"""
    
    def __init__(self):
        self.audio_controller = AudioController()
        self.video_player = VideoPlayer(self.audio_controller)
        self.display_manager = DisplayManager()
        self.file_manager = FileManager()
        
        # UI state
        self.state = "home"
        self.clock = pygame.time.Clock()
        self.font = None
        self.icon_font = None
        
        # Application data
        self.csv_file_list = []
        self.imported_filename = None
        self.show_start_button = False
        self.run_start_time = None
        
        # UI elements
        self.buttons = {}
        self.file_buttons = []
    
    def initialize(self) -> bool:
        """Initialize the application"""
        try:
            # Initialize display
            if not self.display_manager.initialize_display():
                return False
            
            # Initialize fonts
            self.font = pygame.font.SysFont("Arial", config.FONT_SIZE)
            self.icon_font = pygame.font.SysFont("Arial", 35)
            
            # Create UI buttons
            self._create_buttons()
            
            # Ensure assets directory exists
            self.file_manager.ensure_assets_dir()
            
            # Play intro video
            self.video_player.play_video(
                self.display_manager.virtual_screen,
                config.VIDEO_PATH
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            return False
    
    def _create_buttons(self) -> None:
        """Create all UI buttons"""
        w, h = config.TARGET_WIDTH, config.TARGET_HEIGHT
        
        self.buttons = {
            'custom': TouchButton("Custom Run", (w // 2, h // 2 + 130)),
            'nano': TouchButton("Nano-QuIC", (w // 4, h // 2 + 40)),
            'rt': TouchButton("RT-QuIC", (3 * w // 4, h // 2 + 40)),
            'back': TouchButton("←", (70, 40), (50, 50)),
            'start_nano': TouchButton("Start Nano-QuIC", (w // 2, h - 60), (300, 60)),
            'prp_nano': TouchButton("PrP", (w * 0.3, h // 2)),
            'alpha_nano': TouchButton("Alpha-Syn", (w * 0.7, h // 2)),
            'prp_rt': TouchButton("PrP", (w * 0.2, h // 2)),
            'alpha_rt': TouchButton("Alpha-Syn", (w * 0.5, h // 2)),
            'tdp43_rt': TouchButton("TDP-43", (w * 0.8, h // 2))
        }
    
    def run(self) -> None:
        """Main application loop"""
        running = True
        
        try:
            while running:
                # Clear screen
                self.display_manager.virtual_screen.fill(config.BACKGROUND_COLOR)
                
                # Render current state
                self._render_current_state()
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self._handle_mouse_click(event.pos)
                
                # Update display
                self.display_manager.update_display()
                self.clock.tick(30)
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self._cleanup()
    
    def _render_current_state(self) -> None:
        """Render the current application state"""
        screen = self.display_manager.virtual_screen
        
        if self.state == "home":
            self._render_home(screen)
        elif self.state == "nano":
            self._render_nano_selection(screen)
        elif self.state == "rt":
            self._render_rt_selection(screen)
        elif self.state == "prp_import":
            self._render_file_import(screen)
        elif self.state == "nano_run":
            self._render_nano_run(screen)
    
    def _render_home(self, screen: pygame.Surface) -> None:
        """Render home screen"""
        self._draw_logo(screen)
        self._draw_datetime_bar(screen)
        self.buttons['custom'].draw(screen, self.font)
        self.buttons['nano'].draw(screen, self.font)
        self.buttons['rt'].draw(screen, self.font)
    
    def _render_nano_selection(self, screen: pygame.Surface) -> None:
        """Render Nano-QuIC substrate selection"""
        self._draw_title(screen, "Substrate Selection")
        self.buttons['prp_nano'].draw(screen, self.font)
        self.buttons['alpha_nano'].draw(screen, self.font)
        self.buttons['back'].draw(screen, self.icon_font)
    
    def _render_rt_selection(self, screen: pygame.Surface) -> None:
        """Render RT-QuIC substrate selection"""
        self._draw_title(screen, "Substrate Selection")
        self.buttons['prp_rt'].draw(screen, self.font)
        self.buttons['alpha_rt'].draw(screen, self.font)
        self.buttons['tdp43_rt'].draw(screen, self.font)
        self.buttons['back'].draw(screen, self.icon_font)
    
    def _render_file_import(self, screen: pygame.Surface) -> None:
        """Render file import screen"""
        self._draw_title(screen, "Import Nano-QuIC PrP Run Information")
        self.buttons['back'].draw(screen, self.icon_font)
        
        for btn in self.file_buttons:
            btn.draw(screen, self.font)
        
        if not self.file_buttons and not self.imported_filename:
            msg = self.font.render("No .csv files found on USB.", True, (150, 0, 0))
            msg_rect = msg.get_rect(center=(config.TARGET_WIDTH // 2, config.TARGET_HEIGHT // 2))
            screen.blit(msg, msg_rect)
        elif self.imported_filename:
            status = f"{self.imported_filename} successfully imported"
            msg = self.font.render(status, True, (0, 150, 0))
            msg_rect = msg.get_rect(center=(config.TARGET_WIDTH // 2, config.TARGET_HEIGHT // 2))
            screen.blit(msg, msg_rect)
            if self.show_start_button:
                self.buttons['start_nano'].draw(screen, self.font)
    
    def _render_nano_run(self, screen: pygame.Surface) -> None:
        """Render active Nano-QuIC run screen"""
        self._draw_logo(screen)
        self._draw_datetime_bar(screen)
        if self.run_start_time:
            label = self.font.render("Nano-QuIC Run Started At", True, config.TEXT_COLOR)
            elapsed_time = self.font.render(self._format_elapsed(self.run_start_time), True, config.TEXT_COLOR)
            
            label_rect = label.get_rect(center=(config.TARGET_WIDTH // 2, config.TARGET_HEIGHT // 2 - 20))
            time_rect = elapsed_time.get_rect(center=(config.TARGET_WIDTH // 2, config.TARGET_HEIGHT // 2 + 20))
            
            screen.blit(label, label_rect)
            screen.blit(elapsed_time, time_rect)
    
    def _draw_logo(self, screen: pygame.Surface) -> None:
        """Draw logo with error handling"""
        try:
            if Path(config.LOGO_PATH).exists():
                logo = pygame.image.load(config.LOGO_PATH).convert_alpha()
                logo_rect = logo.get_rect()
                
                # Calculate scale factor
                max_width = config.TARGET_WIDTH - 40
                max_height = 200
                scale_factor = min(max_width / logo_rect.width, max_height / logo_rect.height)
                
                # Scale logo
                new_size = (int(logo_rect.width * scale_factor), int(logo_rect.height * scale_factor))
                logo_scaled = pygame.transform.smoothscale(logo, new_size)
                
                # Center logo
                logo_rect = logo_scaled.get_rect()
                logo_rect.centerx = config.TARGET_WIDTH // 2
                logo_rect.y = 20
                
                screen.blit(logo_scaled, logo_rect)
            else:
                # Fallback text if logo not found
                text = self.font.render("PRIOGEN", True, config.TEXT_COLOR)
                text_rect = text.get_rect(center=(config.TARGET_WIDTH // 2, 100))
                screen.blit(text, text_rect)
                
        except Exception as e:
            logger.warning(f"Failed to load logo: {e}")
            # Fallback text
            text = self.font.render("PRIOGEN", True, config.TEXT_COLOR)
            text_rect = text.get_rect(center=(config.TARGET_WIDTH // 2, 100))
            screen.blit(text, text_rect)
    
    def _draw_datetime_bar(self, screen: pygame.Surface) -> None:
        """Draw date and time bar"""
        try:
            now = datetime.datetime.now()
            time_text = now.strftime("%I:%M:%S %p")
            date_text = now.strftime("%d %b %Y")
            
            time_surf = self.font.render(time_text, True, config.TEXT_COLOR)
            date_surf = self.font.render(date_text, True, config.TEXT_COLOR)
            
            # Position at bottom corners
            time_pos = (10, config.TARGET_HEIGHT - time_surf.get_height() - 10)
            date_pos = (config.TARGET_WIDTH - date_surf.get_width() - 10, 
                       config.TARGET_HEIGHT - date_surf.get_height() - 10)
            
            screen.blit(time_surf, time_pos)
            screen.blit(date_surf, date_pos)
            
        except Exception as e:
            logger.warning(f"Failed to draw datetime bar: {e}")
    
    def _draw_title(self, screen: pygame.Surface, text: str) -> None:
        """Draw title with shadow effect"""
        try:
            # Create shadow and main text
            shadow_surface = self.font.render(text, True, (0, 0, 0))
            title_surface = self.font.render(text, True, config.TEXT_COLOR)
            
            # Position with shadow offset
            center_x = config.TARGET_WIDTH // 2
            shadow_rect = shadow_surface.get_rect(center=(center_x + 2, 42))
            title_rect = title_surface.get_rect(center=(center_x, 40))
            
            screen.blit(shadow_surface, shadow_rect)
            screen.blit(title_surface, title_rect)
            
        except Exception as e:
            logger.warning(f"Failed to draw title: {e}")
    
    def _format_elapsed(self, start_time: datetime.datetime) -> str:
        """Format elapsed time as MM:SS"""
        try:
            elapsed = datetime.datetime.now() - start_time
            total_seconds = int(elapsed.total_seconds())
            minutes, seconds = divmod(total_seconds, 60)
            return f"{minutes:02}:{seconds:02}"
        except Exception as e:
            logger.warning(f"Failed to format elapsed time: {e}")
            return "00:00"
    
    def _handle_mouse_click(self, pos: Tuple[int, int]) -> None:
        """Handle mouse click events with proper coordinate scaling"""
        try:
            # Play click sound
            self.audio_controller.play_button_click()
            
            # Scale position if needed
            scaled_pos = self.display_manager.get_scaled_pos(pos)
            
            # Handle click based on current state
            if self.state == "home":
                self._handle_home_click(scaled_pos)
            elif self.state == "nano":
                self._handle_nano_click(scaled_pos)
            elif self.state == "rt":
                self._handle_rt_click(scaled_pos)
            elif self.state == "prp_import":
                self._handle_import_click(scaled_pos)
                
        except Exception as e:
            logger.error(f"Error handling mouse click: {e}")
    
    def _handle_home_click(self, pos: Tuple[int, int]) -> None:
        """Handle clicks on home screen"""
        if self.buttons['nano'].is_pressed(pos):
            self.state = "nano"
        elif self.buttons['rt'].is_pressed(pos):
            self.state = "rt"
        elif self.buttons['custom'].is_pressed(pos):
            logger.info("Custom Run button pressed")
    
    def _handle_nano_click(self, pos: Tuple[int, int]) -> None:
        """Handle clicks on Nano-QuIC screen"""
        if self.buttons['back'].is_pressed(pos):
            self.state = "home"
        elif self.buttons['prp_nano'].is_pressed(pos):
            self._start_file_import()
    
    def _handle_rt_click(self, pos: Tuple[int, int]) -> None:
        """Handle clicks on RT-QuIC screen"""
        if self.buttons['back'].is_pressed(pos):
            self.state = "home"
        # Add handlers for RT-QuIC substrates as needed
    
    def _handle_import_click(self, pos: Tuple[int, int]) -> None:
        """Handle clicks on import screen"""
        if self.buttons['back'].is_pressed(pos):
            self.state = "nano"
            self._reset_import_state()
        elif self.show_start_button and self.buttons['start_nano'].is_pressed(pos):
            self._start_nano_run()
        else:
            # Check file buttons
            for btn in self.file_buttons:
                if btn.is_pressed(pos):
                    self._import_selected_file(btn.text)
                    break
    
    def _start_file_import(self) -> None:
        """Initialize file import process"""
        try:
            self.state = "prp_import"
            self.csv_file_list = self.file_manager.find_csv_files()
            self._reset_import_state()
            
            # Create file buttons
            self.file_buttons = [
                TouchButton(
                    Path(path).name, 
                    (config.TARGET_WIDTH // 2, 100 + i * 50), 
                    width_override=500
                )
                for i, path in enumerate(self.csv_file_list[:8])  # Limit to 8 files for UI
            ]
            
        except Exception as e:
            logger.error(f"Failed to start file import: {e}")
            self.audio_controller.play_error_sound()
    
    def _import_selected_file(self, filename: str) -> None:
        """Import selected CSV file"""
        try:
            # Find full path for the selected file
            selected_path = None
            for path in self.csv_file_list:
                if Path(path).name == filename:
                    selected_path = path
                    break
            
            if not selected_path:
                logger.error(f"Selected file not found: {filename}")
                self.audio_controller.play_error_sound()
                return
            
            # Copy file to assets directory
            if self.file_manager.copy_file_safely(selected_path, config.ASSETS_DIR):
                self.imported_filename = filename
                self.show_start_button = True
                self.file_buttons = []
                logger.info(f"Successfully imported {filename}")
                self.audio_controller.play_success_sound()
            else:
                logger.error(f"Failed to import {filename}")
                self.audio_controller.play_error_sound()
                
        except Exception as e:
            logger.error(f"Error importing file {filename}: {e}")
            self.audio_controller.play_error_sound()
    
    def _start_nano_run(self) -> None:
        """Start Nano-QuIC run"""
        try:
            self.run_start_time = datetime.datetime.now()
            self.state = "nano_run"
            self.audio_controller.play_success_sound()
            logger.info("Nano-QuIC run started")
            
        except Exception as e:
            logger.error(f"Failed to start Nano-QuIC run: {e}")
            self.audio_controller.play_error_sound()
    
    def _reset_import_state(self) -> None:
        """Reset import-related state variables"""
        self.imported_filename = None
        self.show_start_button = False
        self.file_buttons = []
    
    def _cleanup(self) -> None:
        """Clean up resources before exit"""
        try:
            logger.info("Cleaning up application resources")
            self.audio_controller.cleanup()
            pygame.quit()
            logger.info("Application cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main() -> int:
    """Main application entry point"""
    try:
        # Create and initialize application
        app = PriogenInterface()
        
        if not app.initialize():
            logger.error("Failed to initialize application")
            return 1
        
        logger.info("Application initialized successfully")
        
        # Run application
        app.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
