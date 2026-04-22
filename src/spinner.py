import time
import threading

from pydantic import BaseModel, PrivateAttr

CYAN = "\033[36m"
RESET = "\033[0m"


class Spinner(BaseModel):
    current_prompt: int = -1
    total_prompts: int = 0
    running: bool = False
    message: str = "-1/0"
    generated: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    i: int = 0
    debug_gap: bool = False

    _thread: threading.Thread = PrivateAttr()
    _lock: threading.Lock = PrivateAttr()

    def model_post_init(self, _: object) -> None:
        """Initialise the worker thread and its output lock."""
        self._thread = threading.Thread(target=self._spin)
        self._lock = threading.Lock()

    def print_above(self, text: str) -> None:
        """Print text on the line above the spinner without disturbing it.

        Args:
            text: The text to print above the spinner line.
        """
        with self._lock:
            print(f"\r{' ' * 60}\r", end="", flush=True)
            if self.debug_gap:
                print(f"\033[A\r{' ' * 60}\r", end="", flush=True)
            print(text)
            print()
            self.debug_gap = True
        self._print_msg()

    def _print_msg(self) -> None:
        """Render the current spinner frame, elapsed time, and message."""
        frames = [
            '⠻', '⠽',
            '⠾', '⠷',
            '⠯', '⠟',
        ]

        now = time.time()

        sec = now - self.start_time

        loading = f"{CYAN}{frames[self.i % 6]}{RESET}"

        with self._lock:
            print(
                f"\r({sec:.2f}s) {self.message} {loading}",
                end="",
                flush=True
            )

    def _spin(self) -> None:
        """Run the spinner animation loop until `running` becomes False."""
        self.i = 0
        while self.running:
            self._print_msg()
            self.i += 1
            time.sleep(0.10)

    def start(self) -> None:
        """Start the spinner thread and record the start time."""
        self.running = True
        self.start_time = time.time()
        self._thread.start()

    def update_message(self) -> None:
        """Advance the progress counter and refresh the displayed message."""
        self.current_prompt += 1
        self.message = f"{self.current_prompt}/{self.total_prompts}"

    def stop(self) -> None:
        """Stop the spinner, join its thread, and print a closing line."""
        if not self.running:
            return

        self.running = False
        self.end_time = time.time()
        self._thread.join()
        generation_time = self.end_time - self.start_time

        if self.generated:
            print(
                f"\r({generation_time:.2f}s) " +
                f"{self.message} Done !",
                end="",
                flush=True
            )

        else:
            print(
                f"\r({generation_time:.2f}s) {self.message} "
                "Couldn't finish generating.",
                end="",
                flush=True)

        print()
