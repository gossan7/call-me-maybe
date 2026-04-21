import time
import threading

RED = "\033[31m"
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"


class Spinner:
    def __init__(self) -> None:
        self.current_prompt: int = -1
        self.total_prompts: int = 0
        self.running: bool = False
        self.thread = threading.Thread(target=self._spin)
        self.message: str = f"{self.current_prompt}/{self.total_prompts}"
        self.generated = False
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.lock = threading.Lock()
        self.i = 0
        self.debug_gap = False

    def print_above(self, text: str) -> None:
        """Print text above the spinner line."""
        with self.lock:
            print(f"\r{' ' * 60}\r", end="", flush=True)
            if self.debug_gap:
                print(f"\033[A\r{' ' * 60}\r", end="", flush=True)
            print(text)
            print()
            self.debug_gap = True
        self._print_msg()

    def _print_msg(self) -> None:
        frames = [
            '⠻', '⠽',
            '⠾', '⠷',
            '⠯', '⠟',
        ]

        now = time.time()

        sec = now - self.start_time

        loading = f"{CYAN}{frames[self.i % 6]}{RESET}"

        with self.lock:
            print(
                f"\r({sec:.2f}s) {self.message} {loading}",
                end="",
                flush=True
            )

    def _spin(self) -> None:

        self.i = 0
        while self.running:
            self._print_msg()
            self.i += 1
            time.sleep(0.10)

    def start(self) -> None:
        self.running = True
        self.start_time = time.time()
        self.thread.start()

    def update_message(self) -> None:
        self.current_prompt += 1
        self.message = f"{self.current_prompt}/{self.total_prompts}"

    def stop(self) -> None:
        if not self.running:
            return

        self.running = False
        self.end_time = time.time()
        self.thread.join()
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
