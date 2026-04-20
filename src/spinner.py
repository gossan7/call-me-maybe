import time
import threading


class Spinner:
    def __init__(self) -> None:
        self.current_prompt: int = 0
        self.total_prompts: int = 0
        self.running: bool = False
        self.thread = None
        self.message: str = f"{self.current_prompt}/{self.total_prompts}"
        self.generated = False
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.lock = threading.Lock()
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

    def _spin(self):
        frames = [
            '⠻', '⠽',
            '⠾', '⠷',
            '⠯', '⠟',
        ]

        i = 0
        while self.running:
            now = time.time()

            sec = now - self.start_time

            loading = f"\033[36m{frames[i % 6]}\033[0m"

            with self.lock:
                print(
                    f"\r({sec:.2f}s) {self.message} {loading}",
                    end="",
                    flush=True
                )

            i += 1
            time.sleep(0.10)

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def update_message(self) -> None:
        self.current_prompt += 1
        self.message = f"{self.current_prompt}/{self.total_prompts}"

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.end_time = time.time()
        self.thread.join()
        generation_time = self.end_time - self.start_time

        if self.generated:

            print(
                f"\r({generation_time:.2f}s) {self.message} Done !\r",
                end="",
                flush=True
            )

        else:
            print(
                f"\r({generation_time:.2f}s) {self.message} "
                f"Couldn't finish generating. {' ' * 10}\r",
                end="",
                flush=True)

        print()
