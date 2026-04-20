import threading
import time


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

    def _spin(self):
        frames = [
            '⠻', '⠽',
            '⠾', '⠷',
            '⠯', '⠟',
        ]

        i = 0
        while self.running:
            print(
                f"\r{self.message} \033[36m{frames[i % 6]}\033[0m",
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

        if self.generated:
            print(
                f"\r{self.message} Done !{' ' * 10}\r",
                end="",
                flush=True
            )

            generation_time = self.end_time - self.start_time

            print(f"Took {generation_time:.3f} to generate answers.")

        else:
            print(
                f"\r{self.message} Couldn't finish generating.{' ' * 10}\r",
                end="",
                flush=True
            )

        print()
