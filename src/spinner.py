import threading
import time


class Spinner:
    def __init__(self, total_prompts: int) -> None:
        self.current_prompt: int = 0
        self.total_prompts: int = total_prompts
        self.running: bool = False
        self.thread = None
        self.message: str = f"{self.current_prompt}/{total_prompts}"

    def _spin(self):
        frames = [
            '⠟', '⠯',
            '⠷', '⠾',
            '⠽', '⠻'
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
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def update_message(self) -> None:
        self.current_prompt += 1
        self.message = f"{self.current_prompt}/{self.total_prompts}"

    def stop(self):
        self.running = False
        self.thread.join()
        print(f"\r{self.message} Done !{' ' * 40}\r", end="", flush=True)
