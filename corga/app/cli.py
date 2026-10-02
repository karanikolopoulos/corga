import logging

from enum import StrEnum
from collections.abc import Callable

from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown

from corga.agent import ChatSession

HELP = """/new    close this conversation (trace saved) and start a fresh one
/state  show what the assistant remembers: time window, preferences, current itinerary
/trace  show the path of this conversation's trace file
/help   show this help
/exit   close the conversation and quit (Ctrl+D also works)"""
EXIT = {"/exit", "/quit"}
NOT_TRACED = "not traced (hydra=debug)"

log = logging.getLogger(__name__)
console = Console(highlight=False)


class Speaker(StrEnum):
    USER = "🙋"
    ASSISTANT = "🤖"


def said_by(speaker: Speaker, content: Markdown) -> Table:
    """Lay out a message: the speaker's emoji, then the message, its wrapped lines aligned after the emoji."""
    row = Table.grid(padding=(0, 1))
    row.add_column(width=2)
    row.add_column()
    row.add_row(speaker.value, content)

    return row


class Repl:
    def __init__(self, new_session: Callable[[], ChatSession]):
        self.new_session = new_session
        self.session = self.start()
        self.commands: dict[str, Callable[[], None]] = {
            "/help": lambda: console.print(HELP),
            "/state": lambda: console.print(self.session.summary()),
            "/trace": lambda: console.print(self.session.trace_path or NOT_TRACED),
            "/new": self.new,
        }

    def run(self) -> None:
        try:
            while (text := self.next_line()) is not None and text not in EXIT:
                if command := self.commands.get(text):
                    command()
                elif text.startswith("/"):
                    console.print(f"unknown command {text}\n{HELP}")
                else:
                    self.turn(text)
        finally:
            self.close()

    def next_line(self) -> str | None:
        while True:
            try:
                text = input(f"{Speaker.USER} ").strip()
            except KeyboardInterrupt:
                console.print("\n(/new for a new chat, /exit or Ctrl+D to quit)")
                continue
            except EOFError:
                return None

            if text:
                return text

    def turn(self, text: str) -> None:
        """Send one message and print the reply. Ctrl+C cancels the turn, an error does not end the chat."""
        console.rule(style="dim")

        try:
            reply = self.session.send(text)

            console.print(said_by(Speaker.ASSISTANT, Markdown(reply)))
        except KeyboardInterrupt:  # during a turn: the trace keeps it with its error, the conversation goes on
            console.print("\n(turn cancelled)")
        except Exception as exc:  # e.g. the LLM endpoint is down: report it and keep the conversation
            log.exception("turn failed")
            console.print(f"(that turn failed: {type(exc).__name__}: {exc})")

        console.rule(style="dim")

    def start(self) -> ChatSession:
        session = self.new_session()
        console.print(f"▶ session {session.id} → {session.trace_path or NOT_TRACED}   (/help for commands)")

        return session

    def close(self) -> None:
        self.session.close()
        console.print(f"✔ closed session {self.session.id}")

    def new(self) -> None:
        self.close()
        self.session = self.start()
