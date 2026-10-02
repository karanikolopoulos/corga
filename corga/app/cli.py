"""Terminal chat loop. Commands: /new /state /trace /help /exit. Ctrl+C cancels a turn, Ctrl+D exits."""

import logging

from typing import Protocol
from pathlib import Path
from collections.abc import Callable

HELP = """/new    close this conversation (trace saved) and start a fresh one
/state  show what the assistant remembers: time window, preferences, current itinerary
/trace  show the path of this conversation's trace file
/help   show this help
/exit   close the conversation and quit (Ctrl+D also works)"""
EXIT = {"/exit", "/quit"}
NOT_TRACED = "not traced (hydra=debug)"

log = logging.getLogger(__name__)


class Session(Protocol):
    """What the loop needs from a conversation; `corga.agent.runtime.ChatSession` provides it."""

    id: str

    @property
    def trace_path(self) -> Path | None: ...
    def send(self, text: str) -> str: ...
    def summary(self) -> str: ...
    def close(self) -> None: ...


class Repl:
    """Reads lines, runs commands or turns, and always closes the current session on the way out."""

    def __init__(self, new_session: Callable[[], Session], read: Callable[[str], str], write: Callable[[str], None]):
        self.new_session, self.read, self.write = new_session, read, write
        self.session = self.start()
        self.commands: dict[str, Callable[[], None]] = {
            "/help": lambda: self.write(HELP),
            "/state": lambda: self.write(self.session.summary()),
            "/trace": lambda: self.write(str(self.session.trace_path or NOT_TRACED)),
            "/new": self.new,
        }

    def run(self) -> None:
        try:
            while (text := self.next_line()) is not None and text not in EXIT:
                if command := self.commands.get(text):
                    command()
                else:
                    self.turn(text)
        finally:
            self.close()

    def next_line(self) -> str | None:
        """The next non-empty line, or None on Ctrl+D; Ctrl+C at the prompt only prints a hint."""
        while True:
            try:
                text = self.read("you> ").strip()
            except KeyboardInterrupt:
                self.write("\n(/new for a new chat, /exit or Ctrl+D to quit)")
                continue
            except EOFError:
                return None
            if text:
                return text

    def turn(self, text: str) -> None:
        try:
            self.write(f"bot> {self.session.send(text)}")
        except KeyboardInterrupt:  # during a turn: the session records it, the conversation goes on
            self.write("\n(turn cancelled)")
        except Exception as exc:  # e.g. the LLM endpoint is down: report it and keep the conversation
            log.exception("turn failed")
            self.write(f"(that turn failed: {type(exc).__name__}: {exc})")

    def start(self) -> Session:
        session = self.new_session()
        self.write(f"▶ session {session.id} → {session.trace_path or NOT_TRACED}   (/help for commands)")
        return session

    def close(self) -> None:
        self.session.close()
        self.write(f"✔ closed session {self.session.id}")

    def new(self) -> None:
        self.close()
        self.session = self.start()


def repl(
    new_session: Callable[[], Session],
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    Repl(new_session, read, write).run()
