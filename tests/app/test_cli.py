from corga.app import repl


class FakeSession:
    count = 0

    def __init__(self, traced=True):
        FakeSession.count += 1
        self.id = f"s{FakeSession.count}"
        self.trace_path = f"/runs/{self.id}.jsonl" if traced else None
        self.sent, self.closed = [], False

    def send(self, text):
        self.sent.append(text)
        return f"echo {text}"

    def summary(self):
        return "time window: none"

    def close(self):
        self.closed = True


def scripted(*items):
    """A fake `input`: returns each item in turn, raising those that are exceptions, then EOF (Ctrl+D)."""
    it = iter(items)

    def read(_prompt):
        item = next(it, EOFError())
        if isinstance(item, BaseException):
            raise item
        return item

    return read


def run_repl(*inputs, make=FakeSession):
    sessions, out = [], []

    def factory():
        sessions.append(make())
        return sessions[-1]

    repl(factory, read=scripted(*inputs), write=out.append)
    return sessions, "\n".join(out)


def test_new_closes_old_session_and_starts_a_fresh_one():
    sessions, out = run_repl("hello", "/new", "again", "/exit")
    assert [s.sent for s in sessions] == [["hello"], ["again"]]
    assert all(s.closed for s in sessions) and "bot> echo hello" in out


def test_eof_exits_and_closes():
    sessions, _ = run_repl("hi")
    assert sessions[0].closed


def test_ctrl_c_at_prompt_only_prints_a_hint():
    sessions, out = run_repl(KeyboardInterrupt(), "still here", "/exit")
    assert sessions[0].sent == ["still here"] and "/new for a new chat" in out


def test_ctrl_c_during_a_turn_cancels_only_that_turn():
    class Slow(FakeSession):
        def send(self, text):
            raise KeyboardInterrupt

    sessions, out = run_repl("plan", "/exit", make=Slow)
    assert "(turn cancelled)" in out and sessions[0].closed


def test_a_failing_turn_is_reported_and_the_chat_goes_on():
    class Down(FakeSession):
        def send(self, text):
            if text == "first":
                raise ConnectionError("endpoint unreachable")
            return super().send(text)

    sessions, out = run_repl("first", "second", "/exit", make=Down)
    assert "that turn failed: ConnectionError: endpoint unreachable" in out and "bot> echo second" in out


def test_trace_help_and_state_commands():
    _, out = run_repl("/trace", "/help", "/state", "/exit")
    assert ".jsonl" in out and "/new" in out and "time window: none" in out


def test_untraced_session_says_so():
    _, out = run_repl("/trace", "/exit", make=lambda: FakeSession(traced=False))
    assert "not traced (hydra=debug)" in out
