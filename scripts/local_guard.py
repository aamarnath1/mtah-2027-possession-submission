"""Process-local I/O restrictions for a cooperative local agent, not an OS sandbox."""

from pathlib import Path
import sys


def install(read_paths: list[Path], write_paths: list[Path], inference: bool = False) -> None:
    """Restrict Python file, subprocess and socket operations with an audit hook."""
    readable = [p.resolve() for p in read_paths + write_paths]
    writable = [p.resolve() for p in write_paths]

    def permitted(value: object, roots: list[Path]) -> bool:
        if isinstance(value, int):
            return True
        if not isinstance(value, (str, bytes, Path)):
            return False
        path = Path(value.decode() if isinstance(value, bytes) else value).resolve()
        if path.name == ".env" or path.name.startswith(".env."):
            return False
        return any(path == root or root.is_dir() and path.is_relative_to(root) for root in roots)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            mode, flags = args[1], args[2]
            writing = any(c in (mode or "") for c in "wax+") or bool(flags & 0x703)
            if not permitted(args[0], writable if writing else readable):
                raise PermissionError("Local file policy refused access")
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod"}:
            if not permitted(args[0], writable):
                raise PermissionError("Local write policy refused access")
        elif event in {"os.rename", "os.link", "os.symlink"}:
            if not all(permitted(arg, writable) for arg in args[:2]):
                raise PermissionError("Local move policy refused access")
        elif event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
            raise PermissionError("Subprocess tools are disabled")
        elif event == "socket.getaddrinfo":
            if not inference or args[0] not in {"localhost", "127.0.0.1", "::1"}:
                raise PermissionError("External name lookup disabled")
        elif event == "socket.connect":
            address = args[1]
            if not (inference and isinstance(address, tuple)
                    and address[0] in {"127.0.0.1", "::1"} and address[1] == 11434):
                raise PermissionError("Only local Ollama inference is allowed")
        elif event in {"socket.bind", "socket.sendto"}:
            raise PermissionError("Listening and datagram networking disabled")

    sys.addaudithook(hook)
