"""Logging utilities with colored terminal output."""


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class Log:
    """Logging utility class with colored output."""

    @staticmethod
    def info(msg: str) -> None:
        """Log informational message in blue."""
        print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")

    @staticmethod
    def success(msg: str) -> None:
        """Log success message in green."""
        print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

    @staticmethod
    def warning(msg: str) -> None:
        """Log warning message in yellow."""
        print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")

    @staticmethod
    def error(msg: str) -> None:
        """Log error message in red."""
        print(f"{Colors.RED}✗ {msg}{Colors.RESET}")

    @staticmethod
    def query(source: str, cause: str, query: str, count: int, elapsed: float) -> None:
        """Log query details with structured formatting."""
        print(
            f"{Colors.CYAN}→ {source}{Colors.RESET} | {Colors.BOLD}{cause}{Colors.RESET}"
        )
        print(f"  {Colors.DIM}Query: {query}{Colors.RESET}")
        print(
            f"  {Colors.GREEN}Found {count:,} mentions{Colors.RESET} {Colors.DIM}({elapsed:.2f}s){Colors.RESET}"
        )

    @staticmethod
    def section(title: str) -> None:
        """Log section header."""
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{title}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")

    @staticmethod
    def summary(title: str, **kwargs) -> None:
        """Log summary information with key-value pairs.

        Args:
            title: Section title
            **kwargs: Key-value pairs to display
        """
        print()
        print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{title}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
        for key, value in kwargs.items():
            # Format the key nicely (replace underscores with spaces, capitalize)
            formatted_key = key.replace("_", " ").capitalize()
            print(f"{Colors.DIM}{formatted_key}:{Colors.RESET} {value}")
        print()

    @staticmethod
    def elapsed(minutes: int, seconds: int) -> None:
        """Log elapsed time in dim color.

        Args:
            minutes: Number of minutes elapsed
            seconds: Number of seconds elapsed
        """
        print(f"{Colors.DIM}Elapsed: {minutes:02d}:{seconds:02d}{Colors.RESET}")
