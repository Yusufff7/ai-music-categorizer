from datetime import datetime


def print_step(message, level=1):
    """Helper function for consistent step printing"""
    prefix = "  " * (level-1) + "↳" if level > 1 else ""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {prefix} {message}")
