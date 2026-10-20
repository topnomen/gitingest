#!/usr/bin/env python3

"""Entry point for the package execution script."""

from src.main import main


if __name__ == "__main__":
    """Run the main application entry point."""
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVoluntary Interruption!!!!\n")
