"""Entry point for running wikipedia_health as a module.

This allows running the CLI using:
    python -m wikipedia_health [args]
"""

from wikipedia_health.cli import main

if __name__ == '__main__':
    exit(main())
