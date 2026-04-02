"""Runner script to start the Panda app as a separate process (popout)."""
import os
import sys

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from panda_main import PandaApp


def main():
    app = PandaApp()
    app.run()


if __name__ == "__main__":
    main()
