import os
import sys

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from panda_main import PandaApp


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = PandaApp(csv_path=csv_path)
    app.run()


if __name__ == "__main__":
    main()
