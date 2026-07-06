from pathlib import Path
import sys


APP_DIR = Path(__file__).resolve().parent / "app"
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

from pm25_streamlit_app import main  # noqa: E402


main()
