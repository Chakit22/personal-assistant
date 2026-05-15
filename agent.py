import os
import sys

from dotenv import load_dotenv

REQUIRED_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")


def _require_env(*keys: str) -> None:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in your .env (see .env.example)."
        )


def main() -> None:
    load_dotenv()
    _require_env(*REQUIRED_ENV_VARS)
    print("personal-assistant scaffold ready")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
