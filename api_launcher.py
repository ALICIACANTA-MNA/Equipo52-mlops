#!/usr/bin/env python3
"""
Compatibility wrapper for the API launcher.
Redirects to the unified launcher in src/api/ following MLOps best practices.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == '__main__':
    from api.launcher import main
    sys.exit(main())