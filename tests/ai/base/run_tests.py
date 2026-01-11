#!/usr/bin/env python3
"""
Test runner script for base module tests.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py test_providers     # Run specific test file
    python run_tests.py --coverage         # Run with coverage
"""

import sys
import unittest
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_tests(test_name=None, coverage=False):
    """Run tests."""
    if coverage:
        try:
            import coverage
            cov = coverage.Coverage(source=['src/core/ai/base'])
            cov.start()
        except ImportError:
            print("Warning: coverage not installed. Install with: pip install coverage")
            coverage = False
    
    # Discover and run tests
    loader = unittest.TestLoader()
    if test_name:
        suite = loader.loadTestsFromName(f'tests.ai.base.{test_name}')
    else:
        suite = loader.discover('tests/ai/base', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if coverage:
        cov.stop()
        cov.save()
        print("\n" + "="*70)
        print("Coverage Report")
        print("="*70)
        cov.report()
        print("\nHTML report generated in htmlcov/")
        cov.html_report(directory='htmlcov')
    
    return result.wasSuccessful()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run base module tests')
    parser.add_argument('test_name', nargs='?', help='Specific test file to run (without test_ prefix)')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    args = parser.parse_args()
    
    success = run_tests(args.test_name, args.coverage)
    sys.exit(0 if success else 1)
