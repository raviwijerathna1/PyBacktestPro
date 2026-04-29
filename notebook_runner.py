"""
═══════════════════════════════════════════════════════════
notebook_runner.py — Jupyter Notebook (.ipynb) Strategy Loader
═══════════════════════════════════════════════════════════
Supports:
  ✅ Upload .ipynb file
  ✅ Extract strategy cell (class MyStrategy)
  ✅ Extract helper cells (imports, functions)
  ✅ Execute and load strategy
  ✅ Display notebook cells in UI
═══════════════════════════════════════════════════════════
"""

import json
import os
import re
import sys
import tempfile
import importlib
import traceback
from typing import Optional, Tuple, List, Dict


class NotebookLoader:
    """Load and execute strategy from Jupyter Notebook"""

    def __init__(self):
        self.cells: List[Dict] = []
        self.strategy_code: str = ""
        self.helper_code: str = ""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.notebook_name: str = ""

    def load_notebook(self, file_content: str = None,
                      file_path: str = None) -> bool:
        """
        Load notebook from file content (base64/json) or file path.
        Returns True if strategy found.
        """
        self.errors.clear()
        self.warnings.clear()
        self.cells.clear()

        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                self.notebook_name = os.path.basename(file_path)
            elif file_content:
                nb = json.loads(file_content)
                self.notebook_name = "uploaded.ipynb"
            else:
                self.errors.append("No notebook provided")
                return False

            # Extract cells
            cells = nb.get('cells', [])
            if not cells:
                self.errors.append("Notebook has no cells")
                return False

            code_cells = []
            for i, cell in enumerate(cells):
                if cell.get('cell_type') == 'code':
                    source = ''.join(cell.get('source', []))
                    if source.strip():
                        code_cells.append({
                            'index': i,
                            'source': source,
                            'has_strategy': 'class MyStrategy' in source or
                                           'class Strategy' in source,
                            'has_imports': source.strip().startswith('import ') or
                                         source.strip().startswith('from '),
                            'has_function': 'def ' in source,
                        })

            self.cells = code_cells

            # Find strategy cell
            strategy_cells = [c for c in code_cells if c['has_strategy']]
            if not strategy_cells:
                self.errors.append(
                    "No 'class MyStrategy(Strategy)' found in notebook. "
                    "Make sure your strategy class is named MyStrategy."
                )
                return False

            # Gather helper code (imports, functions before strategy)
            strategy_idx = strategy_cells[0]['index']
            helper_parts = []
            for cell in code_cells:
                if cell['index'] < strategy_idx:
                    # Skip cells that just have output/display
                    if not any(skip in cell['source'] for skip in [
                        'plt.show', 'display(', 'print(result',
                        '.plot(', 'fig.show'
                    ]):
                        helper_parts.append(cell['source'])

            self.helper_code = '\n\n'.join(helper_parts)
            self.strategy_code = strategy_cells[0]['source']

            # Check for common issues
            if 'from engine import' not in self.helper_code and \
               'from engine import' not in self.strategy_code:
                self.warnings.append(
                    "Missing 'from engine import Strategy' — will be added automatically"
                )

            if 'def init(self' not in self.strategy_code:
                self.warnings.append("Strategy missing init() method")

            if 'def next(self' not in self.strategy_code:
                self.errors.append("Strategy missing next() method — required!")
                return False

            return True

        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid notebook JSON: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error loading notebook: {e}")
            return False

    def get_full_code(self) -> str:
        """Get the complete executable Python code"""
        parts = []

        # Standard imports
        parts.append("""
import pandas as pd
import numpy as np
from engine import Strategy, IndicatorEngine as IND
""")

        # Pine-style helpers (if not already in notebook)
        if 'ta_sma' not in self.helper_code and 'ta_sma' not in self.strategy_code:
            parts.append("""
# Helper functions
def ta_sma(src, length):    return IND.sma(src, length)
def ta_ema(src, length):    return IND.ema(src, length)
def ta_rsi(src, length=14): return IND.rsi(src, length)
def ta_atr(h, l, c, length=14): return IND.atr(h, l, c, length)
def ta_macd(src, fast=12, slow=26, sig=9): return IND.macd(src, fast, slow, sig)
def ta_bb(src, length=20, mult=2.0): return IND.bollinger_bands(src, length, mult)
def ta_stoch(h, l, c, k=14, d=3): return IND.stochastic(h, l, c, k, d)
def ta_supertrend(h, l, c, length=10, mult=3.0): return IND.supertrend(h, l, c, length, mult)

def cross_above(a, b, bar):
    if bar < 1: return False
    try: return a.iloc[bar-1] <= b.iloc[bar-1] and a.iloc[bar] > b.iloc[bar]
    except: return False

def cross_below(a, b, bar):
    if bar < 1: return False
    try: return a.iloc[bar-1] >= b.iloc[bar-1] and a.iloc[bar] < b.iloc[bar]
    except: return False
""")

        # User's helper code
        if self.helper_code.strip():
            parts.append(f"\n# ── From Notebook ──\n{self.helper_code}")

        # Strategy code
        parts.append(f"\n# ── Strategy ──\n{self.strategy_code}")

        return '\n'.join(parts)

    def save_as_strategy(self, filepath: str = 'my_strategy.py') -> bool:
        """Save the extracted strategy to a Python file"""
        try:
            code = self.get_full_code()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            return True
        except Exception as e:
            self.errors.append(f"Error saving: {e}")
            return False

    def execute_strategy(self) -> Tuple[Optional[object], List[str]]:
        """
        Execute the strategy code and return the strategy class.
        Returns (strategy_instance, errors)
        """
        errors = []
        try:
            code = self.get_full_code()

            # Write to temp file
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, 'nb_strategy.py')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # Add to path and import
            if tmp_dir not in sys.path:
                sys.path.insert(0, tmp_dir)

            # Import the module
            spec = importlib.util.spec_from_file_location(
                'nb_strategy', tmp_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get MyStrategy class
            if hasattr(module, 'MyStrategy'):
                strategy = module.MyStrategy()
                return strategy, []
            else:
                errors.append("No MyStrategy class found after execution")
                return None, errors

        except SyntaxError as e:
            errors.append(f"Syntax error in notebook code: {e}")
            return None, errors
        except Exception as e:
            errors.append(f"Execution error: {e}\n{traceback.format_exc()}")
            return None, errors

    def get_cell_preview(self) -> List[Dict]:
        """Get cell previews for UI display"""
        previews = []
        for cell in self.cells:
            source = cell['source']
            # Truncate long cells
            lines = source.split('\n')
            preview = '\n'.join(lines[:10])
            if len(lines) > 10:
                preview += f'\n... ({len(lines) - 10} more lines)'

            previews.append({
                'index': cell['index'],
                'preview': preview,
                'is_strategy': cell['has_strategy'],
                'is_import': cell['has_imports'],
                'has_function': cell['has_function'],
                'line_count': len(lines),
            })
        return previews


def load_ipynb(file_content: str = None,
               file_path: str = None) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Convenience function: Load .ipynb → Python code string
    Returns: (python_code, errors, warnings)
    """
    loader = NotebookLoader()
    success = loader.load_notebook(file_content, file_path)
    if success:
        return loader.get_full_code(), loader.errors, loader.warnings
    return None, loader.errors, loader.warnings


def save_ipynb_as_strategy(file_content: str = None,
                           file_path: str = None,
                           output_path: str = 'my_strategy.py') -> bool:
    """Save notebook strategy directly to my_strategy.py"""
    loader = NotebookLoader()
    if loader.load_notebook(file_content, file_path):
        return loader.save_as_strategy(output_path)
    return False