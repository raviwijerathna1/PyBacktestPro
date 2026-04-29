"""
═══════════════════════════════════════════════════════════
pine_parser.py — TradingView Pine Script → Python Converter
═══════════════════════════════════════════════════════════
Supports:
  ✅ ta.sma, ta.ema, ta.rsi, ta.macd, ta.atr, ta.bb
  ✅ ta.crossover, ta.crossunder
  ✅ strategy.entry, strategy.close, strategy.exit
  ✅ input() parameters
  ✅ if/else conditions
  ✅ Variables & assignments
  ✅ math operations
  ✅ plot() (ignored for backtest)
  ✅ strategy() declaration
═══════════════════════════════════════════════════════════
"""

import re
import textwrap
from typing import Dict, List, Tuple, Optional


class PineScriptParser:
    """
    Converts TradingView Pine Script v5 → Python Strategy class.
    Not 100% coverage but handles most common patterns.
    """

    def __init__(self):
        self.params: Dict[str, dict] = {}
        self.indicators: List[str] = []
        self.conditions: List[str] = []
        self.init_lines: List[str] = []
        self.next_lines: List[str] = []
        self.strategy_name = "PineStrategy"
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # Pine → Python function mapping
        self.func_map = {
            'ta.sma': 'ta_sma',
            'ta.ema': 'ta_ema',
            'ta.rsi': 'ta_rsi',
            'ta.atr': 'ta_atr',
            'ta.macd': 'ta_macd',
            'ta.bb': 'ta_bb',
            'ta.stoch': 'ta_stoch',
            'ta.cci': 'IND.cci',
            'ta.mfi': 'IND.mfi',
            'ta.obv': 'IND.obv',
            'ta.vwap': 'IND.vwap',
            'ta.roc': 'IND.roc',
            'ta.mom': 'IND.momentum',
            'ta.wma': 'IND.wma',
            'ta.dema': 'IND.dema',
            'ta.tema': 'IND.tema',
            'ta.hma': 'IND.hull_ma',
            'ta.supertrend': 'ta_supertrend',
            'ta.tr': 'IND.true_range',
            'ta.kc': 'IND.keltner_channels',
            'ta.dc': 'IND.donchian_channels',
            'ta.adx': 'IND.adx',
            'ta.aroon': 'IND.aroon',
            'ta.sar': 'IND.parabolic_sar',
            'ta.willr': 'IND.williams_r',
            'ta.crossover': 'cross_above',
            'ta.crossunder': 'cross_below',
            'ta.highest': '_highest',
            'ta.lowest': '_lowest',
            'ta.change': '_change',
            'ta.valuewhen': '_valuewhen',
            'ta.barssince': '_barssince',
            'math.abs': 'abs',
            'math.max': 'max',
            'math.min': 'min',
            'math.round': 'round',
            'math.sqrt': 'np.sqrt',
            'math.log': 'np.log',
            'math.pow': 'pow',
            'nz': '_nz',
            'na': 'pd.isna',
        }

        # Pine source variables → Python
        self.var_map = {
            'close': "data['Close'].iloc[bar]",
            'open': "data['Open'].iloc[bar]",
            'high': "data['High'].iloc[bar]",
            'low': "data['Low'].iloc[bar]",
            'volume': "data['Volume'].iloc[bar]",
            'hl2': "((data['High'].iloc[bar] + data['Low'].iloc[bar]) / 2)",
            'hlc3': "((data['High'].iloc[bar] + data['Low'].iloc[bar] + data['Close'].iloc[bar]) / 3)",
            'ohlc4': "((data['Open'].iloc[bar] + data['High'].iloc[bar] + data['Low'].iloc[bar] + data['Close'].iloc[bar]) / 4)",
            'bar_index': 'bar',
        }

        # Pine series variables → Python (for indicator calculations)
        self.series_map = {
            'close': "c",
            'open': "self._data['Open']",
            'high': "self._data['High']",
            'low': "self._data['Low']",
            'volume': "self._data['Volume']",
            'hl2': "((self._data['High'] + self._data['Low']) / 2)",
            'hlc3': "((self._data['High'] + self._data['Low'] + self._data['Close']) / 3)",
        }

    def parse(self, pine_code: str) -> str:
        """Parse Pine Script and return Python strategy code"""
        self.errors.clear()
        self.warnings.clear()
        self.params.clear()
        self.init_lines.clear()
        self.next_lines.clear()

        lines = pine_code.strip().split('\n')
        clean_lines = self._preprocess(lines)

        for line in clean_lines:
            self._parse_line(line)

        return self._generate_python()

    def _preprocess(self, lines: List[str]) -> List[str]:
        """Clean and preprocess Pine Script lines"""
        result = []
        in_block_comment = False
        continued = ""

        for line in lines:
            # Remove block comments
            if '/*' in line:
                in_block_comment = True
                line = line[:line.index('/*')]
            if '*/' in line:
                in_block_comment = False
                line = line[line.index('*/') + 2:]
                continue
            if in_block_comment:
                continue

            # Remove single line comments
            if '//' in line:
                line = line[:line.index('//')]

            line = line.rstrip()

            # Skip empty lines
            if not line.strip():
                continue

            # Skip version indicator
            if line.strip().startswith('//@version'):
                continue

            # Handle line continuation
            if line.rstrip().endswith('\\'):
                continued += line.rstrip()[:-1] + " "
                continue
            if continued:
                line = continued + line
                continued = ""

            result.append(line)

        return result

    def _parse_line(self, line: str):
        """Parse a single Pine Script line"""
        stripped = line.strip()

        # Skip indicator/strategy declaration (extract name)
        if stripped.startswith('strategy(') or stripped.startswith('indicator('):
            self._parse_strategy_decl(stripped)
            return

        # Skip plot/plotshape/bgcolor etc
        if any(stripped.startswith(f) for f in [
            'plot(', 'plotshape(', 'plotchar(', 'bgcolor(',
            'barcolor(', 'hline(', 'fill(', 'label.',
            'line.', 'box.', 'table.', 'alert('
        ]):
            return

        # Parse input()
        if 'input(' in stripped or 'input.int(' in stripped or \
           'input.float(' in stripped or 'input.bool(' in stripped or \
           'input.string(' in stripped or 'input.source(' in stripped:
            self._parse_input(stripped)
            return

        # Parse variable assignments with indicators
        if '=' in stripped and not stripped.startswith('if') and \
           not stripped.startswith('else') and not stripped.startswith('for'):
            self._parse_assignment(stripped)
            return

        # Parse strategy entries/exits
        if 'strategy.entry' in stripped:
            self._parse_entry(stripped)
            return
        if 'strategy.close' in stripped:
            self._parse_close(stripped)
            return
        if 'strategy.exit' in stripped:
            self._parse_exit(stripped)
            return

        # Parse if conditions
        if stripped.startswith('if '):
            self._parse_if(stripped)
            return

    def _parse_strategy_decl(self, line: str):
        """Extract strategy name"""
        m = re.search(r'(?:strategy|indicator)\s*\(\s*["\']([^"\']+)["\']', line)
        if m:
            name = m.group(1)
            self.strategy_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def _parse_input(self, line: str):
        """Parse input() declarations → parameters"""
        # Match: varName = input(defval=X, title="Y") or input.int(X, "Y")
        m = re.match(r'(\w+)\s*=\s*input(?:\.(?:int|float|bool|string|source))?\s*\((.+)\)', line)
        if not m:
            return

        var_name = m.group(1)
        args_str = m.group(2)

        # Extract default value
        defval = None
        title = var_name

        # Try defval= keyword
        dm = re.search(r'defval\s*=\s*([^,\)]+)', args_str)
        if dm:
            defval = dm.group(1).strip().strip('"\'')
        else:
            # First positional arg
            parts = self._split_args(args_str)
            if parts:
                defval = parts[0].strip().strip('"\'')

        # Try title= keyword
        tm = re.search(r'title\s*=\s*["\']([^"\']+)["\']', args_str)
        if tm:
            title = tm.group(1)

        # Determine type and convert
        try:
            if defval in ('true', 'True'):
                defval = True
            elif defval in ('false', 'False'):
                defval = False
            elif '.' in str(defval):
                defval = float(defval)
            else:
                defval = int(defval)
        except (ValueError, TypeError):
            if defval is None:
                defval = 0

        self.params[var_name] = {
            'default': defval,
            'title': title,
        }

    def _parse_assignment(self, line: str):
        """Parse variable assignments, especially indicator calculations"""
        # Split on first =
        parts = line.split('=', 1)
        if len(parts) != 2:
            return

        lhs = parts[0].strip()
        rhs = parts[1].strip()

        # Skip type declarations
        for prefix in ['var ', 'varip ', 'int ', 'float ', 'bool ', 'string ', 'color ']:
            if lhs.startswith(prefix):
                lhs = lhs[len(prefix):].strip()

        # Handle tuple unpacking: [a, b, c] = ta.macd(...)
        if lhs.startswith('[') and ']' in lhs:
            self._parse_tuple_assignment(lhs, rhs)
            return

        # Convert RHS
        py_rhs = self._convert_expression(rhs, context='init')

        # Check if it's an indicator (series) or a scalar
        is_indicator = any(f in rhs for f in [
            'ta.sma', 'ta.ema', 'ta.rsi', 'ta.atr', 'ta.macd',
            'ta.bb', 'ta.stoch', 'ta.cci', 'ta.mfi', 'ta.wma',
            'ta.dema', 'ta.tema', 'ta.hma', 'ta.supertrend',
            'ta.adx', 'ta.aroon', 'ta.sar', 'ta.willr',
            'ta.obv', 'ta.vwap', 'ta.roc', 'ta.mom',
            'ta.kc', 'ta.dc', 'ta.tr',
        ])

        if is_indicator:
            self.init_lines.append(f"        self._data['{lhs}'] = {py_rhs}")
        else:
            # Could be a simple variable used in conditions
            self.init_lines.append(f"        # {lhs} = {rhs}")

    def _parse_tuple_assignment(self, lhs: str, rhs: str):
        """Parse [a, b, c] = ta.macd(close)"""
        vars_str = lhs.strip('[] ')
        var_names = [v.strip() for v in vars_str.split(',')]

        if 'ta.macd' in rhs:
            m = re.search(r'ta\.macd\(([^)]*)\)', rhs)
            args = self._parse_func_args(m.group(1) if m else 'close')
            src = self._convert_series_var(args.get(0, 'close'))
            fast = args.get(1, args.get('fastlen', '12'))
            slow = args.get(2, args.get('slowlen', '26'))
            sig = args.get(3, args.get('siglen', '9'))
            if len(var_names) >= 3:
                self.init_lines.append(
                    f"        _{var_names[0]}, _{var_names[1]}, _{var_names[2]} = "
                    f"ta_macd({src}, {fast}, {slow}, {sig})"
                )
                for vn in var_names:
                    if vn != '_':
                        self.init_lines.append(
                            f"        self._data['{vn}'] = _{vn}"
                        )
        elif 'ta.bb' in rhs:
            m = re.search(r'ta\.bb\(([^)]*)\)', rhs)
            args = self._parse_func_args(m.group(1) if m else 'close, 20, 2')
            src = self._convert_series_var(args.get(0, 'close'))
            period = args.get(1, '20')
            mult = args.get(2, '2')
            if len(var_names) >= 3:
                self.init_lines.append(
                    f"        _{var_names[0]}, _{var_names[1]}, _{var_names[2]} = "
                    f"ta_bb({src}, {period}, {mult})"
                )
                for vn in var_names:
                    if vn != '_':
                        self.init_lines.append(
                            f"        self._data['{vn}'] = _{vn}"
                        )
        elif 'ta.stoch' in rhs:
            m = re.search(r'ta\.stoch\(([^)]*)\)', rhs)
            args_str = m.group(1) if m else 'close, high, low, 14, 3'
            self.init_lines.append(
                f"        # Stochastic: {rhs}"
            )
            if len(var_names) >= 2:
                self.init_lines.append(
                    f"        self._data['{var_names[0]}'], self._data['{var_names[1]}'] = "
                    f"ta_stoch(self._data['High'], self._data['Low'], c)"
                )
        elif 'ta.supertrend' in rhs:
            m = re.search(r'ta\.supertrend\(([^)]*)\)', rhs)
            args = self._parse_func_args(m.group(1) if m else '3, 10')
            mult = args.get(0, '3')
            period = args.get(1, '10')
            if len(var_names) >= 2:
                self.init_lines.append(
                    f"        self._data['{var_names[0]}'], self._data['{var_names[1]}'] = "
                    f"ta_supertrend(self._data['High'], self._data['Low'], c, {period}, {mult})"
                )

    def _parse_entry(self, line: str):
        """Parse strategy.entry()"""
        m = re.search(r'strategy\.entry\s*\(([^)]+)\)', line)
        if not m:
            return
        args = self._parse_func_args(m.group(1))
        entry_id = args.get(0, '"Long"').strip('"\'')
        direction = args.get(1, 'strategy.long')

        # Get condition from preceding if
        is_long = 'long' in direction.lower()

        if is_long:
            # Check for stop/limit/qty
            sl = args.get('stop', None)
            tp = args.get('limit', None)
            qty = args.get('qty', None)
            comment = args.get('comment', f'"{entry_id}"')

            sl_str = f", sl={self._convert_bar_expr(sl)}" if sl else ""
            tp_str = f", tp={self._convert_bar_expr(tp)}" if tp else ""

            self.next_lines.append(
                f"            self.buy({sl_str}{tp_str}, tag='{entry_id}')"
            )
        else:
            sl = args.get('stop', None)
            tp = args.get('limit', None)
            sl_str = f", sl={self._convert_bar_expr(sl)}" if sl else ""
            tp_str = f", tp={self._convert_bar_expr(tp)}" if tp else ""
            self.next_lines.append(
                f"            self.sell({sl_str}{tp_str}, tag='{entry_id}')"
            )

    def _parse_close(self, line: str):
        """Parse strategy.close()"""
        m = re.search(r'strategy\.close\s*\(([^)]*)\)', line)
        if not m:
            return
        args = self._parse_func_args(m.group(1))
        close_id = args.get(0, '"Long"').strip('"\'')
        self.next_lines.append(
            f"            self.close_position(tag='{close_id}')"
        )

    def _parse_exit(self, line: str):
        """Parse strategy.exit()"""
        m = re.search(r'strategy\.exit\s*\(([^)]+)\)', line)
        if not m:
            return
        args = self._parse_func_args(m.group(1))
        exit_id = args.get(0, '"Exit"').strip('"\'')
        from_entry = args.get(1, args.get('from_entry', '""')).strip('"\'')
        sl = args.get('stop', args.get('loss', None))
        tp = args.get('limit', args.get('profit', None))
        trail = args.get('trail_points', args.get('trail_offset', None))

        # This gets added as a note — actual SL/TP is set on entry
        self.warnings.append(
            f"strategy.exit '{exit_id}' → set SL/TP on entry instead"
        )

    def _parse_if(self, line: str):
        """Parse if condition"""
        # Extract condition
        condition = line[3:].strip()
        if condition.endswith(':'):
            condition = condition[:-1]

        py_condition = self._convert_expression(condition, context='bar')

        # Store as pending condition for next entry/close
        self.next_lines.append(f"        if {py_condition}:")

    def _convert_expression(self, expr: str, context='init') -> str:
        """Convert Pine expression to Python"""
        result = expr.strip()

        # Replace Pine functions with Python equivalents
        for pine_func, py_func in self.func_map.items():
            if pine_func in result:
                result = self._replace_function(result, pine_func, py_func, context)

        # Replace Pine operators
        result = result.replace(' and ', ' and ')
        result = result.replace(' or ', ' or ')
        result = result.replace(' not ', ' not ')
        result = result.replace('true', 'True')
        result = result.replace('false', 'False')

        # Replace source variables
        if context == 'init':
            for pine_var, py_var in self.series_map.items():
                result = re.sub(r'\b' + pine_var + r'\b', py_var, result)
        elif context == 'bar':
            for pine_var, py_var in self.var_map.items():
                result = re.sub(r'\b' + pine_var + r'\b', py_var, result)

        # Handle close[1] → data['Close'].iloc[bar-1]
        result = re.sub(
            r"data\['(\w+)'\]\.iloc\[bar\]\[(\d+)\]",
            r"data['\1'].iloc[bar-\2]",
            result
        )
        result = re.sub(
            r'(\w+)\[(\d+)\]',
            lambda m: f"data['{m.group(1)}'].iloc[bar-{m.group(2)}]"
            if m.group(1) in ('close','open','high','low','volume')
            else m.group(0),
            result
        )

        # Handle param references
        for param_name in self.params:
            result = re.sub(
                r'\b' + param_name + r'\b',
                f'self.{param_name}',
                result
            )

        return result

    def _convert_bar_expr(self, expr: str) -> str:
        """Convert expression for bar-by-bar context"""
        if expr is None:
            return 'None'
        return self._convert_expression(expr, context='bar')

    def _convert_series_var(self, var_name: str) -> str:
        """Convert Pine series variable for init context"""
        var_name = var_name.strip()
        if var_name in self.series_map:
            return self.series_map[var_name]
        if var_name in self.params:
            return f'self.{var_name}'
        return var_name

    def _replace_function(self, text: str, pine_func: str,
                          py_func: str, context: str) -> str:
        """Replace Pine function call with Python equivalent"""
        pattern = re.escape(pine_func) + r'\s*\(([^)]*)\)'
        m = re.search(pattern, text)
        if not m:
            return text

        args_str = m.group(1)

        # Special handling for crossover/crossunder
        if pine_func in ('ta.crossover', 'ta.crossunder'):
            args = [a.strip() for a in args_str.split(',')]
            if len(args) >= 2:
                if context == 'bar':
                    a = self._convert_bar_series_ref(args[0])
                    b = self._convert_bar_series_ref(args[1])
                    return text[:m.start()] + f"{py_func}({a}, {b}, bar)" + text[m.end():]
                else:
                    a = self._convert_series_var(args[0])
                    b = self._convert_series_var(args[1])
                    return text[:m.start()] + f"{py_func}({a}, {b}, bar)" + text[m.end():]

        # For indicator functions in init context
        if context == 'init':
            converted_args = []
            for arg in self._split_args(args_str):
                arg = arg.strip()
                converted_args.append(self._convert_series_var(arg))
            new_call = f"{py_func}({', '.join(converted_args)})"
            return text[:m.start()] + new_call + text[m.end():]

        return text

    def _convert_bar_series_ref(self, var_name: str) -> str:
        """Convert a variable reference for crossover/crossunder at bar level"""
        var_name = var_name.strip()
        if var_name in ('close','open','high','low','volume'):
            return f"data['{var_name.capitalize()}']"
        if var_name in self.params:
            return f"data['{var_name}']"
        # Assume it's a computed indicator stored in data
        return f"data['{var_name}']"

    def _parse_func_args(self, args_str: str) -> Dict:
        """Parse function arguments into dict (positional + keyword)"""
        result = {}
        parts = self._split_args(args_str)
        pos = 0
        for part in parts:
            part = part.strip()
            if '=' in part and not part.startswith('"'):
                key, val = part.split('=', 1)
                result[key.strip()] = val.strip()
            else:
                result[pos] = part
                pos += 1
        return result

    def _split_args(self, args_str: str) -> List[str]:
        """Split arguments respecting nested parentheses and strings"""
        result = []
        depth = 0
        current = ""
        in_string = False
        string_char = None

        for ch in args_str:
            if ch in ('"', "'") and not in_string:
                in_string = True
                string_char = ch
                current += ch
            elif ch == string_char and in_string:
                in_string = False
                string_char = None
                current += ch
            elif ch == '(' and not in_string:
                depth += 1
                current += ch
            elif ch == ')' and not in_string:
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0 and not in_string:
                result.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            result.append(current)
        return result

    def _generate_python(self) -> str:
        """Generate the final Python strategy code"""
        # Build parameter lines
        param_lines = []
        for name, info in self.params.items():
            val = info['default']
            if isinstance(val, str):
                param_lines.append(f"    {name} = '{val}'")
            elif isinstance(val, bool):
                param_lines.append(f"    {name} = {val}")
            else:
                param_lines.append(f"    {name} = {val}")

        # Build init body
        init_body = []
        init_body.append("        c = self._data['Close']")
        init_body.append("        h = self._data['High']")
        init_body.append("        l = self._data['Low']")
        init_body.append("        v = self._data.get('Volume', c * 0)")
        init_body.extend(self.init_lines)
        if not self.init_lines:
            init_body.append("        pass")

        # Build next body — figure out min bars needed
        min_bars = 50  # safe default
        for p in self.params.values():
            if isinstance(p['default'], (int, float)) and p['default'] > min_bars:
                min_bars = int(p['default']) + 5

        next_body = []
        next_body.append(f"        if bar < {min_bars}:")
        next_body.append(f"            return")
        next_body.append("")

        if self.next_lines:
            next_body.extend(self.next_lines)
        else:
            next_body.append("        pass  # No trading logic parsed")

        # Assemble
        code = f'''"""
Pine Script Strategy → Python (auto-converted)
Original: {self.strategy_name}
"""

import pandas as pd
import numpy as np
from engine import Strategy, IndicatorEngine as IND

# Pine Script helper functions
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
    try: return (a.iloc[bar-1] <= b.iloc[bar-1] and a.iloc[bar] > b.iloc[bar])
    except: return False

def cross_below(a, b, bar):
    if bar < 1: return False
    try: return (a.iloc[bar-1] >= b.iloc[bar-1] and a.iloc[bar] < b.iloc[bar])
    except: return False

def _nz(val, replacement=0):
    return replacement if pd.isna(val) else val

def _highest(src, length):
    return src.rolling(length).max()

def _lowest(src, length):
    return src.rolling(length).min()

def _change(src, length=1):
    return src.diff(length)


class MyStrategy(Strategy):
    """
    {self.strategy_name}
    Auto-converted from Pine Script
    """

    # ── Parameters ──
{chr(10).join(param_lines) if param_lines else "    pass"}

    def init(self):
        """Indicators (vectorized)"""
{chr(10).join(init_body)}

    def next(self, bar, data):
        """Bar-by-bar logic"""
{chr(10).join(next_body)}
'''
        return code

    def get_errors(self) -> List[str]:
        return self.errors

    def get_warnings(self) -> List[str]:
        return self.warnings


def convert_pine_to_python(pine_code: str) -> Tuple[str, List[str], List[str]]:
    """
    Convenience function: Pine Script → Python code string
    Returns: (python_code, errors, warnings)
    """
    parser = PineScriptParser()
    py_code = parser.parse(pine_code)
    return py_code, parser.get_errors(), parser.get_warnings()


def validate_pine_code(pine_code: str) -> dict:
    """Quick validation of Pine Script code"""
    result = {
        'valid': True,
        'has_strategy': False,
        'has_entry': False,
        'has_close': False,
        'has_indicators': False,
        'has_inputs': False,
        'issues': [],
    }

    if 'strategy(' in pine_code or 'indicator(' in pine_code:
        result['has_strategy'] = True
    if 'strategy.entry' in pine_code:
        result['has_entry'] = True
    if 'strategy.close' in pine_code or 'strategy.exit' in pine_code:
        result['has_close'] = True
    if any(f in pine_code for f in ['ta.sma','ta.ema','ta.rsi','ta.macd','ta.atr']):
        result['has_indicators'] = True
    if 'input(' in pine_code or 'input.' in pine_code:
        result['has_inputs'] = True

    if not result['has_entry']:
        result['issues'].append("No strategy.entry() found")
    if not result['has_indicators']:
        result['issues'].append("No indicators detected")

    return result