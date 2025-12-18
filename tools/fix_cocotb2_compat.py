#!/usr/bin/env python3
"""
Fix cocotb 2.x compatibility issues in CVDP benchmark dataset files.

Transforms tests written for cocotb 1.x to work with cocotb 2.x:
1. cocotb.runner -> cocotb_tools.runner
2. dut.signal[index] -> dut.signal.value[index]

Creates a NEW modified dataset file (does not overwrite original).

Usage:
    python fix_cocotb2_compat.py datasets/input.jsonl                    # Creates input_cocotb2.jsonl
    python fix_cocotb2_compat.py datasets/input.jsonl -o output.jsonl    # Custom output path
    python fix_cocotb2_compat.py datasets/input.jsonl --dry-run          # Preview changes
"""

import json
import argparse
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Transformations: cocotb 1.x -> cocotb 2.x
IMPORT_REPLACEMENTS = [
    # Runner import: cocotb.runner -> cocotb_tools.runner
    (r'from cocotb\.runner import', 'from cocotb_tools.runner import'),
    (r'import cocotb\.runner', 'import cocotb_tools.runner'),
    # Binary import: cocotb.binary -> cocotb.types (BinaryValue -> LogicArray)
    (r'from cocotb\.binary import BinaryValue', 'from cocotb.types import LogicArray'),
    (r'from cocotb\.binary import', 'from cocotb.types import'),
    (r'import cocotb\.binary', 'import cocotb.types'),
    # TestFailure removed in cocotb 2.0 - remove the import line entirely
    (r'from cocotb\.result import TestFailure\n?', ''),
    (r'from cocotb\.result import.*,\s*TestFailure', 'from cocotb.result import'),
    # cocotb.utils renamed to cocotb.sim_time_utils in 2.x
    (r'import cocotb\.utils\n', 'from cocotb import sim_time_utils\n'),
    (r'from cocotb\.utils import', 'from cocotb.sim_time_utils import'),
]

API_REPLACEMENTS = [
    # LogicObject subscript: dut.signal[index] -> dut.signal.value[index]
    # Pattern matches dut.signal[number] or dut.signal[variable] but NOT dut.signal.value[...]
    (r'(dut\.\w+)\[(\d+)\]', r'\1.value[\2]'),  # numeric index
    (r'(dut\.\w+)\[([a-zA-Z_]\w*)\]', r'\1.value[\2]'),  # variable index like [i], [idx]
    # BinaryValue -> LogicArray (class name replacement)
    (r'\bBinaryValue\b', 'LogicArray'),
    # Remove redundant .value after subscript: .value[x].value -> .value[x]
    # In cocotb 2.x, subscripting already returns the value, no need for .value
    (r'\.value\[(\w+)\]\.value\b', r'.value[\1]'),
    # Remove deprecated @cocotb.coroutine decorator (native async/await works in 2.x)
    # Pattern: @cocotb.coroutine followed by newline and async def
    (r'@cocotb\.coroutine\n(async def)', r'\1'),
    # TestFailure removed in cocotb 2.0 - use AssertionError instead
    (r'raise TestFailure\(', 'raise AssertionError('),
    (r'cocotb\.result\.TestFailure', 'AssertionError'),
    # cocotb.utils -> sim_time_utils (API usage)
    (r'cocotb\.utils\.', 'sim_time_utils.'),
    # LogicArray format string fix: {dut.signal.value:#x} -> {int(dut.signal.value):#x}
    # In cocotb 2.x, LogicArray doesn't support format specifiers directly, need int() wrapper
    (r'\{(dut\.\w+\.value)(:[\#0-9]*[xXbBoOdD][^}]*)\}', r'{int(\1)\2}'),
    # Also handle indexed: {dut.signal.value[i]:#x} -> {int(dut.signal.value[i]):#x}
    (r'\{(dut\.\w+\.value\[\w+\])(:[\#0-9]*[xXbBoOdD][^}]*)\}', r'{int(\1)\2}'),
]


def apply_replacements(content: str, replacements: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
    """Apply regex replacements to content, return modified content and list of changes made."""
    changes = []
    modified = content
    
    for pattern, replacement in replacements:
        matches = re.findall(pattern, modified)
        if matches:
            count = len(matches) if isinstance(matches[0], str) else len(matches)
            modified = re.sub(pattern, replacement, modified)
            changes.append(f"{pattern} -> {replacement} ({count}x)")
    
    return modified, changes


def fix_harness_files(harness: Dict, problem_id: str) -> Tuple[Dict, List[str]]:
    """Fix cocotb compatibility issues in harness files."""
    if 'files' not in harness:
        return harness, []
    
    all_changes = []
    files = harness['files'].copy()
    
    for filename, content in files.items():
        if not isinstance(content, str):
            continue
            
        if filename.endswith('.py'):
            original = content
            file_changes = []
            
            # Apply import replacements
            content, import_changes = apply_replacements(content, IMPORT_REPLACEMENTS)
            file_changes.extend(import_changes)
            
            # Apply API replacements
            content, api_changes = apply_replacements(content, API_REPLACEMENTS)
            file_changes.extend(api_changes)
            
            if content != original:
                files[filename] = content
                if file_changes:
                    all_changes.append(f"  {filename}: {', '.join(file_changes)}")
    
    harness['files'] = files
    return harness, all_changes


def process_jsonl_file(input_path: str, output_path: str, dry_run: bool = False) -> Dict:
    """Process a JSONL file and fix cocotb compatibility issues."""
    
    stats = {
        'total_problems': 0,
        'modified_problems': 0,
        'changes': [],
        'errors': []
    }
    
    modified_lines = []
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                modified_lines.append('')
                continue
            
            try:
                problem = json.loads(line)
                stats['total_problems'] += 1
                
                problem_id = problem.get('id', f'line_{line_num}')
                
                # Fix harness files if present
                if 'harness' in problem:
                    original_harness = json.dumps(problem['harness'])
                    problem['harness'], changes = fix_harness_files(problem['harness'], problem_id)
                    
                    if changes:
                        stats['modified_problems'] += 1
                        stats['changes'].append(f"[{problem_id}]")
                        stats['changes'].extend(changes)
                
                modified_lines.append(json.dumps(problem, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_num}: JSON decode error - {e}")
                modified_lines.append(line)
            except Exception as e:
                stats['errors'].append(f"Line {line_num}: {type(e).__name__} - {e}")
                modified_lines.append(line)
    
    # Write output
    if not dry_run:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines))
            if modified_lines:
                f.write('\n')
        print(f"✅ Written to: {output_path}")
    else:
        print(f"[DRY RUN] Would write to: {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Fix cocotb 2.x compatibility issues in CVDP benchmark datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s datasets/cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl
    -> Creates datasets/cvdp_v1.0.2_nonagentic_code_generation_no_commercial_cocotb2.jsonl

  %(prog)s input.jsonl -o fixed_output.jsonl
    -> Creates fixed_output.jsonl

  %(prog)s input.jsonl --dry-run
    -> Shows what would be changed without writing
"""
    )
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('--output', '-o', help='Output JSONL file (default: input_cocotb2.jsonl)')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Show changes without writing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed changes')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_cocotb2{input_path.suffix}")
    
    # Process file
    stats = process_jsonl_file(args.input, output_path, args.dry_run)
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total problems: {stats['total_problems']}")
    print(f"Modified problems: {stats['modified_problems']}")
    
    if args.verbose and stats['changes']:
        print()
        print("Changes made:")
        for change in stats['changes']:
            print(change)
    
    if stats['errors']:
        print()
        print(f"Errors ({len(stats['errors'])}):")
        for error in stats['errors']:
            print(f"  {error}")
    
    if not args.dry_run and stats['modified_problems'] > 0:
        print()
        print(f"🎉 Fixed {stats['modified_problems']} problems for cocotb 2.x compatibility")
        print(f"   Output: {output_path}")


if __name__ == '__main__':
    main()
