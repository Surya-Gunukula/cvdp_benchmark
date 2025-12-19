#!/usr/bin/env python3
"""
Pin cocotb to 1.9.x in CVDP benchmark dataset.

Handles both:
- Dockerfiles: adds 'RUN pip3 install cocotb>=1.9,<2.0' after FROM
- docker-compose.yml: prepends pip install to the command

Usage:
    python fix_cocotb_pin19.py datasets/input.jsonl -o output.jsonl
    python fix_cocotb_pin19.py datasets/input.jsonl --dry-run
"""

import json
import argparse
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple

COCOTB_PIN = "RUN pip3 install 'cocotb>=1.9,<2.0'"
COCOTB_PIN_CMD = "pip3 install 'cocotb>=1.9,<2.0' && "


def fix_dockerfile(content: str) -> Tuple[str, bool]:
    """Add cocotb pin to Dockerfile if it uses __OSS_SIM_IMAGE__."""
    
    # Skip if already pinned
    if 'cocotb>=1.9' in content or 'cocotb<2.0' in content:
        return content, False
    
    # Only modify if using OSS_SIM_IMAGE (simulation image with cocotb)
    if '__OSS_SIM_IMAGE__' not in content:
        return content, False
    
    # Add cocotb pin after FROM line
    lines = content.split('\n')
    new_lines = []
    modified = False
    
    for line in lines:
        new_lines.append(line)
        # Add pin right after FROM __OSS_SIM_IMAGE__
        if line.strip().startswith('FROM') and '__OSS_SIM_IMAGE__' in line:
            new_lines.append(COCOTB_PIN)
            modified = True
    
    return '\n'.join(new_lines), modified


def fix_docker_compose(content: str) -> Tuple[str, bool]:
    """Add cocotb pin to docker-compose.yml command."""
    
    # Skip if already pinned
    if 'cocotb>=1.9' in content or 'cocotb<2.0' in content:
        return content, False
    
    # Only modify if using OSS_SIM_IMAGE
    if '__OSS_SIM_IMAGE__' not in content:
        return content, False
    
    modified = False
    modified_content = content
    
    # Pattern 1a: command : /bin/sh -c "..." or sh -c "..."
    pattern1a = r'(command\s*:\s*(?:/bin/)?sh\s+-c\s+["\'])([^"\']+)(["\'])'
    
    def add_pin_shell(match):
        prefix = match.group(1)
        cmd = match.group(2)
        suffix = match.group(3)
        if 'pip' in cmd and 'cocotb' in cmd:
            return match.group(0)
        return f"{prefix}{COCOTB_PIN_CMD}{cmd}{suffix}"
    
    new_content = re.sub(pattern1a, add_pin_shell, modified_content)
    if new_content != modified_content:
        modified_content = new_content
        modified = True
    
    # Pattern 1b: multi-line YAML with > (command : >\n  sh -c "...")
    pattern1b = r'(command\s*:\s*>\s*\n\s*(?:/bin/)?sh\s+-c\s+["\'])([^"\']+)(["\'])'
    
    new_content = re.sub(pattern1b, add_pin_shell, modified_content)
    if new_content != modified_content:
        modified_content = new_content
        modified = True
    
    # Pattern 2: command : pytest ... (direct command, no shell)
    # Wrap in /bin/sh -c with pip install
    pattern2 = r'(command\s*:\s*)(pytest\s+[^\n]+)'
    
    def wrap_in_shell(match):
        indent = match.group(1)
        cmd = match.group(2)
        if 'pip' in cmd and 'cocotb' in cmd:
            return match.group(0)
        return f'{indent}/bin/sh -c "{COCOTB_PIN_CMD}{cmd}"'
    
    new_content = re.sub(pattern2, wrap_in_shell, modified_content)
    if new_content != modified_content:
        modified_content = new_content
        modified = True
    
    return modified_content, modified


def fix_harness_files(harness: Dict, problem_id: str) -> Tuple[Dict, List[str]]:
    """Fix Dockerfiles and docker-compose.yml in harness."""
    if 'files' not in harness:
        return harness, []
    
    all_changes = []
    files = harness['files'].copy()
    
    for filename, content in files.items():
        if not isinstance(content, str):
            continue
        
        # Handle Dockerfile
        if 'Dockerfile' in filename:
            content, modified = fix_dockerfile(content)
            if modified:
                files[filename] = content
                all_changes.append(f"  {filename}: added cocotb 1.9.x pin (Dockerfile)")
        
        # Handle docker-compose.yml
        elif filename == 'docker-compose.yml':
            content, modified = fix_docker_compose(content)
            if modified:
                files[filename] = content
                all_changes.append(f"  {filename}: added cocotb 1.9.x pin (command)")
    
    harness['files'] = files
    return harness, all_changes


def process_jsonl_file(input_path: str, output_path: str, dry_run: bool = False) -> Dict:
    """Process JSONL file and add cocotb pins."""
    
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
                
                if 'harness' in problem:
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
    parser = argparse.ArgumentParser(description='Pin cocotb to 1.9.x in CVDP benchmark datasets')
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('--output', '-o', help='Output JSONL file')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Preview changes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show details')
    
    args = parser.parse_args()
    
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_cocotb19{input_path.suffix}")
    
    stats = process_jsonl_file(args.input, output_path, args.dry_run)
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total problems: {stats['total_problems']}")
    print(f"Modified problems: {stats['modified_problems']}")
    
    if args.verbose and stats['changes']:
        print("\nChanges made:")
        for change in stats['changes']:
            print(change)
    
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors']:
            print(f"  {error}")


if __name__ == '__main__':
    main()

