#!/usr/bin/env python3
"""
Analyze benchmark failures using GPT-5.2 to classify them as:
- MODEL_FAILURE: Model generated bad RTL (syntax, lint, logic errors, iverilog compile failures)
- HARNESS_FAILURE: Test infrastructure issues (cocotb API, Docker, Python deps)
- UNKNOWN_FAILURE: Log doesn't contain enough info to determine cause
- PASS: Test actually passed
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import config
from gpt_instance import GPT_Instance

ANALYSIS_PROMPT = """You are analyzing a benchmark test failure log. Your task is to classify the failure.

CLASSIFICATION TYPES:
1. MODEL_FAILURE - The MODEL (LLM) generated bad code. This includes:
   - Functionally incorrect RTL (simulation ran but assertions failed)
   - Syntax errors (iverilog/verilator compilation failed with non-zero exit)
   - iverilog returning non-zero exit status (e.g., "CalledProcessError", "returned non-zero exit status 2")
   - Lint errors (verilator lint warnings treated as errors: WIDTHTRUNC, LATCH, etc.)
   - Wrong module names, port names, or signal widths
   - "Failed to optimize" from Verilator (model's code couldn't be optimized)
   - Any issue where the MODEL's generated RTL code is the problem

2. HARNESS_FAILURE - The TEST HARNESS/INFRASTRUCTURE failed, NOT the model's code:
   - cocotb import errors or API version incompatibilities (e.g., "module 'cocotb' has no attribute 'utils'")
   - cocotb TypeError like "LogicObject not subscriptable" or "unsupported format string"
   - Docker build/setup failures (not RTL compilation)
   - Missing Python dependencies
   - Test timeout without running
   - pytest/cocotb framework crashes
   - File system or permission errors

3. UNKNOWN_FAILURE - The log does NOT contain enough information to determine the cause:
   - Log is empty or truncated with no error messages
   - Log shows test starting but stops with no clear error output
   - Cannot determine if the issue is MODEL or HARNESS related

4. PASS - The test actually passed (no failure)

KEY DISTINCTION:
- If the error is in the MODEL's generated RTL/Verilog code -> MODEL_FAILURE
- If the error is in the test harness Python code or infrastructure -> HARNESS_FAILURE
- Lint errors (WIDTHTRUNC, LATCH, etc.) are MODEL_FAILURE because the model should generate lint-clean code
- Syntax errors are MODEL_FAILURE because the model generated invalid code
- "Failed to optimize" from Verilator is MODEL_FAILURE (model's code issue)

Analyze this test log and respond with ONLY a JSON object:
{
    "classification": "MODEL_FAILURE" | "HARNESS_FAILURE" | "UNKNOWN_FAILURE" | "PASS",
    "reason": "one sentence explanation"
}

TEST LOG:
```
%s
```
"""


def analyze_report(gpt: GPT_Instance, report_content: str) -> Dict[str, Any]:
    """Analyze a single report using GPT-5.2"""
    # Truncate if too long (keep first and last parts)
    max_len = 12000
    if len(report_content) > max_len:
        half = max_len // 2
        report_content = report_content[:half] + "\n\n... [TRUNCATED] ...\n\n" + report_content[-half:]
    
    prompt = ANALYSIS_PROMPT % report_content
    
    text = ""
    try:
        response = gpt.prompt(prompt, files=["analysis.json"], category=9)
        
        if response is None:
            return {"classification": "ERROR", "reason": "GPT returned None"}
        
        # Parse the response - handle various formats
        if isinstance(response, tuple):
            # ModelHelpers.parse_model_response returns (dict, bool)
            response = response[0]
        
        if isinstance(response, dict):
            if "direct_text" in response:
                text = response["direct_text"]
            elif "response" in response:
                text = response["response"]
            elif "classification" in response:
                # Already parsed!
                return response
            else:
                text = json.dumps(response)
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)
        
        if not text or text.strip() == "":
            return {"classification": "ERROR", "reason": "Empty response from GPT"}
        
        # Extract JSON from response
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1].strip()
        
        # Try to find JSON object in text
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
        
        result = json.loads(text)
        return result
        
    except json.JSONDecodeError as e:
        return {"classification": "ERROR", "reason": f"JSON parse error: {e}", "raw": text[:200] if text else "empty"}
    except Exception as e:
        return {"classification": "ERROR", "reason": str(e)}


def find_reports(work_dir: Path) -> Dict[str, list]:
    """Find all report files in the work directory"""
    reports = {}
    
    for problem_dir in work_dir.iterdir():
        if not problem_dir.is_dir():
            continue
        if problem_dir.name.startswith('.'):
            continue
        if problem_dir.name in ['prompt_response.jsonl', 'raw_result.json', 'report.json', 'report.txt', 'run.log']:
            continue
            
        reports_dir = problem_dir / "reports"
        if not reports_dir.exists():
            continue
            
        problem_reports = []
        for report_file in sorted(reports_dir.glob("*.txt")):
            problem_reports.append(report_file)
        
        if problem_reports:
            reports[problem_dir.name] = problem_reports
    
    return reports


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark failures using GPT-5.2")
    parser.add_argument("work_dir", help="Path to work directory (e.g., work/work-gpt_5_2_cocotb_fix)")
    parser.add_argument("-o", "--output", default="failure_analysis.json", help="Output JSON file")
    parser.add_argument("--model", default="gpt-5.2", help="Model to use for analysis")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems to analyze")
    parser.add_argument("--only-failed", action="store_true", help="Only analyze problems that have failures in report.json")
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        print(f"Error: {work_dir} does not exist")
        sys.exit(1)
    
    # Initialize GPT
    print(f"Initializing {args.model}...")
    gpt = GPT_Instance(
        context="You are an expert at analyzing test failures and classifying them.",
        model=args.model
    )
    
    # Find all reports
    print(f"Scanning {work_dir} for reports...")
    reports = find_reports(work_dir)
    print(f"Found {len(reports)} problems with reports")
    
    # Load report.json to check which problems failed
    failed_problems = set()
    report_json_path = work_dir / "report.json"
    if report_json_path.exists():
        with open(report_json_path) as f:
            report_data = json.load(f)
            # Structure: {"cid002": {"logs": [{"id": "problem_id", "log": "/path/..."}]}}
            for category, category_data in report_data.items():
                if isinstance(category_data, dict) and "logs" in category_data:
                    for log_entry in category_data["logs"]:
                        if isinstance(log_entry, dict) and "id" in log_entry:
                            # Extract problem name from id (e.g., cvdp_copilot_xxx_0001 -> cvdp_copilot_xxx)
                            problem_id = log_entry["id"]
                            # The directory name format
                            parts = problem_id.rsplit("_", 1)
                            if len(parts) == 2 and parts[1].isdigit():
                                problem_dir = parts[0]
                            else:
                                problem_dir = problem_id
                            failed_problems.add(problem_dir)
    
    print(f"Found {len(failed_problems)} failed problems in report.json")
    
    # Analyze each problem
    results = {
        "summary": {
            "total_problems": 0,
            "model_failures": 0,
            "harness_failures": 0,
            "unknown_failures": 0,
            "passes": 0,
            "errors": 0
        },
        "problems": {}
    }
    
    problems_to_analyze = list(reports.keys())
    if args.only_failed:
        problems_to_analyze = [p for p in problems_to_analyze if p in failed_problems]
        print(f"Filtering to {len(problems_to_analyze)} failed problems")
    
    if args.limit:
        problems_to_analyze = problems_to_analyze[:args.limit]
        print(f"Limiting to {args.limit} problems")
    
    for i, problem_name in enumerate(problems_to_analyze):
        report_files = reports[problem_name]
        print(f"\n[{i+1}/{len(problems_to_analyze)}] Analyzing {problem_name}...")
        
        problem_results = []
        
        for report_file in report_files:
            harness_id = report_file.stem
            print(f"  - Harness {harness_id}...", end=" ", flush=True)
            
            with open(report_file) as f:
                content = f.read()
            
            # Quick check if it passed
            if "TESTS=" in content and "FAIL=0" in content:
                result = {"classification": "PASS", "reason": "All tests passed"}
            else:
                result = analyze_report(gpt, content)
            
            print(f"{result['classification']}")
            
            problem_results.append({
                "harness_id": harness_id,
                "classification": result.get("classification", "ERROR"),
                "reason": result.get("reason", "Unknown")
            })
            
            # Update summary
            classification = result.get("classification", "ERROR")
            if classification == "MODEL_FAILURE":
                results["summary"]["model_failures"] += 1
            elif classification == "HARNESS_FAILURE":
                results["summary"]["harness_failures"] += 1
            elif classification == "UNKNOWN_FAILURE":
                results["summary"]["unknown_failures"] += 1
            elif classification == "PASS":
                results["summary"]["passes"] += 1
            else:
                results["summary"]["errors"] += 1
        
        results["problems"][problem_name] = problem_results
        results["summary"]["total_problems"] += 1
        
        # Save intermediate results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total problems analyzed: {results['summary']['total_problems']}")
    print(f"Model failures (bad RTL/syntax/lint): {results['summary']['model_failures']}")
    print(f"Harness failures (test infra issues): {results['summary']['harness_failures']}")
    print(f"Unknown failures (unclear logs): {results['summary']['unknown_failures']}")
    print(f"Passes: {results['summary']['passes']}")
    print(f"Errors (analysis failed): {results['summary']['errors']}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

