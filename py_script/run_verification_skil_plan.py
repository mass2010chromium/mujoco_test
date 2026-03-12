#!/usr/bin/env python3
"""
Main test script for the VLA subtask verification framework.

Tests the first-layer verification pipeline by:
  1. Constructing a scene graph from an initial scene image (VLM).
  2. Loading pre-generated subtask samples from pi0.5.
  3. Running each subtask through the verification pipeline.
  4. Reporting which subtasks are accepted/rejected and comparing
     against known-false labels.

Usage:
  export OPENROUTER_API_KEY="sk-or-v1-..."
  python run_verification.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from vla_verify.verifier import VLAVerifier
from vla_verify.scene_graph import TaskSceneGraph
from vla_verify.pddl_parsing import setup_pddl_simulation

from vlm_interfaces import *

def load_plan_samples(path: Path):
    """Load plan samples (one plan per non-empty line)."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(line)
    return samples


def get_unique_samples(samples):
    """Deduplicate while preserving order."""
    seen = set()
    unique = []
    for sample in samples:
        if sample not in seen:
            seen.add(sample)
            unique.append(sample)
    return unique


def main():
    # ── Configuration ────────────────────────────────────────────────
    # IMAGE_PATH = SCRIPT_DIR / "trial_imgs_0" / "frame_agentview10.png"
    IMAGE_PATH = SCRIPT_DIR / "scene_graph_test.png"                    # initial scene image
    #IMAGE_PATH = SCRIPT_DIR / "../not_libero.png"                    # initial scene image

    PLAN_PATH = SCRIPT_DIR / "scene_graph_test_plan.txt"                    # one plan per line
    PDDL_PATH = SCRIPT_DIR / "pddl" / "libero_spatial_domain.pddl"

    if not IMAGE_PATH.is_file():
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        sys.exit(1)
    if not PLAN_PATH.is_file():
        print(f"ERROR: Plan samples not found: {PLAN_PATH}")
        sys.exit(1)

    print("[STEP 0] Loading LLM interfaces...")

    llm_interface, vlm_interface = get_openrouter_interfaces()
    #llm_interface, vlm_interface = get_ollama_interfaces()
    #llm_interface, vlm_interface = get_r4b_interfaces()

    print("=" * 70)
    print("VLA Subtask Verification Framework -- First Layer (Scene Graph)")
    print("=" * 70)

    # ── Step 1: Construct scene graph ────────────────────────────────
    print("\n[STEP 1] Constructing scene graph from image...")

    pddl_domain_text = open(PDDL_PATH).read()
    sanity_problem = (
        "(define (problem sanity_check)\n"
        " (:domain tabletop)\n"
        " (:objects\n"
        "  robot_0 - robot\n"
        "  table_0 - scene_object\n"
        " )\n"
        " (:init\n"
        "  (free robot_0)\n"
        " )\n"
        ")\n"
    )
    try:
        setup_pddl_simulation(sanity_problem, pddl_domain_text)
    except Exception as e:
        print(f"ERROR: Failed to parse PDDL domain: {PDDL_PATH}")
        print(f"  Parser exception: {type(e).__name__}: {e}")
        if "(forall" in pddl_domain_text:
            print("  Hint: current pddlsim parser does not support quantified preconditions like `(forall ...)`.")
        sys.exit(2)

    scene_graph = TaskSceneGraph(pddl_domain_text, vlm_interface)
    # Save some tokens for now
    #llm_out = open('tmp').read()
    #scene_graph.construct_from_pddl(mediapy.read_image(IMAGE_PATH), llm_out, ground=False)
    scene_graph.read_image(cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB), ground=True)
    if scene_graph.simulator is None:
        print("ERROR: Scene graph PDDL parsing failed; simulator was not created.")
        print("  Check the earlier `PDDL Parse Error!` block for the invalid generated problem PDDL.")
        print("  The script stops here to avoid the downstream `NoneType.state` crash.")
        sys.exit(2)
    print(scene_graph.summary())

    verifier = VLAVerifier(scene_graph, llm_interface)

    # ── Step 2: Load plan samples ─────────────────────────────────
    print("\n[STEP 2] Loading plan samples...")
    all_samples = load_plan_samples(PLAN_PATH)
    test_samples = get_unique_samples(all_samples)
    print(f"  Total lines: {len(all_samples)}")
    print(f"  Unique plans: {len(test_samples)}")

    # ── Step 3: Verify each plan ──────────────────────────────────
    print(f"\n[STEP 3] Running verification...")
    print("=" * 70)

    results = []
    for i, plan in enumerate(test_samples):
        print(f"\n--- Plan {i+1}/{len(test_samples)} ---")
        print(plan)

        result = verifier.verify_skill_plan(plan)
        results.append((plan, result))

        if result["feasible"]:
            print("Plan result: FEASIBLE")
        else:
            print("Plan result: INFEASIBLE")
            print(
                f"  Failed at step {result['failed_step']}: "
                f"{result['failure_reason']}"
            )

    # ── Step 4: Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    total = len(results)
    feasible = sum(1 for _, r in results if r["feasible"])
    infeasible = total - feasible

    print(f"\nTotal unique plans tested: {total}")
    print(f"  Feasible:   {feasible}")
    print(f"  Infeasible: {infeasible}")

    print("\n--- Per-plan Results ---")
    for i, (plan, result) in enumerate(results):
        status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
        detail = ""
        if not result["feasible"]:
            detail = (
                f" (failed at step {result['failed_step']}: "
                f"{result['failure_reason']})"
            )
        print(f"  {i+1:2d}. [{status}] {plan}{detail}")

    # Save full results as JSON
    results_save = SCRIPT_DIR / "verification_plan_results.json"
    results_json = []
    for plan, result in results:
        step_results = []
        for step_result in result["step_results"]:
            step_results.append({
                "subtask": step_result.subtask,
                "accepted": step_result.accepted,
                "reason": step_result.reasoning,
                "action": step_result.grounded_action,
            })

        results_json.append({
            "plan": plan,
            "steps": result["steps"],
            "feasible": result["feasible"],
            "failed_step": result["failed_step"],
            "failure_reason": result["failure_reason"],
            "applied_actions": result["applied_actions"],
            "step_results": step_results,
        })
    with open(results_save, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Full results saved to: {results_save}")


if __name__ == "__main__":
    main()
