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

from vlm_interfaces import *

def load_subtask_samples(path: Path):
    """
    Load subtask samples from file.

    Lines ending with ", f" are known-false (hallucinated).
    Returns list of (subtask_text, is_labeled_false) tuples.
    """
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(", f"):
                subtask = line[:-3].strip()
                samples.append((subtask, True))
            else:
                samples.append((line, False))
    return samples


def get_unique_samples(samples):
    """Deduplicate while preserving order and label info."""
    seen = set()
    unique = []
    for subtask, is_false in samples:
        key = (subtask, is_false)
        if key not in seen:
            seen.add(key)
            unique.append((subtask, is_false))
    return unique


def main():
    # ── Configuration ────────────────────────────────────────────────
    # IMAGE_PATH = SCRIPT_DIR / "trial_imgs_0" / "frame_agentview10.png"
    #IMAGE_PATH = SCRIPT_DIR / "scene_graph_test.png"                    # initial scene image
    IMAGE_PATH = SCRIPT_DIR / "../not_libero.png"                    # initial scene image

    SUBTASK_PATH = SCRIPT_DIR / "scene_graph_test_subtasks.txt"            # generatedsubtask samples
    # SUBTASK_PATH = SCRIPT_DIR / "subtask_samples.txt"
    PDDL_PATH = SCRIPT_DIR / "pddl" / "pick_place_domain.pddl"

    if not IMAGE_PATH.is_file():
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        sys.exit(1)
    if not SUBTASK_PATH.is_file():
        print(f"ERROR: Subtask samples not found: {SUBTASK_PATH}")
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
    scene_graph = TaskSceneGraph(pddl_domain_text, vlm_interface)
    # Save some tokens for now
    #llm_out = open('tmp').read()
    #scene_graph.construct_from_pddl(mediapy.read_image(IMAGE_PATH), llm_out, ground=False)
    scene_graph.read_image(cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB), ground=True)
    print(scene_graph.summary())

    verifier = VLAVerifier(scene_graph, llm_interface)

    # ── Step 2: Load subtask samples ─────────────────────────────────
    print("\n[STEP 2] Loading subtask samples...")
    all_samples = load_subtask_samples(SUBTASK_PATH)
    test_samples = get_unique_samples(all_samples)

    n_false = sum(1 for _, f in test_samples if f)
    n_unlabeled = len(test_samples) - n_false
    print(f"  Total lines: {len(all_samples)}")
    print(f"  Unique test samples: {len(test_samples)}")
    print(f"    Known-false: {n_false}")
    print(f"    Unlabeled:   {n_unlabeled}")

    # ── Step 3: Verify each subtask ──────────────────────────────────
    print(f"\n[STEP 3] Running verification...")
    print("=" * 70)

    results = []
    for i, (subtask, is_labeled_false) in enumerate(test_samples):
        label = " [KNOWN FALSE]" if is_labeled_false else ""
        print(f"\n--- Test {i+1}/{len(test_samples)}{label} ---")

        result = verifier.verify_subtask(subtask)
        results.append((result, is_labeled_false))

        print(result.summary())

    # ── Step 4: Summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    total = len(results)
    accepted = sum(1 for r, _ in results if r.accepted)
    rejected = total - accepted

    labeled_false = [(r, f) for r, f in results if f]
    caught = sum(1 for r, _ in labeled_false if not r.accepted)
    missed = sum(1 for r, _ in labeled_false if r.accepted)

    unlabeled = [(r, f) for r, f in results if not f]
    accepted_ul = sum(1 for r, _ in unlabeled if r.accepted)
    rejected_ul = sum(1 for r, _ in unlabeled if not r.accepted)

    print(f"\nTotal unique subtasks tested: {total}")
    print(f"  Accepted: {accepted}")
    print(f"  Rejected: {rejected}")

    print(f"\nKnown-False samples (should be REJECTED):")
    print(f"  Correctly rejected: {caught}/{len(labeled_false)}")
    print(f"  Incorrectly accepted: {missed}/{len(labeled_false)}")

    print(f"\nUnlabeled samples (assumed mostly valid):")
    print(f"  Accepted: {accepted_ul}/{len(unlabeled)}")
    print(f"  Rejected: {rejected_ul}/{len(unlabeled)}")

    print("\n--- Per-subtask Results ---")
    for i, (result, is_false) in enumerate(results):
        status = "ACCEPT" if result.accepted else "REJECT"
        label = " [FALSE]" if is_false else ""
        verdict = ""
        if is_false:
            verdict = " OK" if not result.accepted else " MISS"
        print(
            f"  {i+1:2d}. [{status}]{label}{verdict}  "
            f"\"{result.subtask}\""
        )

    # Save full results as JSON
    results_save = SCRIPT_DIR / "verification_results.json"
    results_json = []
    for result, is_false in results:
        results_json.append({
            "subtask": result.subtask,
            "labeled_false": is_false,
            "accepted": result.accepted,
            "reason": result.reasoning,
            "action": result.grounded_action
        })
    with open(results_save, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Full results saved to: {results_save}")


if __name__ == "__main__":
    main()
