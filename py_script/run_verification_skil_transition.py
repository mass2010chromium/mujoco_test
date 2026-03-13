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
    scene_graph.read_image(cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB), ground=True)
    if scene_graph.simulator is None:
        print("ERROR: Scene graph PDDL parsing failed; simulator was not created.")
        print("  Check the earlier `PDDL Parse Error!` block for the invalid generated problem PDDL.")
        print("  The script stops here to avoid the downstream `NoneType.state` crash.")
        sys.exit(2)
    print(scene_graph.summary())

    verifier = VLAVerifier(scene_graph, llm_interface)

    # ── Step 3: Verify plan ──────────────────────────────────
    print(f"\n[STEP 3] Running verification...")
    print("=" * 70)

    plan = '1. OPEN(top drawer of the cabinet) 2. PICK(silver bowl) 3. PLACE(silver bowl, top drawer of the cabinet, inside) 4. CLOSE(top drawer of the cabinet)'

    print(plan)
    result = verifier.verify_skill_plan(plan)
    if result["feasible"]:
        print("Plan result: FEASIBLE")
    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"  [{status}] {plan}")

    # ── Step 4: Verify skill transition ──────────────────────────────────
    print(f"\n[STEP 4] Running skill transition verification...")
    print("=" * 70)
    verifier.set_skill_plan(plan)

    result = verifier.verify_skill_transition('OPEN(top drawer of the cabinet)')
    print(result["feasible"], result["failure_reason"])
    
    result = verifier.verify_skill_transition('PICK(silver bowl)')
    print(result["feasible"], result["failure_reason"])

    result = verifier.verify_skill_transition('PLACE(silver bowl, top drawer of the cabinet, inside)')
    print(result["feasible"], result["failure_reason"])

    result = verifier.verify_skill_transition('CLOSE(top drawer of the cabinet)')
    print(result["feasible"], result["failure_reason"])


if __name__ == "__main__":
    main()
