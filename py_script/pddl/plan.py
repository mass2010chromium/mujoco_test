"""
Partially ripped from tutorial: https://github.com/galk-research/pddlsim/wiki/Tutorial

Requires: `uv pip install pddlsim`
"""
import textwrap

from pddlsim.parser import parse_domain, parse_problem
from pddlsim.simulation import Simulation

with open("calvin_domain.pddl") as domain_file, open("calvin_problem.pddl") as problem_file:
    domain = parse_domain(domain_file.read())             # Returns a `Domain` object
    problem = parse_problem(problem_file.read(), domain)

simulator = Simulation.from_domain_and_problem(domain, problem)

print("PDDL interactive")
print(f"Problem name: {problem.name}")
print("Problem goal(s):")
print(repr(problem.raw_problem.goals_section))
print(simulator)
#while not simulator.is_solved():
while True:
    print("(:objects")
    print(textwrap.indent('\n'.join(repr(s) for s in simulator.problem.objects_section), "  "))
    print(")")
    print("(:state")
    print(textwrap.indent('\n'.join(repr(s) for s in simulator.state), "  "))
    print(")")
    actions = list(simulator.get_grounded_actions())
    print("Actions:")
    for i, action in enumerate(actions):
        print(i, action)
    i = input("Pick an action by entering a number: ")
    try:
        i = int(i)
    except:
        print("Failed to parse:", i)
        continue
    if i < 0 or i >= len(actions):
        print("Invalid action index:", i)
        continue
    simulator.apply_grounded_action(actions[i])
print("Congrats! You are planner")
