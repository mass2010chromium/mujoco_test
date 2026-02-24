"""
PDDL parsing utilities.

Copied from: https://github.com/galk-research/pddlsim/blob/main/src/pddlsim/parser.py

This is a bit of a monkey patch. Maybe we should consider forking pddlsim
if we need more functionality or stability.
"""
import os

from lark import Lark, Token, v_args
from pddlsim.ast import (
    ActionFallibilitiesSection,
    Condition,
    GoalsSection,
    Identifier,
    InitializationSection,
    Object,
    ObjectsSection,
    RequirementsSection,
    RevealablesSection,

    RawProblem,
    Problem,
    Predicate
)
from pddlsim.parser import _PDDLTransformer, parse_domain
from pddlsim.simulation import Simulation

GRAMMAR_FILE = os.path.join(os.path.dirname(__file__), 'pddl_problem.lark')

class _DummyGoal(Predicate):
    """Fool the pddlsim machinery into becoming a pure simulator!"""
    def __init__(self):
        super().__init__(name="dummy_goal", assignment=tuple())

    def _validate(self, params, objects, domain):
        """No need to validate anything. :) """
        pass

    def __repr__(self):
        # This can't just be anything. See pddl_problem.lark
        return "(none)"

@v_args(inline=True)
class _NoGoalTransformer(_PDDLTransformer):
    """
    Override problem parsing to make last argument (goals_section)
    optional, defaulting to a dummy goal.
    """
    def dummy_goal_section(self) -> None:
        return None

    def problem(
        self,
        name: Identifier,
        used_domain_name: Identifier,
        requirements_section: RequirementsSection | None,
        objects_section: ObjectsSection | None,
        action_fallibilities_section: ActionFallibilitiesSection | None,
        revealables_section: RevealablesSection | None,
        initialization_section: InitializationSection | None,
        goals_section: GoalsSection | Condition[Object] | None,
    ) -> RawProblem:
        return RawProblem.from_raw_parts(
            name,
            used_domain_name,
            requirements_section
            if requirements_section
            else RequirementsSection(),
            objects_section if objects_section else ObjectsSection(),
            action_fallibilities_section,
            revealables_section,
            initialization_section
            if initialization_section
            else InitializationSection(),
            goals_section if goals_section else _DummyGoal(),
        )

_GOAL_OPTIONAL_PDDL_PARSER = Lark(
    open(GRAMMAR_FILE, 'r').read(),
    parser="lalr",
    cache=True,
    transformer=_NoGoalTransformer(),
    start=["domain", "problem"],
)

def setup_pddl_simulation(problem_desc: str, domain_desc: str):
    """
    Spin up a PDDL simulation given the domain and problem strings.

    Should the domain be pre-parsed?

    @param problem_desc     Problem description (PDDL problem file)
    @param domain_desc      Domain description (PDDL domain file)

    @return tuple (problem, domain, simulation):
        problem: pddlsim.ast.Problem object, for inspecting problem internals
        domain: pddlsim.ast.Domain object, for inspecting domain internals
        simulation: pddlsim.simulator.Simulation instance for interaction
    """
    domain = parse_domain(domain_desc)
    raw_problem = _GOAL_OPTIONAL_PDDL_PARSER.parse(problem_desc, "problem")
    problem = Problem(raw_problem, domain)
    return problem, domain, Simulation.from_domain_and_problem(domain, problem)

if __name__ == "__main__":
    test_problem = open("../tmp").read()
    domain_desc = open("../pddl/pick_place_domain.pddl").read()

    problem, domain, simulator = setup_pddl_simulation(test_problem, domain_desc)
    print((list(simulator.state)[0]).name.value)
    print([x.value for x in (list(simulator.state)[0]).assignment])
    while not simulator.is_solved():
        actions = list(simulator.get_grounded_actions())
        print("Actions:")
        for i, action in enumerate(actions):
            print(i, action, action.name)
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
