(define (domain tabletop)
 (:requirements :equality :typing :negative-preconditions :disjunctive-preconditions)
 (:types
  scene_object - object                         ; Objects the robot can directly interact with.
  immovable - scene_object                      ; Containers, surfaces, etc. Might be articulated.
  movable - scene_object                        ; Things that can be picked up by the robot.

  robot - object
  background - object                           ; Objects the robot cannot directly interact with, like lights.
 )

 (:predicates
  (on ?x - scene_object ?y - scene_object)      ; x is resting on y
  (in ?x - scene_object ?y - scene_object)      ; x is inside y

  (carry ?r - robot ?x - scene_object)          ; robot is holding x
  (free ?r - robot)                             ; robot gripper is empty

  (openable ?x - scene_object)                  ; x can transition between open and closed
  (open ?x - scene_object)
  (closed ?x - scene_object)
  (no_surface ?x - scene_object)                ; object cannot have things placed on it. For example, a drawer
                                                ;   in a desk -- the top surface is just the desktop; things can
                                                ;   be placed in the drawer but not on it.

  (switchable ?x)                               ; x can be turned on / off (lamp, led)
  (turned_on ?x)
  (turned_off ?x)

  (slidable ?x - scene_object)                  ; x can be slid left / right (slider door)
  (at_left ?x - scene_object)                   ; left / right are with respect to the robot
  (at_right ?x - scene_object)
 )

 (:action pickup_from ; Pick up movable object x from in/on z.
  :parameters (?x - movable ?r - robot ?z - scene_object)
  :precondition
  (and
   (free ?r)
   (or  ; The object must currently rest on something or be accessible inside something.
    (on ?x ?z)
    (and
     (in ?x ?z)
     (open ?z)
    )
   )
  )
  :effect
  (and
   (carry ?r ?x)
   (not (free ?r))
   (not (on ?x ?z))
   (not (in ?x ?z))
  )
 )

 (:action place_on ; Place held object x onto support z, including stacking on another block.
  :parameters (?x - movable ?r - robot ?z - scene_object)
  :precondition
  (and
   (carry ?r ?x)
   (not (no_surface ?z))
  )
  :effect
  (and
   (not (carry ?r ?x))
   (free ?r)
   (on ?x ?z)
  )
 )

 (:action place_in ; Place held object x into open container z.
  :parameters (?x - movable ?r - robot ?z - scene_object)
  :precondition
  (and
   (carry ?r ?x)
   (open ?z)
   (not (= ?x ?z))
  )
  :effect
  (and
   (not (carry ?r ?x))
   (free ?r)
   (in ?x ?z)
  )
 )

 (:action open ; Open articulated object x, such as a drawer.
  :parameters (?x - scene_object ?r - robot)
  :precondition
  (and
   (free ?r)
   (openable ?x)
   (closed ?x)
  )
  :effect
  (and
   (not (closed ?x))
   (open ?x)
  )
 )

 (:action close ; Close articulated object x, such as a drawer.
  :parameters (?x - scene_object ?r - robot)
  :precondition
  (and
   (free ?r)
   (openable ?x)
   (open ?x)
  )
  :effect
  (and
   (not (open ?x))
   (closed ?x)
  )
 )

 (:action turn_on ; Turn on switchable object x, such as the LED or lightbulb.
                  ; The LED is green, and is toggled by pushing a button.
                  ; The lightbulb is yellow, and is toggled by moving a switch.
  :parameters (?x ?r - robot)
  :precondition
  (and
   (free ?r)
   (switchable ?x)
   (turned_off ?x)
  )
  :effect
  (and
   (turned_on ?x)
   (not (turned_off ?x))
  )
 )

 (:action turn_off ; Turn off switchable object x, such as the LED or lightbulb.
                   ; The LED is green, and is toggled by pushing a button.
                   ; The lightbulb is yellow, and is toggled by moving a switch.
  :parameters (?x ?r - robot)
  :precondition
  (and
   (free ?r)
   (switchable ?x)
   (turned_on ?x)
  )
  :effect
  (and
   (turned_off ?x)
   (not (turned_on ?x))
  )
 )

 (:action move_slider_left ; Slide slidable object x to the left to expose the right compartment.
  :parameters (?x - scene_object ?r - robot)
  :precondition
  (and
   (free ?r)
   (slidable ?x)
   (not (at_left ?x))
  )
  :effect
 (and
   (at_left ?x)
   (not (at_right ?x))
  )
 )

 (:action move_slider_right ; Slide slidable object x to the right to expose the left compartment.
  :parameters (?x - scene_object ?r - robot)
  :precondition
  (and
   (free ?r)
   (slidable ?x)
   (not (at_right ?x))
  )
  :effect
  (and
   (at_right ?x)
   (not (at_left ?x))
  )
 )

 (:action push_left ; Push movable object x leftward along support z without grasping it.
  :parameters (?x - movable ?r - robot)
  :precondition
  (free ?r)
  :effect
  (free ?r)
 )

 (:action push_right ; Push movable object x rightward along support z without grasping it.
  :parameters (?x - movable ?r - robot)
  :precondition
  (free ?r)
  :effect
  (free ?r)
 )

 (:action push_into ; Push movable object x from src into open container dst without grasping it.
  :parameters (?x - movable ?r - robot ?src - scene_object ?dst - immovable)
  :precondition
  (and
   (free ?r)
   (on ?x ?src)
   (open ?dst)
   (not (= ?src ?dst))
  )
  :effect
  (and
   (not (on ?x ?src))
   (in ?x ?dst)
  )
 )
 (:action push_onto ; Push movable object x off of src and onto dst without grasping it.
                    ; For example, pushing a block off a tower onto the table.
  :parameters (?x - movable ?r - robot ?src - scene_object ?dst - immovable)
  :precondition
  (and
   (free ?r)
   (on ?x ?src)
   (open ?dst)
   (not (= ?src ?dst))
  )
  :effect
  (and
   (not (on ?x ?src))
   (on ?x ?dst)
  )
 )


 (:action turn_left ; Turn held rotatable object x to the left without modeling orientation state.
  :parameters (?x - movable ?r - robot)
  :precondition
  (and
   (carry ?r ?x)
  )
  :effect
  (and
   ; The held-object relation is preserved because the turn changes no symbolic state we track.
   (carry ?r ?x)
   (not (free ?r))
  )
 )

 (:action turn_right ; Turn held rotatable object x to the right without modeling orientation state.
  :parameters (?x - movable ?r - robot)
  :precondition
  (and
   (carry ?r ?x)
  )
  :effect
  (and
   ; This is intentionally a no-op on object state beyond preserving that the robot still holds x.
   (carry ?r ?x)
   (not (free ?r))
  )
 )
)
