(define (domain tabletop)
 (:requirements :equality :typing :negative-preconditions :disjunctive-preconditions)
 (:types
  scene_object
  immovable - scene_object
  movable - scene_object

  robot
 )

 (:predicates
  (on ?x - scene_object ?y - scene_object)      ; x is resting on y
  (in ?x - scene_object ?y - scene_object)      ; x is inside y
  (part ?x - scene_object ?y - scene_object)    ; x is an articulated part / sub-region of y

  (carry ?r - robot ?x - scene_object)          ; robot is holding x
  (free ?r - robot)                             ; robot gripper is empty

  (container ?x - scene_object)                 ; x can contain other objects
  (support ?x - scene_object)                   ; x can support an object placed on top

  (openable ?x - scene_object)                  ; x can transition between open and closed
  (open ?x - scene_object)
  (closed ?x - scene_object)

  (switchable ?x - scene_object)                ; x can be turned on / off (lamp, led)
  (turned_on ?x - scene_object)
  (turned_off ?x - scene_object)

  (slidable ?x - scene_object)                  ; x can be slid left / right (slider door)
  (at_left ?x - scene_object)                   ; left / right are with respect to the robot
  (at_right ?x - scene_object)

  (pushable ?x - scene_object)                  ; x can be moved laterally while not grasped
  (rotatable ?x - scene_object)                 ; x can be turned while grasped
 )

 (:action pickup_from ; Pick up movable object x from support/container z.
  :parameters (?x - movable ?r - robot ?z - scene_object)
  :precondition
  (and
   (free ?r)
   (or  ; The object must currently rest on something or be accessible inside something.
    (on ?x ?z)
    (and
     (in ?x ?z)
     (or
        (not (openable ?z))
      (open ?z)
     )
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
   (support ?z)
   (not (carry ?r ?z))
   (not (= ?x ?z))
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
   (container ?z)
   (open ?z)
   (not (carry ?r ?z))
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
  :parameters (?x - scene_object ?r - robot)
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
  :parameters (?x - scene_object ?r - robot)
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

 (:action move_slider_left ; Slide slidable object x to the left to expose the compartment.
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
   ; Sliding the door to either side is treated as making the compartment accessible.
   (open ?x)
   (not (closed ?x))
  )
 )

 (:action move_slider_right ; Slide slidable object x to the right to expose the compartment.
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
   ; Same abstraction as move_slider_left: a shifted slider counts as open.
   (open ?x)
   (not (closed ?x))
  )
 )

 (:action push_left ; Push movable object x leftward along support z without grasping it.
  :parameters (?x - movable ?r - robot ?z - scene_object)
  :precondition
  (and
   (free ?r)
   (pushable ?x)
   (on ?x ?z)
   (not (at_left ?x))
  )
  :effect
  (and
   (at_left ?x)
   (not (at_right ?x))
  )
 )

 (:action push_right ; Push movable object x rightward along support z without grasping it.
  :parameters (?x - movable ?r - robot ?z - scene_object)
  :precondition
  (and
   (free ?r)
   (pushable ?x)
   (on ?x ?z)
   (not (at_right ?x))
  )
  :effect
  (and
   (at_right ?x)
   (not (at_left ?x))
  )
 )

 (:action push_into ; Push movable object x from support src into open container dst without grasping it.
  :parameters (?x - movable ?r - robot ?src - scene_object ?dst - scene_object)
  :precondition
  (and
   (free ?r)
   (pushable ?x)
   (on ?x ?src)
   (container ?dst)
   (open ?dst)
   (not (= ?src ?dst))
  )
  :effect
  (and
   (not (on ?x ?src))
   (in ?x ?dst)
  )
 )

 (:action turn_left ; Turn held rotatable object x to the left without modeling orientation state.
  :parameters (?x - movable ?r - robot)
  :precondition
  (and
   (carry ?r ?x)
   (rotatable ?x)
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
   (rotatable ?x)
  )
  :effect
  (and
   ; This is intentionally a no-op on object state beyond preserving that the robot still holds x.
   (carry ?r ?x)
   (not (free ?r))
  )
 )
)
