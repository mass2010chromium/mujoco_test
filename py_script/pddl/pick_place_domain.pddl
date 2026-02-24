(define (domain tabletop)
 (:requirements :equality :typing :negative-preconditions)
 (:types
  scene_object
  immovable - scene_object
  graspable - scene_object

  robot
 )

 (:predicates 
  (carry ?r - robot ?x - scene_object)      ; Robot is carrying object
  (on ?x - scene_object ?y - scene_object)  ; x is on y
  (free ?r - robot)                         ; Robot has hands free.
  (openable ?x - scene_object)              ; Affordance of being a container that can be open or closed.
                                            ;   Represented as an attribute instead of a class, since either
                                            ;   graspable or immovable objects can be openable...
                                            ;   For example, a drawer or refridgerator.
                                            ;   Openable objects must have exactly one of `open` or `closed` set.
  (open ?x - scene_object)                  ; Container is open.
  (closed ?x - scene_object)                ; Container is closed
 )

 (:action pickup_from ; Pick up object x from on object z.
  :parameters (?x - graspable ?r - robot ?z - scene_object)
  :precondition 
  (and 
   (free ?r)
   (on ?x ?z)
  )
  :effect 
  (and
   (carry ?r ?x)
   (not (free ?r))
   (not (on ?x ?z))
  )
 )

 (:action place ; Place object x onto object z.
  :parameters (?x - graspable ?r - robot ?z - scene_object)
  :precondition
  (and
   (carry ?r ?x)
   (not (carry ?r ?z))
  )
  :effect
  (and
   (not (carry ?r ?x))
   (free ?r)
   (on ?x ?z)
  )
 )

 (:action open  ; Open object x using robot gripper. Gripper must be free
  :parameters (?x - scene_object ?r - robot)
  :precondition
  (and
   (free ?r)
   (closed ?x)
   (openable ?x)
  )
  :effect
  (and
   (not (closed ?x))
   (open ?x)
  )
 )

 (:action close ; Close object x using robot gripper. Gripper must be free
  :parameters (?x - scene_object ?r - robot)
  :precondition
  (and
   (free ?r)
   (open ?x)
   (openable ?x)
  )
  :effect
  (and
   (not (open ?x))
   (closed ?x)
  )
 )
)
