(define (domain tabletop)
 (:requirements :equality :typing :negative-preconditions :disjunctive-preconditions)
 (:types
  scene_object
  immovable - scene_object
  graspable - scene_object

  robot
 )

 (:predicates 
  (on ?x - scene_object ?y - scene_object)  ; x is on y
  (in ?x - scene_object ?y - scene_object)  ; x is in y (y is a container)
  (carry ?r - robot ?x - scene_object)      ; Robot is carrying object
  (free ?r - robot)                         ; Robot has hands free.
                                            ;   Exactly one of `free` and `carry` must be set.
  (openable ?x - scene_object)              ; Affordance of being a container that can be open or closed.
                                            ;   Represented as an attribute instead of a class, since either
                                            ;   graspable or immovable objects can be openable...
                                            ;   For example, a drawer or refridgerator.
                                            ;   Openable objects must have exactly one of `open` or `closed` set.
  (open ?x - scene_object)                  ; Container is open.
  (closed ?x - scene_object)                ; Container is closed. Cannot have objects placed in it.
 )

 (:action pickup_from ; Pick up object x from on object z.
  :parameters (?x - graspable ?r - robot ?z - scene_object)
  :precondition 
  (and 
   (free ?r)
   (or  ; The object we pick up should be on or in something...
     (on ?x ?z)
     (and (in ?x ?z) (not (closed ?z)))
   )
   (forall (?o - scene_object)  ; and it should not have anything on it.
     (not (on ?o ?x))
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

 (:action place_on ; Place object x onto object z.
  :parameters (?x - graspable ?r - robot ?z - scene_object)
  :precondition
  (and
   (carry ?r ?x)
   (not (carry ?r ?z))
   (forall (?o - scene_object)  ; Disallow building stacks in containers.
     (not (in ?z ?o))
   )
  )
  :effect
  (and
   (not (carry ?r ?x))
   (free ?r)
   (on ?x ?z)
  )
 )

 (:action place_in ; Place object x into object z.
  :parameters (?x - graspable ?r - robot ?z - scene_object)
  :precondition
  (and
   (carry ?r ?x)
   (not (carry ?r ?z))
   (open ?z)
  )
  :effect
  (and
   (not (carry ?r ?x))
   (free ?r)
   (in ?x ?z)
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
