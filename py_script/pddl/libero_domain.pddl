(define (domain tabletop)
 (:requirements :equality :typing :negative-preconditions :disjunctive-preconditions)
 (:types
  scene_object
  immovable - scene_object
  movable - scene_object

  robot
 )

 (:predicates 
  (on ?x - scene_object ?y - scene_object)      ; x is on y
  (in ?x - scene_object ?y - scene_object)      ; x is in y (y is a container)
  (part ?x - scene_object ?y - scene_object)    ; x is a component of y (articulated objects)
  (carry ?r - robot ?x - scene_object)          ; Robot is carrying movable object
  (free ?r - robot)                             ; Robot has hands free.
                                                ;   If `free` is set, the robot is not `carry`ing an object.
  (openable ?x - scene_object)                  ; Affordance of being a container that can be open or closed.
                                                ;   Represented as an attribute instead of a class, since either
                                                ;   movable or immovable objects can be openable...
                                                ;   For example, a drawer or refridgerator.
                                                ;   Openable objects must have exactly one of `open` or `closed` set.
  (open ?x - scene_object)                      ; Container is open.
  (closed ?x - scene_object)                    ; Container is closed. Cannot have objects placed in it.
  (switchable ?x - scene_object)                ; Affordance of being a switchable object that can be turned on or off.
                                                ;   Switchable objects must have exactly one of `turned_on` or `turned_off` set.
  (turned_on ?x - scene_object)                 ; Object is on.
  (turned_off ?x - scene_object)                ; Object is off.
 )

 (:action pickup_from ; Pick up object x from on object z. To pick up a stack of objects, pick up the bottom object
  :parameters (?x - movable ?r - robot ?z - scene_object)
  :precondition 
  (and 
   (free ?r)
   (or  ; The object we pick up should be on or in something (not held in hand)
     (on ?x ?z)
     (and (in ?x ?z) (not (closed ?z)))
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

 (:action place_on ; Place object x onto object z. (Stacks them)
  :parameters (?x - movable ?r - robot ?z - scene_object)
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
  :parameters (?x - movable ?r - robot ?z - scene_object)
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

 (:action turn_on ; Turn on a switchable object x using robot gripper. Gripper must be free.
  :parameters (?x - scene_object ?r - robot)
  :precondition
  (and
   (free ?r)
   (switchable ?x)
  )
  :effect
  (and
   (turned_on ?x)
   (not (turned_off ?x))
  )
 )

 (:action turn_off ; Turn off a switchable object x using robot gripper. Gripper must be free.
  :parameters (?x - scene_object ?r - robot)
  :precondition
  (and
   (free ?r)
   (switchable ?x)
  )
  :effect
  (and
   (turned_off ?x)
   (not (turned_on ?x))
  )
 )
)
