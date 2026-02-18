(define (domain libero_tabletop)
 (:requirements :equality :typing :negative-preconditions)
 (:types
  scene_object
  openable - scene_object
  graspable - scene_object

  robot
 )

 (:predicates 
  (carry ?r - robot ?x - scene_object)      ; Robot is carrying object
  (on ?x - scene_object ?y - scene_object)  ; x is on y
  (free ?r - robot)                         ; robot has hands free.
  (open ?x - openable)                      ; container is open
  (closed ?x - openable)                    ; container is closed
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
  :parameters (?x - openable ?r - robot)
  :precondition
  (and
   (free ?r)
   (open ?x)
  )
  :effect
  (and
   (not (open ?x))
   (closed ?x)
  )
 )

 (:action close ; Close object x using robot gripper. Gripper must be free
  :parameters (?x - openable ?r - robot)
  :precondition
  (and
   (free ?r)
   (closed ?x)
  )
  :effect
  (and
   (not (closed ?x))
   (open ?x)
  )
 )
)
