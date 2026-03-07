(define (problem libero_spatial_0)
 (:domain tabletop)
 (:objects
  arm - robot
  table - scene_object
  bowl1 - graspable
  bowl2 - graspable
  plate - scene_object
  ramekin - graspable
  cookies - graspable
  drawers - immovable
 )
 (:init
  (on bowl1 table)
  (on bowl2 table)
  (on plate table)
  (on ramekin table)
  (on cookies table)
  (on drawers table)
  (openable drawers)
  (closed drawers)
  (free arm)
 )
 (:goal (on bowl1 plate))
)
