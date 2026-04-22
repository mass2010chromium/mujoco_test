(define (problem calvin_default)
 (:domain tabletop)
 (:objects
  ; Default present objects, in all scenes.
  arm - robot
  table - immovable
  drawer - immovable
  led_light - background    ; colored green, rectangular in shape
  bulb_light - background   ; colored yellow, round in shape
  slider_shelf - immovable  ; slider shelf in the back

  block_1 - movable
  block_2 - movable
 )
 (:init
  ; Object variables
  (on block_1 table)
  (on block_2 table)

  ; Table state variables
  (turned_on bulb_light)
  (turned_off led_light)
  (closed drawer)

  ; These are always true.
  (openable drawer)
  (no_surface drawer)
  (open slider_shelf)
  (no_surface slider_shelf)
  (free arm)
 )
 (:goal (in block_1 slider_shelf))
)
