
;;; ================================================================
;;; PYTHON BRIDGE (STRATEGIC LAYER)
;;; ================================================================
(defvar *last-disc-pos* '(0 0))
(defvar *last-rect-pos* '(0 0))
(defvar *stuck-counter* 0)
(defvar *partner-message* nil)

;; Declaim the function so ask-python-for-move can see it even if defined later
(declaim (ftype (function (t) t) check-position-changed))

;;; ================================================================
;;; PYTHON BRIDGE (STRATEGIC LAYER)
;;; ================================================================

(defun get-x (role)
  (if (or (equal role "disc") (equal role 'disc))
      (first *disc-coords*)
      (first *rect-coords*)))

(defun get-y (role)
  (if (or (equal role "disc") (equal role 'disc))
      (second *disc-coords*)
      (second *rect-coords*)))

(defun send-to-python (method params wait-for-reply)
  "Send JSON-RPC message to Python strategic planner via stdout pipe."
  (let ((*print-pretty* nil))
    (format *standard-output* "JSON:{\"method\":\"~a\", \"params\":~a, \"id\":1}~%"
            method
            (json:encode-json-to-string params)))
  (finish-output *standard-output*)
  (if wait-for-reply
      (let ((response (read-line *standard-input* nil "s")))
        ;; NOTE: Only trim newlines/tabs, NOT spaces!
        ;; Space is a valid game key (rect hold position).
        (string-trim '(#\Tab #\Newline #\Return) response))
      "ok"))

(defun ask-python-for-move (role)
  "Query Python planner for next tactical action.
   Returns one of: a d w s (space) h
   Motor actions: a=left, d=right, w=jump/stretch, s=stop/shrink, space=hold
   Vocal action:  h=request help from partner (triggers speak production)"
  (let ((r-str (if (stringp role) role (string-downcase (symbol-name role)))))
    ;; Update cognitive stuck monitoring
    (check-position-changed r-str)
    (send-to-python "get-move"
                    (list r-str
                          (get-x r-str)
                          (get-y r-str)
                          *visible-diamonds*
                          *geomates-platforms*)
                    t)))

(defun sync-visicon-to-python ()
  (send-to-python "init-map" (list *geomates-platforms*) nil))

(defun check-assigned-role ()
  (when *my-assigned-role*
    (intern (string-upcase *my-assigned-role*))))


;;; ================================================================
;;; COGNITIVE MONITORING (WP6 — Failure Awareness)
;;; ================================================================

(defun check-position-changed (role)
  "Cognitive monitoring: has the agent moved since last check?
   Models the perceptual process of noticing lack of progress."
  (let* ((cx (get-x role))
         (cy (get-y role))
         (last-pos (if (or (equal role "disc") (equal role 'disc))
                       *last-disc-pos* *last-rect-pos*))
         (dx (abs (- cx (first last-pos))))
         (dy (abs (- cy (second last-pos))))
         (moved (> (+ dx dy) 0.15)))
    ;; Update position memory
    (if (or (equal role "disc") (equal role 'disc))
        (setf *last-disc-pos* (list cx cy))
        (setf *last-rect-pos* (list cx cy)))
    ;; Increment or reset stuck counter
    (if moved
        (setf *stuck-counter* 0)
        (incf *stuck-counter*))
    moved))

(defun is-agent-stuck ()
  "Returns T if agent hasn't moved for many cycles.
   Used by stuck-detection production to trigger cognitive awareness."
  (> *stuck-counter* 20))

(defun diamonds-visible-p ()
  "Returns T if diamonds are in the visual scene."
  (and *visible-diamonds* (> (length *visible-diamonds*) 0)))

(defun get-diamond-count ()
  (length *visible-diamonds*))


;;; ================================================================
;;; ACT-R MODEL DEFINITION
;;; ================================================================
;;;
;;; Production Rule Categories (WP3):
;;;   1. Role Identification   — Perceive assigned role from server
;;;   2. Visual Confirmation   — Attend to self in visual field
;;;   3. Strategic Delegation  — Request plan from Python planner
;;;   4. Tactical Execution    — Motor actions per plan
;;;      - move-left-production   (key "a")
;;;      - move-right-production  (key "d")
;;;      - jump-production        (key "w") — disc jump / rect stretch
;;;      - stop-production        (key "s") — disc stop / rect shrink
;;;      - hold-production        (key " ") — rect hold position
;;;   5. Stuck Monitoring      — Detect lack of progress (WP6)
;;;   6. Vocal Request         — Speak help request to partner (WP4)
;;;   7. Aural Processing      — Hear and process partner messages (WP4)
;;;
;;; State Machine:
;;;   identifying -> checking-{disc|rect} -> finalizing-{disc|rect}
;;;     -> planning -> executing -> planning -> ...   (main loop)
;;;     -> speaking -> planning                       (vocal out)
;;;     -> listening -> planning                      (aural in)
;;;
;;; ================================================================

(clear-all)

(define-model geomates-hybrid-agent

  ;; --- Model Parameters ---
  (sgp :v nil           ;; Verbose trace off (set t for debugging)
       :esc t           ;; Enable sub-symbolic computation
       :needs-mouse nil ;; No mouse needed
       :trace-detail high
       :er t            ;; Enable randomness
       :dat 0.01        ;; Fast default action time (responsive control)
       :visual-attention-latency 0.01  ;; Fast visual shifts
  )

  ;; --- Chunk Types ---
  ;; Visual features (extending ACT-R visual-location)
  (chunk-type (polygon-feature (:include visual-location))
    value height width rotation color)
  (chunk-type (oval-feature (:include visual-location))
    value radius color)

  ;; Goal buffer: tracks agent state through perception-action cycle
  (chunk-type task-goal
    state              ;; Current cognitive state (see state machine above)
    role               ;; disc or rect (set during identification)
    next-action        ;; Action key from Python planner
    stuck-count        ;; Cognitive awareness of lack of progress
    assist-active      ;; T when collaboration is in progress
    partner-message    ;; Last received vocal message from partner
  )

  ;; Aural processing types
  (chunk-type audio-event location type)
  (chunk-type sound event content)

  ;; --- Declarative Memory (initial knowledge) ---
  (add-dm
    (identifying) (checking-disc) (checking-rect)
    (finalizing-disc) (finalizing-rect)
    (planning) (executing) (speaking)
    (disc) (rect)
    (start-goal isa task-goal
      state identifying
      stuck-count 0
      assist-active nil
      partner-message nil))

  (goal-focus start-goal)


  ;; ============================================================
  ;; 1. ROLE IDENTIFICATION PRODUCTIONS (WP5)
  ;;    Agent perceives which role it was assigned by the server.
  ;;    Models the cognitive process of attending to an external
  ;;    assignment and updating self-knowledge.
  ;; ============================================================

  (p wait-for-role
     "Poll server until role assignment is perceived."
     =goal>
       isa task-goal
       state identifying
     ==>
     !bind! =role (check-assigned-role)
     !eval! (when =role
              (format *error-output* "~%[ACT-R] Perceived role: ~a~%" =role))
     !eval! (unless =role (read-and-update-from-server))
     =goal>
       state identifying)

  (p found-role-disc
     "Recognized: I am the DISC agent (round, can jump high)."
     =goal>
       isa task-goal
       state identifying
     !eval! (equal (check-assigned-role) 'DISC)
     ==>
     !eval! (format *error-output* "~%*** PRODUCTION: found-role-disc ***~%")
     =goal>
       state checking-disc
       role disc)

  (p found-role-rect
     "Recognized: I am the RECT agent (rectangular, can reshape)."
     =goal>
       isa task-goal
       state identifying
     !eval! (equal (check-assigned-role) 'RECT)
     ==>
     !eval! (format *error-output* "~%*** PRODUCTION: found-role-rect ***~%")
     =goal>
       state checking-rect
       role rect)


  ;; ============================================================
  ;; 2. VISUAL CONFIRMATION PRODUCTIONS
  ;;    Attend to own agent in the visual field to confirm identity.
  ;;    Uses ACT-R's visual-location buffer for feature search.
  ;; ============================================================

  (p confirm-disc-visual
     "Visual search: look for disc (oval) feature in scene."
     =goal>
       isa task-goal
       state checking-disc
     ?visual-location>
       state free
     ==>
     !eval! (format *error-output* "*** PRODUCTION: confirm-disc-visual ***~%")
     +visual-location>
       isa oval-feature
       value "disc"
     =goal>
       state finalizing-disc)

  (p finalize-disc-identity
     "Disc confirmed in visual field. Initialize Python bridge."
     =goal>
       isa task-goal
       state finalizing-disc
     =visual-location>
       isa oval-feature
     ==>
     !eval! (format *error-output* "*** PRODUCTION: finalize-disc-identity ***~%")
     !eval! (sync-visicon-to-python)
     =goal>
       state planning
       role disc)

  (p confirm-rect-visual
     "Visual search: look for rect (polygon) feature in scene."
     =goal>
       isa task-goal
       state checking-rect
     ?visual-location>
       state free
     ==>
     !eval! (format *error-output* "*** PRODUCTION: confirm-rect-visual ***~%")
     +visual-location>
       isa polygon-feature
       value "rect"
     =goal>
       state finalizing-rect)

  (p finalize-rect-identity
     "Rect confirmed in visual field. Initialize Python bridge."
     =goal>
       isa task-goal
       state finalizing-rect
     =visual-location>
       isa polygon-feature
     ==>
     !eval! (format *error-output* "*** PRODUCTION: finalize-rect-identity ***~%")
     !eval! (sync-visicon-to-python)
     =goal>
       state planning
       role rect)


  ;; ============================================================
  ;; 3. STRATEGIC DELEGATION PRODUCTIONS
  ;;    Request next action from Python's D* Lite planner.
  ;;    Python performs System-2 deliberation (pathfinding, diamond
  ;;    selection, failure analysis) and returns a primitive action.
  ;;
  ;;    Returns: "a" "d" "w" "s" " " (motor) or "h" (vocal)
  ;; ============================================================

  (p request-disc-plan
     "DISC: Delegate strategic decision to Python planner."
     =goal>
       isa task-goal
       state planning
       role disc
     ?manual>
       state free
     ==>
     !bind! =action (ask-python-for-move "disc")
     !eval! (format *error-output* "[ACT-R:DISC] Python says: ~a~%" =action)
     =goal>
       state executing
       next-action =action)

  (p request-rect-plan
     "RECT: Delegate strategic decision to Python planner."
     =goal>
       isa task-goal
       state planning
       role rect
     ?manual>
       state free
     ==>
     !bind! =action (ask-python-for-move "rect")
     !eval! (format *error-output* "[ACT-R:RECT] Python says: ~a~%" =action)
     =goal>
       state executing
       next-action =action)


  ;; ============================================================
  ;; 4. TACTICAL EXECUTION PRODUCTIONS (WP3)
  ;;    Each primitive action has its own production rule, modeling
  ;;    distinct cognitive-motor processes:
  ;;
  ;;    move-left:  Activate left-movement motor program
  ;;    move-right: Activate right-movement motor program
  ;;    jump:       Activate vertical motor program (disc=jump, rect=stretch)
  ;;    stop:       Inhibit movement / activate shrink (rect)
  ;;    hold:       Maintain current state (rect only)
  ;;
  ;;    Each production fires the ACT-R manual module to press
  ;;    the corresponding key, then returns to planning state.
  ;; ============================================================

  ;; --- P4a: Move Left ---
  (p move-left-production
     "Tactical: Execute leftward movement.
      Fires when Python planner returns 'a' (move left).
      Models activation of left-directed motor program."
     =goal>
       isa task-goal
       state executing
       next-action "a"
     ?manual>
       state free
     ==>
     !eval! (format *error-output* "  [MOTOR] move-left~%")
     +manual>
       cmd press-key
       key "a"
     =goal>
       state planning
       next-action nil)

  ;; --- P4b: Move Right ---
  (p move-right-production
     "Tactical: Execute rightward movement.
      Fires when Python planner returns 'd' (move right).
      Models activation of right-directed motor program."
     =goal>
       isa task-goal
       state executing
       next-action "d"
     ?manual>
       state free
     ==>
     !eval! (format *error-output* "  [MOTOR] move-right~%")
     +manual>
       cmd press-key
       key "d"
     =goal>
       state planning
       next-action nil)

  ;; --- P4c: Jump / Stretch Tall ---
  (p jump-or-stretch-production
     "Tactical: Execute vertical action.
      DISC: Jump upward (ballistic motor program).
      RECT: Stretch taller (reshape motor program).
      Fires when Python planner returns 'w'."
     =goal>
       isa task-goal
       state executing
       next-action "w"
     ?manual>
       state free
     ==>
     !eval! (format *error-output* "  [MOTOR] jump/stretch-up~%")
     +manual>
       cmd press-key
       key "w"
     =goal>
       state planning
       next-action nil)

  ;; --- P4d: Stop / Shrink Wide ---
  (p stop-or-shrink-production
     "Tactical: Execute braking/reshape action.
      DISC: Stop movement (motor inhibition).
      RECT: Shrink wider / squish (reshape motor program).
      Fires when Python planner returns 's'."
     =goal>
       isa task-goal
       state executing
       next-action "s"
     ?manual>
       state free
     ==>
     !eval! (format *error-output* "  [MOTOR] stop/shrink-wide~%")
     +manual>
       cmd press-key
       key "s"
     =goal>
       state planning
       next-action nil)

  ;; --- P4e: Hold Position ---
  (p hold-position-production
     "Tactical: Maintain current state (rect holds shape/position).
      Fires when Python planner returns space.
      Models deliberate inaction as a motor decision."
     =goal>
       isa task-goal
       state executing
       next-action " "
     ?manual>
       state free
     ==>
     !eval! (format *error-output* "  [MOTOR] hold-position~%")
     +manual>
       cmd press-key
       key " "
     =goal>
       state planning
       next-action nil)


  ;; ============================================================
  ;; 5. STUCK MONITORING PRODUCTION (WP6)
  ;;    ACT-R detects when the agent hasn't moved for many cycles.
  ;;    This models the cognitive process of noticing lack of
  ;;    progress — a meta-cognitive monitoring function.
  ;;    Python handles the actual replanning; ACT-R's role is
  ;;    perceptual awareness of the stuck state.
  ;; ============================================================

  (p detect-stuck-state
     "Meta-cognitive monitor: Notice agent hasn't moved.
      Fires when position hasn't changed for 20+ cycles.
      Logs awareness; Python planner handles escalation
      (retry different diamond -> request ASSIST -> blacklist)."
     =goal>
       isa task-goal
       state planning
     !eval! (is-agent-stuck)
     ==>
     !eval! (format *error-output*
              "~%[ACT-R:META] STUCK detected! (~a frames without movement)~%"
              *stuck-counter*)
     ;; Stay in planning — Python planner's stuck_limit will handle escalation
     ;; ACT-R's contribution: cognitive awareness of the failure state
     =goal>
       state planning)


  ;; ============================================================
  ;; 6. VOCAL COMMUNICATION PRODUCTIONS (WP4 — Sending)
  ;;    When Python signals ASSIST is needed (returns "h"),
  ;;    ACT-R uses the vocal module to speak to the partner.
  ;;    This implements inter-agent communication through the
  ;;    cognitive architecture's speech production system.
  ;; ============================================================

  ;; --- P6a: Speak ASSIST Request (Disc -> Rect) ---
  (p speak-help-request
     "Vocal: Disc requests assistance from Rect.
      Fires when Python returns 'h' (help signal).
      Uses ACT-R vocal module to produce speech act.
      Models the cognitive process of formulating and
      articulating a help request to a collaborative partner."
     =goal>
       isa task-goal
       state executing
       next-action "h"
     ?vocal>
       state free
     ==>
     !eval! (format *error-output*
              "~%*** PRODUCTION: speak-help-request ***~%")
     !eval! (format *error-output*
              "  [VOCAL] Speaking: NEED-ASSIST~%")
     +vocal>
       cmd speak
       string "NEED-ASSIST"
     =goal>
       state planning
       next-action nil
       assist-active t)

  ;; --- P6b: Acknowledge ASSIST (Rect -> Disc) ---
  (p speak-assist-acknowledge
     "Vocal: Rect acknowledges partner's help request.
      Fires when partner-message indicates help is needed.
      Models cooperative response — willingness to assist."
     =goal>
       isa task-goal
       state planning
       partner-message "NEED-ASSIST"
     ?vocal>
       state free
     ==>
     !eval! (format *error-output*
              "~%*** PRODUCTION: speak-assist-acknowledge ***~%")
     !eval! (format *error-output*
              "  [VOCAL] Speaking: ASSIST-READY~%")
     +vocal>
       cmd speak
       string "ASSIST-READY"
     =goal>
       partner-message nil
       assist-active t)

  ;; --- P6c: Report Collaboration Complete ---
  (p speak-assist-complete
     "Vocal: Report that collaboration task is finished.
      Fires when assist was active but mode returned to COLLECT."
     =goal>
       isa task-goal
       state planning
       assist-active t
     ?vocal>
       state free
     ==>
     !eval! (format *error-output*
              "~%*** PRODUCTION: speak-assist-complete ***~%")
     !eval! (format *error-output*
              "  [VOCAL] Speaking: ASSIST-DONE~%")
     +vocal>
       cmd speak
       string "ASSIST-DONE"
     =goal>
       assist-active nil)


  ;; ============================================================
  ;; 7. AURAL PROCESSING PRODUCTIONS (WP4 — Receiving)
  ;;    ACT-R's aural module detects speech from the partner agent.
  ;;    These productions model the cognitive process of:
  ;;      1. Detecting a sound event (attention capture)
  ;;      2. Attending to the sound (auditory processing)
  ;;      3. Comprehending the message (semantic processing)
  ;;      4. Updating goal state (action implications)
  ;; ============================================================

  ;; --- P7a: Detect Partner Sound ---
  (p hear-partner-sound
     "Aural attention: Detect external sound event.
      Models bottom-up attention capture by auditory stimulus."
     =goal>
       isa task-goal
     =aural-location>
       isa audio-event
       location external
     ?aural>
       state free
     ==>
     !eval! (format *error-output*
              "~%*** PRODUCTION: hear-partner-sound ***~%")
     +aural>
       isa sound
       event =aural-location)

  ;; --- P7b: Process Help Request ---
  (p process-help-request
     "Aural comprehension: Partner said NEED-ASSIST.
      Updates goal to reflect partner's need for collaboration.
      Models semantic processing of cooperative request."
     =goal>
       isa task-goal
     =aural>
       isa sound
       content "NEED-ASSIST"
     ==>
     !eval! (format *error-output*
              "*** PRODUCTION: process-help-request ***~%")
     !eval! (format *error-output*
              "  [AURAL] Partner needs help!~%")
     !eval! (setf *partner-message* "NEED-ASSIST")
     =goal>
       partner-message "NEED-ASSIST"
       assist-active t
     -aural>)

  ;; --- P7c: Process Acknowledgment ---
  (p process-help-acknowledged
     "Aural comprehension: Partner said ASSIST-READY.
      Models confirmation that collaborative partner is available."
     =goal>
       isa task-goal
     =aural>
       isa sound
       content "ASSIST-READY"
     ==>
     !eval! (format *error-output*
              "*** PRODUCTION: process-help-acknowledged ***~%")
     !eval! (format *error-output*
              "  [AURAL] Partner is ready to help!~%")
     =goal>
       partner-message "ASSIST-READY"
     -aural>)

  ;; --- P7d: Process Completion ---
  (p process-assist-finished
     "Aural comprehension: Partner said ASSIST-DONE.
      Models understanding that collaboration episode is over."
     =goal>
       isa task-goal
     =aural>
       isa sound
       content "ASSIST-DONE"
     ==>
     !eval! (format *error-output*
              "*** PRODUCTION: process-assist-finished ***~%")
     !eval! (format *error-output*
              "  [AURAL] Collaboration complete.~%")
     =goal>
       partner-message nil
       assist-active nil
     -aural>)

  ;; --- P7e: Process Other Messages ---
  (p process-other-message
     "Aural comprehension: Generic message from partner.
      Stores message content for potential future use."
     =goal>
       isa task-goal
     =aural>
       isa sound
       content =msg
     ;; Only fire if not one of the specific messages above
     !eval! (not (member =msg '("NEED-ASSIST" "ASSIST-READY" "ASSIST-DONE")
                  :test #'equal))
     ==>
     !eval! (format *error-output*
              "*** PRODUCTION: process-other-message: ~a ***~%" =msg)
     =goal>
       partner-message =msg
     -aural>)

) ;; end define-model
