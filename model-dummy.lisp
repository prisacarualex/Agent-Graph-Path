;;; ================================================================
;;; PART 1: PYTHON BRIDGE (STRATEGIC UPDATE)
;;; ================================================================

(defvar *geomates-platforms* nil)
(defvar *disc-coords* '(0 0)) 
(defvar *rect-coords* '(0 0))
;; NEW: Store a list of all visible diamonds
(defvar *visible-diamonds* nil) 
(defvar *my-assigned-role* nil)

;; 1. Coordinate Accessors
(defun get-x (role)
  (if (or (equal role "disc") (equal role 'disc)) (first *disc-coords*) (first *rect-coords*)))

(defun get-y (role)
  (if (or (equal role "disc") (equal role 'disc)) (second *disc-coords*) (second *rect-coords*)))

;; 2. Communication Helper
(defun send-to-python (method params wait-for-reply)
  (let ((*print-pretty* nil)) 
    (format *standard-output* "JSON:{\"method\":\"~a\", \"params\":~a, \"id\":1}~%" 
            method 
            (json:encode-json-to-string params)))
  (finish-output *standard-output*)
  
  (if wait-for-reply
      (let ((response (read-line *standard-input* nil "s")))
        (string-trim '(#\Space #\Tab #\Newline #\Return) response))
      "ok"))

;; 3. Bridge Function (UPDATED)
;; Now sends the FULL LIST of diamonds to Python
(defun ask-python-for-move (role)
  (let ((r-str (if (stringp role) role (string-downcase (symbol-name role)))))
    (send-to-python "get-move" 
                    (list r-str 
                          (get-x r-str) (get-y r-str) 
                          *visible-diamonds*) ;; <--- Sending List!
                    t)))

(defun sync-visicon-to-python ()
  (send-to-python "init-map" (list *geomates-platforms*) nil))

(defun check-assigned-role ()
  (when *my-assigned-role*
    (intern (string-upcase *my-assigned-role*))))

;;; ================================================================
;;; PART 2: THE ACT-R MODEL (HYBRID)
;;; ================================================================

(clear-all)

(define-model geomates-hybrid-agent
    (sgp :v nil :esc t :needs-mouse nil :trace-detail high)

    (chunk-type (polygon-feature (:include visual-location)) value height width rotation color)
    (chunk-type (oval-feature (:include visual-location)) value radius color)
    (chunk-type task-goal state role)

    (add-dm (orange) (identifying) (moving) (disc) (rect) (start-goal isa task-goal state identifying))
    (goal-focus start-goal)

    ;; --- IDENTITY ---
    (p wait-for-role
       =goal> state identifying
       ==>
       !bind! =role (check-assigned-role)
       !eval! (when =role (format *error-output* "~%[LISP] I am the ~a!~%" =role))
       !eval! (unless =role (read-and-update-from-server)) 
       =goal> state identifying)

    (p found-role-disc
       =goal> state identifying
       !eval! (equal (check-assigned-role) 'DISC)
       ==>
       =goal> state checking-disc)

    (p found-role-rect
       =goal> state identifying
       !eval! (equal (check-assigned-role) 'RECT)
       ==>
       =goal> state checking-rect)

    ;; --- CONFIRMATION ---
    (p confirm-disc
       =goal> state checking-disc
       ?visual-location> state free
       ==>
       +visual-location> isa oval-feature value "disc"
       =goal> state finalizing-disc)

    (p finalize-disc
       =goal> state finalizing-disc
       =visual-location> isa oval-feature
       ==>
       =goal> state moving role disc
       !eval! (sync-visicon-to-python))

    (p confirm-rect
       =goal> state checking-rect
       ?visual-location> state free
       ==>
       +visual-location> isa polygon-feature value "rect"
       =goal> state finalizing-rect)

    (p finalize-rect
       =goal> state finalizing-rect
       =visual-location> isa polygon-feature
       ==>
       =goal> state moving role rect
       !eval! (sync-visicon-to-python))

    ;; --- MOVEMENT ---
    (p disc-move
       =goal> state moving role disc
       ?manual> state free
       ==>
       !bind! =key (ask-python-for-move "disc")
       +manual> isa press-key key =key)

    (p rect-move
       =goal> state moving role rect
       ?manual> state free
       ==>
       !bind! =key (ask-python-for-move "rect")
       +manual> isa press-key key =key)
)