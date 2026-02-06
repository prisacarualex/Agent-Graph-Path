Ensure the following are installed on your system (Linux or WSL):

Docker (for running the simulation environment)

Python 3 (for the agent planning logic)

SBCL (Steel Bank Common Lisp, for running the ACT-R agents locally)


Project Structure Setup
Before running the system, ensure your file structure is organized as follows. You need to copy the necessary Lisp and Python files into the correct locations.
Geomates Folder: Ensure act-r-experiment.lisp and model-dummy.lisp are inside the geomates directory.
ACT-R Folder: Ensure the actr7.x folder is in the root directory (one level up from geomates).
Planner: Place planner.py in the root directory, alongside the actr7.x and geomates folders.
Expected Layout:
Plaintext
/project-root
├── planner.py
├── to_lisp            # (Named pipe or file created during execution)
├── to_python          # (Named pipe or file created during execution)
├── /actr7.x
│   └── load-act-r.lisp
└── /geomates
    ├── geomates.lisp
    ├── act-r-experiment.lisp
    ├── model-dummy.lisp
    └── viewer.html
🚀 Installation & Setup
1. Clean Up Docker
Stop and remove any existing Docker containers to ensure a fresh start.
Bash
docker stop $(docker ps -q) && docker rm $(docker ps -aq)

3. Start the Geomates Environment
Run the Geomates server container. This will listen for the GUI on port 8000 and agent connections on port 45678.

Bash
cd ./geomates
sudo docker run -p 8000:8000 -p 45678:45678 geomates:latest sbcl --script geomates.lisp; read
Note: You should see the following output indicating the server is ready:

Plaintext
Listening for GUI on port 8000.
Waiting for agents to connect on port 45678.
3. Set Up Python Environment
Open a new terminal window. Create and activate a Python virtual environment to manage dependencies.

Bash
# Navigate to project root
cd /path/to/project-root

# Create env (if not already created)
python3 -m venv env

# Activate env
source env/bin/activate

# Install any required packages (optional)
# pip install -r requirements.txt
🎮 Running the Agents
You need to run two separate agents in two separate terminals. Make sure you are in the project root folder and your Python environment is activated in both.

Agent 1
Open Terminal 2 and run:

Bash
(cat to_python | python3 planner.py > to_lisp) & sbcl \
  --load "actr7.x/load-act-r.lisp" \
  --load "geomates/act-r-experiment.lisp" \
  --eval "(load-act-r-model \"geomates/model-dummy.lisp\")" \
  --eval "(geomates-experiment)" \
  < to_lisp > to_python
Agent 2
Open Terminal 3 and run:

Bash
(cat to_python | python3 planner.py > to_lisp) & sbcl \
  --load "actr7.x/load-act-r.lisp" \
  --load "geomates/act-r-experiment.lisp" \
  --eval "(load-act-r-model \"geomates/model-dummy.lisp\")" \
  --eval "(geomates-experiment)" \
  < to_lisp > to_python
Verification
If successful, the terminal running the Docker container (Terminal 1) should output:

Plaintext
Agent 1 connected.
Agent 2 connected.
📺 Visualization
To watch the agents interact with the environment:

Open your web browser.

Navigate to the geomates/viewer.html file
