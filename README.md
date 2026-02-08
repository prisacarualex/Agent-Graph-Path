# Geomates Multi-Agent Simulation

**Cross-Platform Setup Guide (Windows & macOS)**

---

## Prerequisites

Ensure the following are installed on your system:

- **Docker** (for running the simulation environment)
- **Python 3** (for the agent planning logic)
- **SBCL** (Steel Bank Common Lisp, for running the ACT-R agents locally)

### Platform-Specific Notes

**Windows:**
- Use WSL (Windows Subsystem for Linux) for best compatibility
- Docker Desktop for Windows required

**macOS:**
- Docker Desktop for Mac required
- SBCL can be installed via Homebrew: `brew install sbcl`

---

## Project Structure Setup

Before running the system, ensure your file structure is organized as follows:

```
/project-root
├── planner.py
├── to_lisp          # (Named pipe or file created during execution)
├── to_python        # (Named pipe or file created during execution)
├── /actr7.x
│   └── load-act-r.lisp
└── /geomates
    ├── geomates.lisp
    ├── act-r-experiment.lisp
    ├── model-dummy.lisp
    └── viewer.html
```

Copy the necessary Lisp and Python files into the correct locations:
- **Geomates Folder:** Ensure `act-r-experiment.lisp` and `model-dummy.lisp` are inside the `geomates` directory
- **ACT-R Folder:** Ensure the `actr7.x` folder is in the root directory
- **Planner:** Place `planner.py` in the root directory

---

## 🚀 Installation & Setup

### 1. Clean Up Docker

Stop and remove any existing Docker containers to ensure a fresh start.

**Linux/macOS/WSL:**
```bash
docker stop $(docker ps -q) && docker rm $(docker ps -aq)
```

**Windows (PowerShell):**
```powershell
docker ps -q | ForEach-Object { docker stop $_ }
docker ps -aq | ForEach-Object { docker rm $_ }
```

---

### 2. Start the Geomates Environment

Run the Geomates server container. This will listen for the GUI on port 8000 and agent connections on port 45678.

**Linux/WSL:**
```bash
cd ./geomates
sudo docker run -p 8000:8000 -p 45678:45678 geomates:latest sbcl --script geomates.lisp; read
```

**macOS:**
```bash
cd ./geomates
docker run -p 8000:8000 -p 45678:45678 geomates:latest sbcl --script geomates.lisp
```

**Windows (PowerShell/WSL):**
```bash
cd ./geomates
docker run -p 8000:8000 -p 45678:45678 geomates:latest sbcl --script geomates.lisp
```

**Expected Output:**
```
Listening for GUI on port 8000.
Waiting for agents to connect on port 45678.
```

---

### 3. Set Up Python Environment

Open a new terminal window and set up your Python environment.

**Linux/macOS/WSL:**
```bash
# Navigate to project root
cd /path/to/project-root

# Create virtual environment (if not already created)
python3 -m venv env

# Activate environment
source env/bin/activate

# Install any required packages (optional)
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
# Navigate to project root
cd C:\path\to\project-root

# Create virtual environment
python -m venv env

# Activate environment
env\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
# Navigate to project root
cd C:\path\to\project-root

# Create virtual environment
python -m venv env

# Activate environment
.\env\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

---

## 🎮 Running the Agents

You need to run two separate agents in two separate terminals. Make sure you are in the project root folder and your Python environment is activated in both.

### Agent 1

**Linux/WSL (Terminal 2):**
```bash
(cat to_python | python3 planner.py > to_lisp) & sbcl --load "actr7.x/load-act-r.lisp" --load "geomates/act-r-experiment.lisp" --eval "(load-act-r-model \"geomates/model-dummy.lisp\")" --eval "(geomates-experiment)" < to_lisp > to_python
```

**macOS (Terminal 2):**

First, create named pipes if they don't exist:
```bash
mkfifo to_python to_lisp
```

Then run the agent:
```bash
(cat to_python | python3 planner.py > to_lisp) & sbcl --load "actr7.x/load-act-r.lisp" --load "geomates/act-r-experiment.lisp" --eval "(load-act-r-model \"geomates/model-dummy.lisp\")" --eval "(geomates-experiment)" < to_lisp > to_python
```

**Windows (WSL Terminal 2):**
```bash
(cat to_python | python planner.py > to_lisp) & sbcl --load "actr7.x/load-act-r.lisp" --load "geomates/act-r-experiment.lisp" --eval "(load-act-r-model \"geomates/model-dummy.lisp\")" --eval "(geomates-experiment)" < to_lisp > to_python
```

---

### Agent 2

**Linux/WSL (Terminal 3):**
```bash
(cat to_python | python3 planner.py > to_lisp) & sbcl --load "actr7.x/load-act-r.lisp" --load "geomates/act-r-experiment.lisp" --eval "(load-act-r-model \"geomates/model-dummy.lisp\")" --eval "(geomates-experiment)" < to_lisp > to_python
```

**macOS (Terminal 3):**

Create separate named pipes for agent 2:
```bash
mkfifo to_python2 to_lisp2
```

Run agent 2:
```bash
(cat to_python2 | python3 planner.py > to_lisp2) & sbcl --load "actr7.x/load-act-r.lisp" --load "geomates/act-r-experiment.lisp" --eval "(load-act-r-model \"geomates/model-dummy.lisp\")" --eval "(geomates-experiment)" < to_lisp2 > to_python2
```

**Windows (WSL Terminal 3):**
```bash
(cat to_python | python planner.py > to_lisp) & sbcl --load "actr7.x/load-act-r.lisp" --load "geomates/act-r-experiment.lisp" --eval "(load-act-r-model \"geomates/model-dummy.lisp\")" --eval "(geomates-experiment)" < to_lisp > to_python
```

---

### Verification

If successful, the terminal running the Docker container (Terminal 1) should output:
```
Agent 1 connected.
Agent 2 connected.
```

---

## 📺 Visualization

To watch the agents interact with the environment:

1. Open your web browser
2. Navigate to the `viewer.html` file:
   - **macOS:** `file:///Users/yourusername/project-root/geomates/viewer.html`
   - **Windows:** `file:///C:/path/to/project-root/geomates/viewer.html`
   - **Linux:** `file:///home/yourusername/project-root/geomates/viewer.html`

**Alternative:** Serve via HTTP:
```bash
# In the geomates directory
python3 -m http.server 8080
```
Then open: `http://localhost:8080/viewer.html`

---

## Troubleshooting

### Common Issues

**Named Pipes Not Found (macOS/Linux):**
- Ensure you've created the named pipes with `mkfifo to_python to_lisp`
- Check permissions: `ls -l to_python to_lisp`

**Docker Permission Denied (Linux):**
- Add your user to the docker group: `sudo usermod -aG docker $USER`
- Log out and back in for changes to take effect

**Port Already in Use:**
- Check if another process is using ports 8000 or 45678
- Kill the process or use different ports in the docker run command

**Python Command Not Found (Windows):**
- Try `python` instead of `python3`
- Ensure Python is added to your PATH

---

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
