```
Simulation/                         # Files for the statistical simulation of the higher-order game models
├── Houses.py                       # Contains the functions for each model to be tested
├── Players.py                      # Contains the functions for each model to be tested against
└── simulation.py                   # Contains the actual simulation, including the game definition and
                                      constraints, a trial simulation and naive statistical report
unmodified_from_NAT_net/            # Config files that required no modification
    └── DataDescriptions.py         # 
modified_from_NAT_net/              # Config files that required modification
    ├── MoCapData.py                # modified data sent per frame and the test data contents
    ├── NatNetClient.py             # added a thread check function to allow connection retries
    └── PythonSample.py             # hooked into our frame handler, added new commands
original_source/                    # Files created specifically for our solution
    ├── FeatureExtractor.py         # Main class file, contains abstractions for Trials, Players, Machines and the Game overall
    ├── MocapVisualizer.py          # (OUTDATED) Class to render the arena and machines inside utilizing
    |                                 tkinter and matplot. (Functionality moved to FancyVisualizer.py instead)
    ├── tests.py                    # Unit tests for FeatureExtractor. Not fully implemented
    ├── lstm.py                     # A first pass attempt at an online lstm using the saved trials. Current trials were insufficient and results inconclusive.
    ├── FancyVisualizer.py          # Dearpygui application, handles arena rendering with bounding boxes
    │                                 displayed for foyer and machines. Has interface for automated connection
    │                                 and trial recording. Console to log collision events, connection errors
    |                                 and saving success.
    └── process_recorded_data.py    # Legacy code to read in MOTIVE exported csv and pass into MocapVisualizer

old_saved_trials/               # Legacy folder that contains saved csvs directly exported from MOTIVE
requirements.txt                # List of project dependencies 
                                  - install with: python -m pip install -r requirements.txt
README.md                       # This document
.gitignore                      # ignore python caches
```
