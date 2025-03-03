```
project-root/                       # Root directory of the project
├── unmodified_from_NAT_net/        # Config files that required no modification
│   ├── DataDescriptions.py         # 
│   └── NatNetClient.py             # 
├── modified_from_NAT_net/          # Config files that required modification
│   ├── MoCapData.py                # modified data sent per frame
│   └── PythonSample.py             # hooked into our frame handler, added new commands
├── original_source/                # Files created specifically for our solution
│   ├── FeatureExtractor.py         # Main class file, contains abstractions for Trials, Players, Machines and the Game overall
│   ├── MocapVisualizer.py          # Class to render the arena and machines inside utilizing tkinter and matplot - deprecating, want to write a better GUI thats not so heavy
│   ├── tests.py                    # Unit tests for FeatureExtractor. Not fully implemented
│   └── process_recorded_data.py    # Legacy code to read in MOTIVE exported csv and pass into MocapVisualizer
├── old_saved_trials/               # Legacy folder that contains saved csvs directly exported from MOTIVE
├── requirements.txt                # List of project dependencies
├── README.md                       # This document
└── .gitignore                      # ignore python caches
```