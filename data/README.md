# data/

Datasets go here (they are not in the repository: about 0.7 GB per 1 km drive).
Render them with the simulator:

    cd sim
    matlab -batch "generate_dataset"          % the hall only, about a minute
    matlab -batch "make_paper_datasets"       % all datasets of the paper

To keep the datasets elsewhere, set the environment variable `BATSLAM_DATA` to that folder.
The scripts cache the front-end output next to each dataset (`*.desc_*.npz`).
