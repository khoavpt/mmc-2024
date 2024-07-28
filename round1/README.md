# Round 1
This project is part of Round 1 Math Modeling contest 2024. 



## Setup
To set up the project, follow these steps:

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script:
    ```bash
    python main.py
    ```

You can run the algorithm on different data with different numbers of faculties, days, and constraints by adding the data to the `data` folder and changing the configurations in `config.yaml`. 

Using Hydra, it's also possible to change configurations directly from the command line. For example:
```bash
python main.py dataset.num_days=30 genetic_algorithm.num_generations=200
```


## Folder Structure
```
project-root/
│
├── configs/
│   └── config.yaml          # Configuration file for the project
│
├── data/
│   ├── doctor_1.csv         # Data file for doctor 1
│   ├── doctor2.csv          # Data file for doctor 2
│   └── nurse.csv            # Data file for nurses
│
├── figures/                 # Folder to store figures and plots
│
├── outputs/                 # Folder containing Hydra logs
│
├── notebooks/
│   └── result_analysis.ipynb # Jupyter notebook for result analysis
│
├── src/
│   ├── algorithm/           # Folder containing algorithm implementations
│   ├── data/                # Folder for data processing scripts
│   ├── utils/               # Utility scripts
│   ├── main.py              # Main script to run the algorithm
│   └── finetune.py          # Script for fine-tuning with Optuna
│
├── .env                     # Environment file
│
├── report.pdf               # Final report
│
├── requirements.txt         # List of dependencies
│
└── results/                 # Folder containing final scheduling results for Môn Hòa hospital
```
The final scheduling results can be found in the `results` folder.