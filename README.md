## Adaptive Synchronization with Hebbian Learning and Elasticity

### Setup python enviroment

(1) Create an ASHLE python virtual enviroment

```
python -m venv ashlenv
```

(2) Start the ASHLE virtual enviroment

```
source /path/to/ahslenv/bin/activate
```

(3) Setup the ASHLE virtual enviroment

```
python3 setup.py install
```

### Execute experiments from the paper:
```
python1 exp<experiment_number>.py
```
where `<experiment_number> = [1, 2, 3]`

### Experiment 1 (exp1.py)

Simulation of the MA between a musician’s beat and a metronome beat with a period shorter or longer than the musician’s SMP during solo musical performance.

Note: generates Figure 2 A,B,C from the article. 

### Experiment 2 (exp2.py)

Simulation of the slope between consecutive IBIs when an unpaced musician performs a melody starting at a tempo that is different than the SMT.

Note: generates Figure 3 A,B,C from the article. 

### Experiment 3 (exp3.py)

Simulation of the mean absolute asynchrony between two musicians with matching or mismatching SMTs during duet musical performance.

Note: generates Figure 4 A,B,C from the article. 

### Table to generate all figures from the article

| Figure No.  | Script to execute | File name |
| ------------- | ------------- | ------------- |
| Figure 1 | `N/A` | `/figures_raw/fig1(.eps/.png)` |
| Figure 2  | `/exp1/exp1.py`  | `/figures_raw/fig2(.eps/.png)` |
| Figure 3 | `/exp2/exp2.py`  | `/figures_raw/fig3(.eps/.png)` |
| Figure 4 | `/exp3/exp3.py`  | `/figures_raw/fig4(.eps/.png)` |
| Figure 5 | `/exp3/exp3.py` | `/figures_raw/fig5(.eps/.png)` |
| Figure 6 | `/param_analysis/asynch_lims.py`  | `/figures_raw/fig6(.eps/.png)` |
| Figure 7 | `/param_analysis/asynch_detail.py`  | `/figures_raw/fig7(.eps/.png)` |
| Figure 8 | `/param_analysis/slope.py`  | `/figures_raw/fig8(.eps/.png)` |





