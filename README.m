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

### Execute experiments from the paper by simply:
```
python1 exp<experiment_number>.py
```
where `<experiment_number> = [1, 2, 3]`

### Experiment 1 (exp1.py)

Solo music performance with a metronome tempo different than the SMT

### Experiment 2 (exp2.py)

Unpaced solo music performance with a starting tempo different than the SMT

### Experiment 3 (exp2.py)

Duet musical performance between musicians with matching or mismatching SMTs

