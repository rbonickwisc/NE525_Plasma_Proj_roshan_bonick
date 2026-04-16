# Tokamak DT Neutron Source Model

This repository contains a reusable Python package for building a tokamak D-T neutron source model based on magnetic surface geometry and prescribed plasma profiles

Current codebase includes:

    -Tokamak magnetic surface geometry
    -L-mode plasma profiles
    -General pedestal-mode plasma profiles(pedestal-mode is used for non-l-mode cases such as A-mode or H-mode)
    -Paper aligned A-mode profile case demo(uses pedestal-mode)
    -DT reactivity using Sadler-Van Belle formula
    -Local source density evaluation
    -Source normalization and total neutron rate estimation
    -Continuous in-cell source sampling
    -Convergence and comparison studies
    -Unit tests for main model components

========================================
########## Setup(do these in bash) ##########

1. git clone <https://github.com/rbonickwisc/NE525_Plasma_Proj_roshan_bonick.git>

2. cd NE525_Plasma_Proj_roshan_bonick

3. pip install -e .
4. pip install -r requirements.txt

========================================
######## How to run studies ########
From the project root run:


python studies/l_mode_demo_case.py
python studies/pedestal_mode_demo_case.py
python studies/a_mode_paper_demo_case.py
python studies/compare_l_mode_vs_pedestal.py
python studies/compare_l_mode_vs_a_mode_paper.py

######## How to run tests ########

pytest tests/