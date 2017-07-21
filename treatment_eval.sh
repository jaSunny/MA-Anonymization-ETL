#!/bin/bash

for i in {2..73}; do /usr/bin/python3 anonymizer.py -score yes -size yes --treatment yes --overwrite_treatment suppression --eval_treatment no -ncol $i; done

for i in {2..73}; do /usr/bin/python3 anonymizer.py -score yes -size yes --treatment yes --overwrite_treatment compartmentation --eval_treatment no -ncol $i; done

for i in {2..73}; do /usr/bin/python3 anonymizer.py -score yes -size yes --treatment yes --overwrite_treatment perturbation --eval_treatment no -ncol $i; done

for i in {2..73}; do /usr/bin/python3 anonymizer.py -score yes -size yes --treatment yes --overwrite_treatment generalization --eval_treatment no -ncol $i; done
