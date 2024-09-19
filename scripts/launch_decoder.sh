#!/bin/bash
#
#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH -n 1   # number of cores
#SBATCH --mem 32GB # memory pool for all cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#
echo "Launching decoder script."
cd /nfs/gatsbystor/nicholasg/striatal_replay
source .venv/bin/activate
python scripts/decoder.py
echo "Decoder script finished."