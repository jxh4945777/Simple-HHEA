#!/bin/sh
python -u get_neighbor_sim.py
python -u get_entity_embedding.py
python -u get_neighView_and_desView_interaction_feature.py
python -u interaction_model.py
