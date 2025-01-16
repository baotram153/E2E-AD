#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export CARLA_ROOT=/workspace/simulation
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000            #
export TM_PORT=8000         #
export DEBUG_CHALLENGE=0
export REPETITIONS=1        # multiple evaluation runs
export RESUME=True
export DATA_COLLECTION=True


# Roach data collection
export ROUTES=leaderboard/data/TCP_training_routes/routes_town01.xml     #
export TEAM_AGENT=team_code/tcp_roach_agent.py
export TEAM_CONFIG="/workspace/log/TCP/chkpt/ours_v7/debug/best_epoch=29-val_loss=0.846.ckpt"
export CHECKPOINT_ENDPOINT=3090/data_collect_town01_results.json         #
export SCENARIOS=leaderboard/data/scenarios/town01_all_scenarios.json             #
export SAVE_PATH=/workspace/datasets/CARLA-bev-seg-data/data_collect_town01/     #                          



python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}


