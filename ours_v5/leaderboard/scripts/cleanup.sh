#!/bin/bash
# MODEL="ours_v1"
# cd /workspace/log/TCP/chkpt/${MODEL}
# CHKPT_PATTERN=*
# rm -r ${CHKPT_PATTERN}

MODEL="ours_v1_retrained"
cd /workspace/log/TCP/qualitative_res/result_${MODEL}
ROUTES_PATTERN=*
rm -r ${ROUTES_PATTERN}