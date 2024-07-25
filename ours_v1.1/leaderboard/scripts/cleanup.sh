#!/bin/bash
MODEL="ours_v1/ours_v1.1"
cd /workspace/log/TCP/chkpt/${MODEL}
CHKPT_PATTERN=*
rm -r ${CHKPT_PATTERN}