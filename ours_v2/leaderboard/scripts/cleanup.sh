#!/bin/bash
MODEL="ours_v2"
cd /workspace/log/TCP/chkpt/${MODEL}
CHKPT_PATTERN=*
rm -r ${CHKPT_PATTERN}