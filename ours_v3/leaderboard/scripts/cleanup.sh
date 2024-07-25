#!/bin/bash
MODEL="ours_v3"
cd /workspace/log/TCP/chkpt/${MODEL}
CHKPT_PATTERN=*
rm -r ${CHKPT_PATTERN}