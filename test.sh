#!/bin/bash
source ../../setup-env.sh
source ../setup-run-params.sh

unset SCOREP_LIBWRAP_ENABLE

export SCOREP_LIBWRAP_ENABLE=/lustre/orion/csc688/world-shared/scorep-amd/install/lib/libscorep_libwrap_rocblas.so

export LD_PRELOAD=/lib64/libc.so.6:$SCOREP_LIBWRAP_ENABLE

./main_wrapped