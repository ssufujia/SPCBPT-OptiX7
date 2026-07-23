#pragma once

// Compile-time limits and code-generation switches shared by native CUDA and
// OptiX-IR. Scene selection, application policy, and runtime algorithm
// settings do not belong here.

#define SPCBPT_DEVICE_MAX_PATH_DEPTH 16
#define SPCBPT_DEVICE_PRETRACE_CONNECTION_CAPACITY 10

#define NUM_SUBSPACE 300
#define NUM_SUBSPACE_LIGHTSOURCE ( int( 0.2f * NUM_SUBSPACE ) )

#define RR_MIN_LIMIT
#define MIN_RR_RATE 0.3f
#define CONSERVATIVE_RATE 0.2f

#define LIMIT_PATH_TERMINATE true
#define DIR_JUDGE 0
#define FIX_ITERATION false

#define DOT_DEBUG_INFO_ENABLE false
#define DOT_MORE_PROXY_LIGHT_SUBPATH_NUM false
#define DOT_STOP_LEARNING_LATER false
#define DOT_LESS_MIS_WEIGHT false
#define SPCBPT_TERMINATE_EARLY false
#define DOT_BOUND_LIMIT_LESS false

static_assert( SPCBPT_DEVICE_MAX_PATH_DEPTH > 0, "Device path capacity must be positive" );
static_assert(
    SPCBPT_DEVICE_PRETRACE_CONNECTION_CAPACITY > 0,
    "Pretrace connection capacity must be positive"
);
