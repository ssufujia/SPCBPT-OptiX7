#pragma once

// Small renderer contract shared by the host pipeline builder and device-facing
// launch code. Keep application/UI configuration out of this header.

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_LIGHTSUBPATH = 0,
    RAY_TYPE_EYESUBPATH = 0,
    RAY_TYPE_EYESUBPATH_SIMPLE = 2,
    RAY_TYPE_COUNT = 3
};

enum RayHitType
{
    RAYHIT_TYPE_LIGHTSOURCE = 0,
    RAYHIT_TYPE_NORMAL = 1,
    RAYHIT_TYPE_COUNT = 2
};

static_assert( RAY_TYPE_COUNT == 3, "SBT ray type ABI changed" );
static_assert( RAYHIT_TYPE_COUNT == 2, "SBT hit type ABI changed" );
