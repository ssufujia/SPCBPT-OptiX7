#include <renderer/core/launch_params.h>

#include <iostream>

int main()
{
    std::cout << "DeviceLaunchParams sizeof=" << sizeof( DeviceLaunchParams )
              << " alignof=" << alignof( DeviceLaunchParams )
              << "; LightTraceParams sizeof=" << sizeof( LightTraceParams )
              << " alignof=" << alignof( LightTraceParams )
              << "; PreTraceParams sizeof=" << sizeof( PreTraceParams )
              << " alignof=" << alignof( PreTraceParams ) << '\n';
    return 0;
}
