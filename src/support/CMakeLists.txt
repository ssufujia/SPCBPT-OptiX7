find_package(OpenGL REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(GLFW)
set_property(TARGET glfw PROPERTY C_STANDARD 99)

add_library(glad SHARED
  KHR/khrplatform.h
  glad/glad.c
  glad/glad.h
)
target_compile_definitions(glad
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC GLAD_GLAPI_EXPORT
)
target_include_directories(glad PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(glad PUBLIC OpenGL::GL)

add_subdirectory(imgui)

set_property(TARGET glad PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
set_property(TARGET glfw PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
set_property(TARGET imgui PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
