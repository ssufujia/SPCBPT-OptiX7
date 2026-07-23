find_package(OpenGL REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
# Match the known-good Debug layout: shared glfw3.dll next to the exe.
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
add_subdirectory(GLFW)
set_property(TARGET glfw PROPERTY C_STANDARD 99)
set_target_properties(glfw PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
)

add_library(glad SHARED
  glad/include/KHR/khrplatform.h
  glad/include/glad/gl.h
  glad/src/gl.c
)
target_compile_definitions(glad
  PRIVATE GLAD_API_CALL_EXPORT_BUILD
  PUBLIC GLAD_API_CALL_EXPORT
)
target_include_directories(glad PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/glad/include")
target_link_libraries(glad PUBLIC OpenGL::GL)

add_subdirectory(imgui)

set_property(TARGET glad PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
set_property(TARGET glfw PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
set_property(TARGET imgui PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
