# Project wrapper for Dear ImGui v1.92.8.
add_library(imgui STATIC
  imconfig.h
  imgui.cpp
  imgui_demo.cpp
  imgui_draw.cpp
  imgui.h
  imgui_internal.h
  imgui_tables.cpp
  imgui_widgets.cpp
  imstb_rectpack.h
  imstb_textedit.h
  imstb_truetype.h
  backends/imgui_impl_glfw.cpp
  backends/imgui_impl_glfw.h
  backends/imgui_impl_opengl3.cpp
  backends/imgui_impl_opengl3.h
)

target_include_directories(imgui
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends"
)

target_link_libraries(imgui
  PRIVATE
    glfw
    glad
    OpenGL::GL
)
