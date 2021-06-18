cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(app)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(Torch REQUIRED)
set(OpenCV_DIR "/usr/local/opt/opencv@3/share/OpenCV")
find_package( OpenCV REQUIRED )
include_directories( 
    ${OpenCV_INCLUDE_DIRS}
    lib
)
add_executable(main main.cpp lib)
add_executable(test test.cpp)

target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(test ${TORCH_LIBRARIES} ${OpenCV_LIBS})


