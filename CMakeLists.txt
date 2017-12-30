cmake_minimum_required(VERSION 3.0)
project(ResNet)

# Find dlib
find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# Find the Qt library
find_package(Qt5Core REQUIRED)

# source files
file(
  GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx)
file(GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(ResNet ${SRC} ${HDR})
target_link_libraries(ResNet dlib::dlib ${ITK_LIBRARIES} ${OpenCV_LIBS} Qt5::Core)