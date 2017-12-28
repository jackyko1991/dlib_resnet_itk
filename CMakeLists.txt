cmake_minimum_required(VERSION 3.0)
project(ResNet)

# Find dlib
find_package(dlib)
message(STATUS dlib::dlib)
include_directories(${dlib_INCLUDE_DIRS})

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS system filesystem thread REQUIRED)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

# source files
file(
  GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx)
file(GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(ResNet ${SRC} ${HDR})
target_link_libraries(ResNet dlib::dlib ${ITK_LIBRARIES} ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARIES} ${Boost_SYSTEM_LIBRARIES} Boost::thread)