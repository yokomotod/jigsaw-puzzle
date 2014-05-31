cmake_minimum_required (VERSION 2.6)

project( puzzle )
find_package( OpenCV REQUIRED )
add_executable( puzzle puzzle.cc ) 
target_link_libraries( puzzle ${OpenCV_LIBS} )
