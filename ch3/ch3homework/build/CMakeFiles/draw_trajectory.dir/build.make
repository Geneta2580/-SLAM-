# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/geneta/Project/homework/ch3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/geneta/Project/homework/ch3/build

# Include any dependencies generated for this target.
include CMakeFiles/draw_trajectory.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/draw_trajectory.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/draw_trajectory.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/draw_trajectory.dir/flags.make

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o: CMakeFiles/draw_trajectory.dir/flags.make
CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o: /home/geneta/Project/homework/ch3/draw_trajectory.cpp
CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o: CMakeFiles/draw_trajectory.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/geneta/Project/homework/ch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o -MF CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o.d -o CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o -c /home/geneta/Project/homework/ch3/draw_trajectory.cpp

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/geneta/Project/homework/ch3/draw_trajectory.cpp > CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/geneta/Project/homework/ch3/draw_trajectory.cpp -o CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s

# Object files for target draw_trajectory
draw_trajectory_OBJECTS = \
"CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o"

# External object files for target draw_trajectory
draw_trajectory_EXTERNAL_OBJECTS =

draw_trajectory: CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o
draw_trajectory: CMakeFiles/draw_trajectory.dir/build.make
draw_trajectory: /usr/local/lib/libpango_glgeometry.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_plot.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_python.so
draw_trajectory: /usr/local/lib/libpango_scene.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_tools.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_video.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_geometry.so.0.9.3
draw_trajectory: /usr/local/lib/libtinyobj.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_display.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_vars.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_windowing.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_opengl.so.0.9.3
draw_trajectory: /usr/lib/x86_64-linux-gnu/libEGL.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libOpenGL.so
draw_trajectory: /usr/lib/x86_64-linux-gnu/libepoxy.so
draw_trajectory: /usr/local/lib/libpango_image.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_packetstream.so.0.9.3
draw_trajectory: /usr/local/lib/libpango_core.so.0.9.3
draw_trajectory: CMakeFiles/draw_trajectory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/geneta/Project/homework/ch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable draw_trajectory"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/draw_trajectory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/draw_trajectory.dir/build: draw_trajectory
.PHONY : CMakeFiles/draw_trajectory.dir/build

CMakeFiles/draw_trajectory.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/draw_trajectory.dir/cmake_clean.cmake
.PHONY : CMakeFiles/draw_trajectory.dir/clean

CMakeFiles/draw_trajectory.dir/depend:
	cd /home/geneta/Project/homework/ch3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/geneta/Project/homework/ch3 /home/geneta/Project/homework/ch3 /home/geneta/Project/homework/ch3/build /home/geneta/Project/homework/ch3/build /home/geneta/Project/homework/ch3/build/CMakeFiles/draw_trajectory.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/draw_trajectory.dir/depend

