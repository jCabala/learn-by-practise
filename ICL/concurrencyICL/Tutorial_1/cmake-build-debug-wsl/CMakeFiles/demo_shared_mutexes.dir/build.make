# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_SOURCE_DIR = /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/cmake-build-debug-wsl

# Include any dependencies generated for this target.
include CMakeFiles/demo_shared_mutexes.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/demo_shared_mutexes.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/demo_shared_mutexes.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo_shared_mutexes.dir/flags.make

CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o: CMakeFiles/demo_shared_mutexes.dir/flags.make
CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o: ../src/shared_mutexes/demo_shared_mutexes.cc
CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o: CMakeFiles/demo_shared_mutexes.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/cmake-build-debug-wsl/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o -MF CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o.d -o CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o -c /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/src/shared_mutexes/demo_shared_mutexes.cc

CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/src/shared_mutexes/demo_shared_mutexes.cc > CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.i

CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/src/shared_mutexes/demo_shared_mutexes.cc -o CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.s

# Object files for target demo_shared_mutexes
demo_shared_mutexes_OBJECTS = \
"CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o"

# External object files for target demo_shared_mutexes
demo_shared_mutexes_EXTERNAL_OBJECTS =

demo_shared_mutexes: CMakeFiles/demo_shared_mutexes.dir/src/shared_mutexes/demo_shared_mutexes.cc.o
demo_shared_mutexes: CMakeFiles/demo_shared_mutexes.dir/build.make
demo_shared_mutexes: CMakeFiles/demo_shared_mutexes.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/cmake-build-debug-wsl/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_shared_mutexes"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_shared_mutexes.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo_shared_mutexes.dir/build: demo_shared_mutexes
.PHONY : CMakeFiles/demo_shared_mutexes.dir/build

CMakeFiles/demo_shared_mutexes.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo_shared_mutexes.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo_shared_mutexes.dir/clean

CMakeFiles/demo_shared_mutexes.dir/depend:
	cd /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/cmake-build-debug-wsl && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1 /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1 /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/cmake-build-debug-wsl /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/cmake-build-debug-wsl /home/jcabala/learn-by-practise/ICL/concurrencyICL/Tutorial_1/cmake-build-debug-wsl/CMakeFiles/demo_shared_mutexes.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo_shared_mutexes.dir/depend

