cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 20)

project(BenchmarkGames LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-unused-parameter")

set(SOURCE_FILES
        src/main.cpp
        src/Arena.cpp
        src/Agents/Mcts/MctsAgent.cpp
        src/Agents/Mcts/MctsNode.cpp
        src/Agents/RandomAgent.cpp
        include/Games/Gamestate.h
        include/Agents/Agent.h
        include/Agents/Mcts/MctsAgent.h
        include/Agents/Mcts/MctsNode.h
        include/Agents/RandomAgent.h
        include/Arena.h
        include/Agents/HumanAgent.h
        src/Agents/HumanAgent.cpp
        src/Games/MDPs/Navigation.cpp
        include/Games/MDPs/JoinFive.h
        src/Games/MDPs/JoinFive.cpp
        src/Games/MDPs/Navigation.cpp
        include/Games/MDPs/SailingWind.h
        src/Games/MDPs/SailingWind.cpp
        include/Utils/UnitTest.h
        include/Utils/Argparse.h
        include/Agents/SparseSamplingAgent.h
        src/Agents/SparseSamplingAgent.cpp
        include/Utils/Distributions.h
        src/Utils/Distributions.cpp
        include/Utils/ValueIteration.h
        src/Utils/ValueIteration.cpp
        include/Agents/LookupTableAgent.h
        src/Agents/LookupTableAgent.cpp
        include/Agents/FFHindsightAgent.h
        src/Agents/FFHindsightAgent.cpp
        include/Agents/Asap/AsapNode.h
        include/Agents/Asap/AsapAgent.h
        src/Agents/Asap/AsapNode.cpp
        src/Agents/Asap/AsapAgent.cpp
        include/Games/Wrapper/MultiPlayerToMDP.h
        src/Games/Wrapper/MultiPlayerToMDP.cpp
        include/Games/Wrapper/FiniteHorizon.h
        src/Games/Wrapper/FiniteHorizon.cpp
        include/Utils/MiscAnalysis.h
        include/Games/MDPs/JoinFive.h
        src/Games/MDPs/JoinFive.cpp
        include/Utils/MemoryAnalysis.h
        src/Utils/MemoryAnalysis.cpp
        src/demo.cpp
        include/Agents/OneStepLookahead.h
        src/Agents/OneStepLookaheadAgent.cpp
        src/Utils/CLink.cpp
        include/Utils/ModelMaker.h
        src/Utils/ModelMaker.cpp
        include/Utils/AgentMaker.h
        src/Utils/AgentMaker.cpp
        include/Games/Wrapper/RandomStart.h
        src/Games/Wrapper/RandomStart.cpp
        include/Games/Wrapper/Determinization.h
        src/Games/Wrapper/Determinization.cpp
        include/Games/Wrapper/HeuristicsAsReward.h
        src/Games/Wrapper/HeuristicsAsReward.cpp
        include/Agents/Mcts/NestedMctsAgent.h
        src/Agents/Mcts/NestedMctsAgent.cpp
)

#Shared lib
add_library(liblink SHARED ${SOURCE_FILES}
        include/Agents/Mcts/NestedMctsAgent.h)
target_compile_options(liblink PRIVATE
        -Wall
        -fPIC
        -O3
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # For MinGW / GCC
    target_link_options(liblink PRIVATE
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:-static>
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:-static-libgcc>
            $<$<STREQUAL:${CMAKE_SYSTEM_NAME},Windows>:-static-libstdc++>
    )
    message(STATUS "Using GCC/MinGW compiler")
else()
    message(FATAL_ERROR "Compiler '${CMAKE_CXX_COMPILER_ID}' is not supported by this build.")
endif()
set_target_properties(liblink PROPERTIES OUTPUT_NAME "liblink")

#Debug executable
add_executable(BenchmarkGamesDebug ${SOURCE_FILES}
        include/Agents/Mcts/NestedMctsAgent.h
        src/Agents/Mcts/NestedMctsAgent.cpp)
target_compile_options(BenchmarkGamesDebug PRIVATE
        ${CMAKE_CXX_FLAGS_DEBUG}   # typical debug flags (-O0 -g for gcc/clang)
)
target_link_options(BenchmarkGamesDebug PRIVATE
        ${CMAKE_EXE_LINKER_FLAGS}         # global linker flags
        ${CMAKE_EXE_LINKER_FLAGS_DEBUG}   # typical debug linker flags
)

#Release executable
add_executable(BenchmarkGamesRelease ${SOURCE_FILES}
        include/Agents/Mcts/NestedMctsAgent.h
        src/Agents/Mcts/NestedMctsAgent.cpp)
target_compile_options(BenchmarkGamesRelease PRIVATE
        -O3
        -DNDEBUG
)
target_link_options(BenchmarkGamesRelease PRIVATE
        ${CMAKE_EXE_LINKER_FLAGS}
        ${CMAKE_EXE_LINKER_FLAGS_RELEASE}
)