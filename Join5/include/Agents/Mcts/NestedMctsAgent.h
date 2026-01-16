//
// Created by Manh Phuoc Johnny Ta on 15.09.25.
//
#pragma once
#ifndef BENCHMARKGAMES_NESTEDMCTSAGENT_H
#define BENCHMARKGAMES_NESTEDMCTSAGENT_H
#include "../Agent.h"
#include <random>
#include <vector>
#include <cmath>
#include <fstream>

#include "MctsAgent.h"

class NestedMctsAgent final: public Agent {
public:
    NestedMctsAgent(int level, int horizon, int budget, std::string& heuristic);
    int getAction(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng) override;
    std::vector<int> getTrajectory(ABS::Model *model, ABS::Gamestate *gamestate, std::mt19937 &rng) override;
private:
    int level;
    int horizon; // rollout length
    int budget;
    std::string heuristic;

    // For stats
#if true
    struct BranchingStats {
        long long count = 0;           // N: sum of all steps (samples)
        long long sum_branching = 0;   // sum
        double sum_sq_branching = 0.0; // sum x^2 (for variance calc)
    } current_stats;

    std::ofstream csv_file;
#endif

    // returns (score, sequence of moves)
    std::pair<double, std::vector<int>> nmcs(int lvl, ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng, int horizonCount);
    std::pair<double, std::vector<int>> playout(ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng, int horizonCount);
};

#endif //BENCHMARKGAMES_NESTEDMCTSAGENT_H

