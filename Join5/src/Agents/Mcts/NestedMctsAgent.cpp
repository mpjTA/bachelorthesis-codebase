
#include "../../../include/Agents/Mcts/NestedMctsAgent.h"
#include "../../../include/Games/MDPs/JoinFive.h"
#include <random>
#ifndef NMCS_ENABLE_STATS
#define NMCS_ENABLE_STATS 0  // set to 0 to disable stats at compile time
#endif

//------ static global variable for stats -------------
#if NMCS_ENABLE_STATS
static int GLOBAL_GAME_ID_COUNTER = 0;
static std::ofstream GLOBAL_CSV_FILE;
static bool IS_FILE_INITIALIZED = false;
#endif
//-----------------------------------------------------

// helper
static int uniform_dist(std::mt19937& rng, int n) {
    assert(n > 0);
    std::uniform_int_distribution<int> dist(0, n-1);
    return dist(rng);
}
// helper
inline std::vector<double> softmax(const std::vector<std::pair<int,int>>& weights) {
    std::vector<double> probs;

    auto max_w = static_cast<double>(weights.front().second);
    for (auto& w : weights)
        if (w.second > max_w) max_w = static_cast<double>(w.second);

    double sum = 0;
    for(auto weight : weights) {
        double pr = std::exp((weight.second - max_w)/0.75);
        sum += pr;
        probs.emplace_back(pr);
    }
    for(auto& pr: probs) {
        pr /= sum;
    }
    return probs;
}
//helper
inline int sample_index(const std::vector<double>& probabilities, std::mt19937& rng) {

    std::uniform_real_distribution<double> dist(0.0, 1.0); // uniform 0.0 to 1.0
    double r = dist(rng);
    double cumulative = 0.0;

    for (size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (r <= cumulative) {
            return static_cast<int>(i);
        }
    }

    // Fallback: return first index just in case
    return 0;
}


// Constructor
NestedMctsAgent::NestedMctsAgent(int level, int horizon, int budget, std::string& heuristic): level(level), horizon(horizon), budget(budget), heuristic(heuristic) {
};

// DEPRECATED
inline int mobility(ABS::Model *model, ABS::Gamestate *state, int action, std::mt19937 &rng) {
    ABS::Gamestate* s = model->copyState(state);
    if (s->terminal) {delete s; return 0;}
    model->applyAction(s, action, rng, nullptr);
    int result;
    if (!s->terminal) {
        result = static_cast<int>(model->getActions(s).size());
        if (result < 0) result = 0;
    }else {
        result = 0;
    }
    delete s;
    return result;
}
// DEPRECATED
// returns number of legal next actions and writes new created actions by previous action in newActions
inline int newMobility(ABS::Model *model, ABS::Gamestate *state, int action, std::mt19937 &rng, std::vector<int>& newActions) {
    ABS::Gamestate* s = model->copyState(state);
    auto before = model->getActions(s);
    std::unordered_set<int> copiedActions(before.begin(), before.end()); // previous actions
    try {
        if (s->terminal) {delete s; newActions.clear(); return 0;}
        model->applyAction(s, action, rng, nullptr); // simulating play using copied state
        int result;
        if (!s->terminal) {

            newActions = model->getActions(s); // actions after simulating play
            // removes all actions in "newActions" that are in previous list of actions "before" + action that led to current state
            newActions.erase(std::remove_if(newActions.begin(), newActions.end(), [&](int const x){
                //std::cout << copiedActions.count(x) << std::endl; // DEBUG
                return (copiedActions.find(x) != copiedActions.end()) || (x == action);
;
            }), newActions.end());

            result = static_cast<int>(model->getActions(s).size());
            if (result < 0) result = 0;
        }else {
            result = 0;
        }
        delete s;
        return result;
    } catch (...) {
        delete s;
        return 0;
    }
}


// Combine heuristic
inline int combinedHeuristic(ABS::Model *model, ABS::Gamestate *state, std::mt19937 &rng) {
    std::vector<int> copyActions = model->getActions(state);
    const int originalSize = copyActions.size();
    // 4 actions with best CrossRay score
    if (uniform_dist(rng,10) > 0) { // Probability of 90%: Sample (without replacement) 4 actions and chose the one with highest CrossRay score. Tiebreak by total mobility
        const std::size_t k = std::min<std::size_t>(4, copyActions.size());
        std::partial_sort(
            copyActions.begin(),
            copyActions.begin() + static_cast<int>(k),
            copyActions.end(),
            [&](const int x, const int y) {
                return model->crossWindow(state, x) > model->crossWindow(state, y);
            }
        );
        copyActions.resize(k);

        int bestAction = copyActions.front();
        int bestScore  = originalSize + model->mobilityIf(state, bestAction).net();
        for (std::size_t i = 1; i < copyActions.size(); ++i) {
            const int action = copyActions[i];
            const int cand = originalSize + model->mobilityIf(state, action).net();
            if (cand > bestScore) {
                bestScore = cand;
                bestAction = action;
            }
        }
        return bestAction;
    }
    else { // Probability of 10%: Sample (with replacement) random 5 actions and take the one with the highest total mobility. Tiebreak by random.
        std::vector<int> samples;
        samples.reserve(std::min<int>(5, originalSize));
        const int sampleSize = std::min<int>(5, originalSize);
        std::sample(copyActions.begin(), copyActions.end(),std::back_inserter(samples), sampleSize,rng);
        int bestAction = samples.front();
        auto bestVal= model->mobilityIf(state, bestAction).net();

        for (std::size_t i = 1; i < samples.size(); ++i) {
            const int action = samples[i];
            const auto val= model->mobilityIf(state, action).net();
            if (val > bestVal) {
                bestVal = val;
                bestAction = action;
            }
        }
        return bestAction;
    }
}

// Rosin´s heuristic, which returns chosen action
inline int rosin(ABS::Model& model, ABS::Gamestate& state, std::mt19937 &rng, std::vector<int>& actions, std::vector<int>& newActions, int prev_action) {
    if (!newActions.empty()) { // 1. Sample 4 newly created moves and choose move with the best mobility
        // Sample with replacement
        std::vector<int> result;
        int bestMobility = -1;
        int mobilityScore = -1;
        int sampleSize = 4;
        for(int i = 0; i < sampleSize; ++i){
            int idx = uniform_dist(rng, static_cast<int>(newActions.size()));
            int current = newActions[idx];
            int currentMobility = /*mobility(&model, &state, current, rng);*/ static_cast<int>(actions.size()+(model.mobilityIf(&state, current)).net());
            if (currentMobility > mobilityScore) {
                mobilityScore = currentMobility;
                bestMobility = current;
            }
        }
        return bestMobility;
    }
    //if with probability of 90% ...
    if (0 < uniform_dist(rng, 10) && prev_action != -1) {
        int sample[2]; // sample
        std::vector<int> small_distance_actions;
        for (const auto& action : actions) {
            if (model.mDistance(&state, prev_action, action) <= 2.5) { // push all actions with distance (rounded) 2 or smaller
                small_distance_actions.emplace_back(action);
            }
        }
        // sample 2 with replacement
        sample[0] = small_distance_actions.at(uniform_dist(rng, 2));
        sample[1] = small_distance_actions.at(uniform_dist(rng, 2));

        // return move yielding the highest number of total legal moves (mobility)
        return /*mobility(&model, &state, sample[0], rng)*/ static_cast<int>(actions.size()+(model.mobilityIf(&state, sample[0])).net()) > /*mobility(&model, &state, sample[1], rng)*/ static_cast<int>(actions.size()+(model.mobilityIf(&state, sample[1])).net()) ? sample[0] : sample[1];
    }
    // 3. with probability of 10%
        std::vector<int> result;
        std::vector<int> samples;
        // sample 6 actions
        int sampleSize = 6;
        std::sample(actions.begin(), actions.end(), std::back_inserter(samples), sampleSize, rng);
        auto [max, min] =
            std::minmax_element(samples.begin(), samples.end(), [&](int const action1, int const action2) -> bool {
                // comp newMobility of action1 and action2
                //std::vector<int> temp; // new created moves by action1
                //std::vector<int> temp2; // new created moves by action2
                //newMobility(&model, &state, action1, rng, temp);
                //newMobility(&model, &state, action2, rng, temp2);
                const auto model_temp = (model.mobilityIf(&state, action1));
                const auto model_temp2 = model.mobilityIf(&state, action2);
                std::vector<int> temp = model_temp.added_actions;
                std::vector<int> temp2 = model_temp2.added_actions;
                if (temp.size() == temp2.size()) {
                    return model_temp.net() > model_temp2.net();
                }
                return temp.size() > temp2.size();
        });
        return *max;
}

inline int potential(ABS::Model *model, ABS::Gamestate *state, int action) {
    return model->potentialScore(state, action);
}

inline int crossWindowHeuristic(ABS::Model *model, ABS::Gamestate *state, int action) {
    return model->crossWindow(state, action);
}

inline double touchHeuristic(ABS::Model *model, ABS::Gamestate *state, int action) {
    return model->touch(state, action);
}

std::pair<double, std::vector<int>> NestedMctsAgent::playout(ABS::Model *model, ABS::Gamestate *state, std::mt19937 &rng, int horizonCount) {

    // copy of state
    const auto s = model->copyState(state);
    std::vector<int> seq; // Sequence of moves
    double score = 0.0;

// for Rosins´s heuristic (and newMobility)
    std::vector<int> new_actions_created;
    int previous_action = -1;

#if NMCS_ENABLE_STATS
    int internal_step = 0; // for stats: this-level == 0, which means hyperparameter nesting level 0
#endif

    while (horizonCount != 0 && !s->terminal) {
        auto available_actions = model->getActions(s);

        //----- For stats --------
#if NMCS_ENABLE_STATS
        long branching_factor = available_actions.size();

        // update stats
        if (this->level == 0 && GLOBAL_CSV_FILE.is_open()) {
            GLOBAL_CSV_FILE << GLOBAL_GAME_ID_COUNTER << ";"
                            << internal_step << ";"
                            << branching_factor << ";"  // Mean
                            << branching_factor << ";"  // CI Lower
                            << branching_factor << ";"  // CI Upper
                            << 1 << "\n";               // N_Samples = 1
            GLOBAL_CSV_FILE.flush();
        }else if (this->level > 0) {
            current_stats.count++;
            current_stats.sum_branching += branching_factor;
            current_stats.sum_sq_branching += (double)branching_factor * branching_factor;
        }
#endif
        //------------------------

        if (available_actions.empty()) break;

        // choose action (randomly)
        auto action = -1;
        if (this->heuristic.empty() || this->heuristic == "rand") {action = available_actions.at(uniform_dist(rng, static_cast<int>(available_actions.size())));}

        // Combined Heuristic (inspired by Rosin)
        if (this->heuristic == "combine") { action = combinedHeuristic(model, s, rng);}

        if (this->heuristic == "mob") { //// Mobility Heuristic
            // Shuffle to avoid potential bias in available_actions
            auto shuffled_actions = available_actions;
            std::shuffle(shuffled_actions.begin(), shuffled_actions.end(), rng);
            action = shuffled_actions.at(uniform_dist(rng, static_cast<int>(shuffled_actions.size())));
            int best = static_cast<int>(shuffled_actions.size()+(model->mobilityIf(s, action)).net());
            for (auto a : shuffled_actions) {
                if (int current = static_cast<int>(shuffled_actions.size()+(model->mobilityIf(s, a)).net()); current > best) {
                    //assert(mobility(model, s, a, rng) == static_cast<int>(shuffled_actions.size()+(model->mobilityIf(s, a)).net()));
                    best = current;
                    action = a;
                }
            }
        }    //// Mobility end


        // potential heuristic value is calculated here with 40 - sum of deg
        if (this->heuristic == "pot") { //// Potential Heuristic
            constexpr int MAXDEG = 40; // 5 crosses * 8 half directions

            // Shuffle to avoid potential bias in available_actions
            auto shuffled_actions = available_actions;
            std::shuffle(shuffled_actions.begin(), shuffled_actions.end(), rng);
            action = shuffled_actions.at(uniform_dist(rng, static_cast<int>(shuffled_actions.size())));

            int best = MAXDEG - potential(model, s, action);
            for (const auto a : shuffled_actions) {
                int current = MAXDEG - potential(model, s, a);
                //std::cout << a << " " << current << std::endl;
                if (current > best) { // "<" would be Opposite-Potential
                    best = current;
                    action = a;
                }
            }
            //std::cout << "potentialScore: "<< best << std::endl;
        }  /// Potential end


        if (this->heuristic == "opp-pot") { //// Opposite-Potential Heuristic
            constexpr int MAXDEG = 40; // 5 crosses * 8 half directions

            // Shuffle to avoid potential bias in available_actions
            auto shuffled_actions = available_actions;
            std::shuffle(shuffled_actions.begin(), shuffled_actions.end(), rng);
            action = shuffled_actions.at(uniform_dist(rng, static_cast<int>(shuffled_actions.size())));

            int best = MAXDEG - potential(model, s, action);
            for (const auto a : shuffled_actions) {
                int current = MAXDEG - potential(model, s, a);
                if (current < best) {
                    best = current;
                    action = a;
                }
            }

        }  /// Opposite-Potential end

        if (this->heuristic == "cross") { ///// Cross Window Heuristic, in thesis refered as CrossRay
            // Shuffle to avoid potential bias in available_actions
            auto shuffled_actions = available_actions;
            std::shuffle(shuffled_actions.begin(), shuffled_actions.end(), rng);
            action = shuffled_actions.at(uniform_dist(rng, static_cast<int>(shuffled_actions.size())));

            int best = crossWindowHeuristic(model, s, action);
            for (const auto a: shuffled_actions){
                int current = crossWindowHeuristic(model, s, a);
                if (current > best) {
                    best = current;
                    action = a;
                    //std::cout << action <<" Best: "<< best << " and "<< first << " First: " << first_score << std::endl;
                }
            }
        }  ///// Cross Window end


#if false
        auto shuffled_actions = available_actions;
        std::shuffle(shuffled_actions.begin(), shuffled_actions.end(), rng);
        int action = shuffled_actions.at(uniform_dist(rng, static_cast<int>(shuffled_actions.size())));

        std::vector<std::pair<int, int>> action_value;
        for (const auto& a : shuffled_actions) {
                int value = crossWindowHeuristic(model, s, a);
                action_value.emplace_back(action, value);
            }
        std::sort(action_value.begin(), action_value.end(),
            [](std::pair<int,int> const &action_value1, std::pair<int,int> const &action_value2) {
            return action_value1.second > action_value2.second;
        });
        if (action_value.size() > 2) {
            std::vector<std::pair<int,int>> sample5(action_value.begin(), action_value.begin() + 2);
            //for (auto a : sample5) {std::cout <<a.second << " ";}
            //std::cout << std::endl;
            std::vector<double> probs = softmax(sample5);
            //for (auto a : probs) {std::cout << a << " ";}
            //std::cout << std::endl;
            int chosen_index = sample_index(probs, rng);
            action = sample5[chosen_index].first;
        }else {
            std::vector<double> probs = softmax(action_value);
            int chosen_index = sample_index(probs, rng);
            action = action_value[chosen_index].first;
        }
#endif

#if false
        int action;
        int best = -1;
        for (auto a : available_actions) {
            int current = crossWindowHeuristic(model, s, a);
            if (current > best) {
                best = current;
                action = a;
                //std::cout << a <<" Best: "<< best << std::endl;
            }
        }
        std::vector<int> tie_list;
        for (auto a : available_actions) {
            int current = crossWindowHeuristic(model, s, a);
            if (current == best) {
                tie_list.emplace_back(a);
            }
        }
        action = tie_list.at(uniform_dist(rng, static_cast<int>(tie_list.size())));
#endif

        if (this->heuristic == "touch") { //// Touch
            // Shuffle to avoid potential bias in available_actions
            auto shuffled_actions = available_actions;
            std::shuffle(shuffled_actions.begin(), shuffled_actions.end(), rng);
            action = shuffled_actions.at(uniform_dist(rng, static_cast<int>(shuffled_actions.size())));

            double best = touchHeuristic(model, s, action);
            for (auto a : shuffled_actions) {
                double current = touchHeuristic(model, s, a);
                if (current > best) {
                    best = current;
                    action = a;
                    //std::cout << a <<" Best: "<< best << std::endl;
                }
            }
        } ///// Touch end




#if false ///// newMobility Heuristic
        // Shuffle to avoid potential bias in available_actions
        auto shuffled_actions = available_actions;
        std::shuffle(shuffled_actions.begin(), shuffled_actions.end(), rng);
        int action = shuffled_actions.at(uniform_dist(rng, static_cast<int>(shuffled_actions.size())));
        previous_action = action;
        int best = newMobility(model, s, action, rng, new_actions_created);
        std::vector<int> tmp;
        for (auto a : shuffled_actions) {
            int current = newMobility(model, s, a, rng, tmp);
            if (current > best) {
                best = current;
                action = a;
                previous_action = a;
                new_actions_created = tmp;
            }
        }
        assert(previous_action != -1);
#endif ///// newMobility end

        if (this->heuristic == "rosin") { // Rosin´s heuristic
            // Shuffle to avoid potential bias in available_actions
            auto shuffled_actions = available_actions;
            std::shuffle(shuffled_actions.begin(), shuffled_actions.end(), rng);
            action = rosin(*model, *s, rng, shuffled_actions, new_actions_created, previous_action);
        } // Rosin´s heuristic end


        // apply action and get score (single player), which is stored in vector.
        assert(model->getNumPlayers() == 1);
        if (s->terminal) break;
        score += model->applyAction(s, action, rng, nullptr).first.front();

        seq.emplace_back(action);

        if (horizonCount > 0) --horizonCount;
#if NMCS_ENABLE_STATS
        internal_step++;
#endif
    }

    //--------- for stats nesting level 0 as hyperparameter, last step -----------
#if NMCS_ENABLE_STATS
    if (this->level == 0 && GLOBAL_CSV_FILE.is_open()) {
        GLOBAL_CSV_FILE << GLOBAL_GAME_ID_COUNTER << ";"
                        << internal_step << ";"  // final step / terminal step
                        << "0;0;0;0" << "\n";    // avail actions is 0
        GLOBAL_CSV_FILE.flush();
    }
#endif
    // --------------------------------------

    delete s;
    return std::make_pair(score, seq);
}

#if false
std::pair<double, std::vector<int> > NestedMctsAgent::nmcs(int lvl, ABS::Model *model, ABS::Gamestate *state, std::mt19937 &rng, int horizonCount) {

    // rollout/playout at level 0
    if (lvl == 0) return NestedMctsAgent::playout(model, state, rng, horizonCount);


     // for lvl > 0:
    if (model->getActions(state).empty() || state->terminal) {
        return {0.0, {}};
    }

    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<int> best_seq;


    for (const auto action: model->getActions(state)) {
        // copy state
        auto s = model->copyState(state);
        // apply action to copied state (does not interfere with original state)
        auto reward = model->applyAction(s, action, rng, nullptr);
        // call nmcs with decreased lvl
        auto [score, seq] = nmcs(lvl - 1, model, s, rng, horizonCount);

        reward.first.empty() ? 0.0 : score += reward.first.front(); // consider reward after action!

        delete s;

        if (score > best_score) {
            best_score = score;
            best_seq.clear();
            best_seq.emplace_back(action);
            best_seq.insert(best_seq.end(), seq.begin(), seq.end());
        }
    }
    return {best_score, best_seq};
}
#endif


#if true // with memorization
std::pair<double, std::vector<int> > NestedMctsAgent::nmcs(int lvl, ABS::Model* model, ABS::Gamestate* state, std::mt19937& rng, int horizonCount){
    // rollout/playout at level 0
    if (lvl == 0) return NestedMctsAgent::playout(model, state, rng, horizonCount);

    // else: level >= 1
    if (state->terminal || model->getActions(state).empty()) {
        return {0.0, {}};
    }

    // working state (copy of original state)
    ABS::Gamestate* s = model->copyState(state);

    double total_score = 0.0;
    std::vector<int> seq;

#if NMCS_ENABLE_STATS
    int internal_step = 0; // for stats
#endif

    //-------Memorization-------
    std::vector<int> memo_suffix;
    size_t memo_pos = 0;            // current "front" of memo plan
    double memo_score = -1.0;       // score of memo plan starting at memo_pos
    //--------------------------

    while (!s->terminal){

        // Root-level (highest nesting level n): reset stats
#if NMCS_ENABLE_STATS
        if (lvl == this->level) {
            current_stats = BranchingStats();
        }
#endif

        auto actions = model->getActions(s);
        if (actions.empty()) break;

        double best_val = -1;
        int best_action = -1;
        //-------Memorization-------
        std::vector<int> best_suf_seq;
        bool best_from_memo = false;

        // Use memo as the initial "best" candidate
        if(memo_pos < memo_suffix.size()){
            best_action = memo_suffix[memo_pos];
            best_val = memo_score;
            best_from_memo = true;
        }
        //-------------------------


        for (auto const& a : actions){
            // copy for simulating action
            ABS::Gamestate* sc = model->copyState(s);

            // rewards
            auto r_immediate = model->applyAction(sc, a, rng, nullptr);
            double immediate = r_immediate.first.empty() ? 0.0 : r_immediate.first.front();

            // action seq after current state
            auto [suf_score, suf_seq] = nmcs(lvl - 1, model, sc, rng, horizonCount);

            double cand = immediate + suf_score;
            if (cand > best_val){
                best_val = cand;
                best_action = a;
                //-------Memorization-------
                // memorize the continuation from next state
                best_suf_seq = std::move(suf_seq);
                best_from_memo = false;
                //--------------------------
            }
            delete sc;
        }

        //------ For stats ---------
#if NMCS_ENABLE_STATS
        if (lvl == this->level && GLOBAL_CSV_FILE.is_open()) {
            long long n = current_stats.count;

            if (n > 0) {
                double mean = (double)current_stats.sum_branching / n;
                double ci_lower = mean;
                double ci_upper = mean;

                if (n > 1) {
                    double variance = (current_stats.sum_sq_branching - (current_stats.sum_branching * current_stats.sum_branching / (double)n)) / n;
                    if (variance < 0) variance = 0;
                    double std_error = std::sqrt(variance) / std::sqrt(n);
                    double margin = 1.96 * std_error;
                    ci_lower -= margin;
                    ci_upper += margin;
                }


                GLOBAL_CSV_FILE << GLOBAL_GAME_ID_COUNTER << ";"
                                << internal_step << ";"
                                << mean << ";"
                                << ci_lower << ";"
                                << ci_upper << ";"
                                << n << "\n";
                GLOBAL_CSV_FILE.flush();
            }
            else if (n == 0) {
                // if during search (level > 0) encounter terminal state -> no rollout but still acquire stat
                GLOBAL_CSV_FILE << GLOBAL_GAME_ID_COUNTER << ";"
                                << internal_step << ";"
                                << actions.size() << ";"
                                << actions.size() << ";" // placeholder
                                << actions.size()<< ";" // place holder
                                << "0\n"; // 0 because no rollout wis played

                GLOBAL_CSV_FILE.flush();
            }
        }
#endif
        //--------------------------

        if (best_action == -1) break;

        // apply best action and add rewards
        auto r_commit = model->applyAction(s, best_action, rng, nullptr);
        double immediate_commit = r_commit.first.empty() ? 0.0 : r_commit.first.front();

        total_score += immediate_commit;
        seq.push_back(best_action);
        // update memorised continuation for NEXT state:
        if(best_from_memo){
            // consume one action from memo plan
            memo_pos++;
            memo_score = best_val - immediate_commit;
        }else{
            memo_suffix = std::move(best_suf_seq);
            memo_pos = 0;
            memo_score = best_val - immediate_commit;
        }
#if NMCS_ENABLE_STATS
        internal_step++; // for stats
#endif
    }
    //------- for stats -----------
#if NMCS_ENABLE_STATS
    if (lvl == this->level && GLOBAL_CSV_FILE.is_open()) {
        GLOBAL_CSV_FILE << GLOBAL_GAME_ID_COUNTER << ";"
                        << internal_step << ";"  // last step
                        << "0;0;0;0" << "\n";    // avail action is 0
        GLOBAL_CSV_FILE.flush();
    }
#endif
    //--------------------------------------------------------------------

    delete s;

    return {total_score, seq};
}
#endif

int NestedMctsAgent::getAction(ABS::Model *model, ABS::Gamestate *state, std::mt19937 &rng) {
    if (state->terminal) return -1;
    auto root_state = model->copyState(state);

    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<int> best_seq;

    int remaining = budget;
    while (remaining != 0) {
        auto [score, seq] = nmcs(this->level, model, root_state, rng, this->horizon);
        if (score > best_score) {
            best_score = score;
            best_seq = std::move(seq);
        }
        --remaining;
    }
    //model->printState(state);
    delete root_state;
    //return best_seq.front();
    if (!best_seq.empty()) return best_seq.front();

    // Fallback
    return 0;
}
// ==============================================

#include <bitset>


// ======================= helper + bbox density =======================

#if NMCS_ENABLE_STATS
static std::ofstream GLOBAL_DENSITY_FILE;
static bool IS_DENSITY_FILE_INITIALIZED = false;
static int DENSITY_FIRST_X_MOVES = 40; // first X committed moves to log

struct DotDensityBBox{
    int added_crosses = 0;
    int min_r = 0, max_r = -1;
    int min_c = 0, max_c = -1;
    int bbox_area = 0;
    double bbox_density = 0.0;
};

// Works for BOTH: plain JoinFive state OR FINITEH-wrapped state.
static J5::Gamestate* asJoinFiveState(ABS::Gamestate* s){
    if (!s) return nullptr;

    if (auto* js = dynamic_cast<J5::Gamestate*>(s)) {
        return js;
    }
    if (auto* fh = dynamic_cast<FINITEH::Gamestate*>(s)) {
        return dynamic_cast<J5::Gamestate*>(fh->ground_state);
    }
    return nullptr;
}

// Bounding-box density over crosses that are in current state but not in baseline (initial configuration).
static DotDensityBBox bboxDensityExcludingBaseline(
    const J5::Gamestate* s,
    const std::bitset<J5::SIZE*J5::SIZE>& baseline_crosses
){
    DotDensityBBox d;
    if (!s) return d;

    const int N = J5::SIZE;
    d.min_r = N; d.min_c = N;
    d.max_r = -1; d.max_c = -1;

    for (int idx = 0; idx < N * N; ++idx) {
        if (!s->crosses.test(idx)) continue;
        if (baseline_crosses.test(idx)) continue;

        int r = idx / N;
        int c = idx % N;

        d.added_crosses++;

        if (r < d.min_r) d.min_r = r;
        if (r > d.max_r) d.max_r = r;
        if (c < d.min_c) d.min_c = c;
        if (c > d.max_c) d.max_c = c;
    }

    if (d.added_crosses == 0) {
        d.bbox_area = 0;
        d.bbox_density = 0.0;
        return d;
    }

    int h = (d.max_r - d.min_r + 1);
    int w = (d.max_c - d.min_c + 1);
    d.bbox_area = h * w;
    d.bbox_density = (d.bbox_area > 0) ? (double)d.added_crosses / (double)d.bbox_area : 0.0;
    return d;
}
#endif


// ======================= density logging block =======================

std::vector<int> NestedMctsAgent::getTrajectory(ABS::Model *model, ABS::Gamestate *state, std::mt19937 &rng) {
    //------------ for branching stats -------------
#if NMCS_ENABLE_STATS
    if (!IS_FILE_INITIALIZED) {
        const std::string filename = "branching_stats_" + std::string(this->heuristic) + "_" + std::to_string(this->level) + ".csv";
        GLOBAL_CSV_FILE.open(filename);
        GLOBAL_CSV_FILE << "GameID;Step;AvgBranching;CI_Lower;CI_Upper;N_Samples\n";
        IS_FILE_INITIALIZED = true;
    }
#endif
    //------------------------------------

    //------------ for bbox density stats -------------
#if NMCS_ENABLE_STATS
    if (!IS_DENSITY_FILE_INITIALIZED) {
        const std::string filename = "bbox_density_excl_init_" + std::string(this->heuristic) + "_" + std::to_string(this->level) + ".csv";
        GLOBAL_DENSITY_FILE.open(filename);
        GLOBAL_DENSITY_FILE << "GameID;Step;AddedCrosses;BBoxArea;BBoxDensity\n";
        IS_DENSITY_FILE_INITIALIZED = true;
    }
#endif
    //------------------------------------

    if (state->terminal) return std::vector<int>{};
    auto root_state = model->copyState(state);

    double best_score = -std::numeric_limits<double>::infinity();
    std::vector<int> best_seq;

    int remaining = budget;
    while (remaining != 0) {
        auto [score, seq] = nmcs(this->level, model, root_state, rng, this->horizon);
        if (score > best_score) {
            best_score = score;
            best_seq = std::move(seq);
        }
        --remaining;
    }
    delete root_state;

#if NMCS_ENABLE_STATS
    // --------- LOG BBOX DENSITY (EXCLUDING initial CROSSES) FOR FIRST X MOVES ----------
    if (GLOBAL_DENSITY_FILE.is_open()) {
        ABS::Gamestate* replay = model->copyState(state);

        // baseline crosses = crosses at start of this trajectory (step 0)
        std::bitset<J5::SIZE*J5::SIZE> baseline;
        if (auto* js0 = asJoinFiveState(replay)) {
            baseline = js0->crosses;

            // step 0
            {
                auto dd0 = bboxDensityExcludingBaseline(js0, baseline);
                GLOBAL_DENSITY_FILE << GLOBAL_GAME_ID_COUNTER << ";"
                                    << 0 << ";"
                                    << dd0.added_crosses << ";"
                                    << dd0.bbox_area << ";"
                                    << dd0.bbox_density << "\n";
            }

            const int limit = std::min((int)best_seq.size(), DENSITY_FIRST_X_MOVES);
            for (int step = 1; step <= limit; ++step) {
                if (replay->terminal) break;

                // commit chosen move
                model->applyAction(replay, best_seq[step - 1], rng, nullptr);

                auto* js_step = asJoinFiveState(replay);
                if (!js_step) break;

                auto dd = bboxDensityExcludingBaseline(js_step, baseline);
                GLOBAL_DENSITY_FILE << GLOBAL_GAME_ID_COUNTER << ";"
                                    << step << ";"
                                    << dd.added_crosses << ";"
                                    << dd.bbox_area << ";"
                                    << dd.bbox_density << "\n";
            }

            GLOBAL_DENSITY_FILE.flush();
        }

        delete replay;
    }

    GLOBAL_GAME_ID_COUNTER++;
#endif

    return best_seq;
}

