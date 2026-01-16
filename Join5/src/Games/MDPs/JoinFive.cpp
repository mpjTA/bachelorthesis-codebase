#include "../../../include/Games/MDPs/JoinFive.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <map>

using namespace J5;

inline void cell_index_to_pos(int index, int& row, int& col) {
    row = index / SIZE;
    col = index % SIZE;
}

void decode_action(int action, int& row_start, int &col_start, int& row_end, int& col_end) {
    int encoded_start_pos = action % (SIZE*SIZE);
    cell_index_to_pos(encoded_start_pos, row_start, col_start);
    int encoded_end_pos = action / (SIZE*SIZE);
    cell_index_to_pos(encoded_end_pos, row_end, col_end);
}

inline int encode_action(int row_start, int col_start, int row_end, int col_end) {
    return (row_start*SIZE + col_start) + (row_end*SIZE + col_end)*SIZE*SIZE;
}

inline int pos_to_cell_index(int row, int col) {
    return row*SIZE + col;
}

std::vector<int> Model::obsShape() const {
    return {8,SIZE,SIZE};
}

void Model::getObs(ABS::Gamestate* uncasted_state, int* obs) {
    auto state = dynamic_cast<Gamestate*>(uncasted_state);

    /*
    Layers:
    1. Crosses
    2. Horizontals
    3. Verticals
    4. Diagonals left-tilted
    5. Diagonals right-tilted
    6. Marking of the currently selected endpoint
    7. Layer to mark whether we are in an intermediate step
    8. Layer to mark legal moves
    */

    int bsize = J5::SIZE * J5::SIZE;
    for (int i = 0; i < J5::SIZE; i++) {
        for (int j = 0; j < J5::SIZE; j++) {
            int pos = pos_to_cell_index(i,j);
            obs[pos] = state->crosses.test(pos);
            obs[bsize + pos] = state->horizontals.test(pos);
            obs[2*bsize + pos] = state->verticals.test(pos);
            obs[3*bsize + pos] = state->diagonals_left.test(pos);
            obs[4*bsize + pos] = state->diagonals_right.test(pos);
            obs[5*bsize + pos] = state->first_stone_idx == pos;
            obs[6*bsize + pos] = state->first_stone_idx == -1;
            obs[7*bsize + pos] = 0;
        }
    }

    for (int legal : getActions(state)) {
        if (!decoupled_action_space) {
            int row_start, col_start, row_end, col_end;
            decode_action(legal, row_start, col_start, row_end, col_end);
            int pos1 = pos_to_cell_index(row_start, col_start);
            int pos2 = pos_to_cell_index(row_end, col_end);
            obs[7*bsize + pos1] = 1;
            obs[7*bsize + pos2] = 1;
        }else
            obs[7*bsize + legal] = 1;
    }
}

[[nodiscard]] std::vector<int> Model::actionShape() const {
    if (decoupled_action_space)
        return {SIZE*SIZE};
    else
        return {SIZE*SIZE,SIZE*SIZE};
}

int Model::encodeAction(int* decoded_action) {
    if (decoupled_action_space)
        return decoded_action[0];
    else {
        int row_start, col_start, row_end, col_end;
        cell_index_to_pos(decoded_action[0], row_start, col_start);
        cell_index_to_pos(decoded_action[1], row_end, col_end);
        return encode_action(row_start,col_start,row_end,col_end);
    }
}

inline std::bitset<SIZE*SIZE>* get_line_set(Gamestate* state, int dir_code) {
    switch (dir_code) {
        case 0: return &state->horizontals;
        case 1: return &state->verticals;
        case 2: return &state->diagonals_left;
        case 3: return &state->diagonals_right;
        default: assert(false);
    }
    return nullptr;
}

inline void increment_line_num(Gamestate* state, int dir_code) {
    switch (dir_code) {
        case 0: state->num_horizontals++; break;
        case 1: state->num_verticals++; break;
        case 2: state->num_diagonals_left++; break;
        case 3: state->num_diagonals_right++; break;
        default: assert(false);
    }
}

inline std::unordered_map<int,std::vector<int>>* line_to_actions(Gamestate* state, int dir_code) {
    switch (dir_code) {
        case 0: return &state->horizontals_to_actions;
        case 1: return &state->verticals_to_actions;
        case 2: return &state->diagonals_left_to_actions;
        case 3: return &state->diagonals_right_to_actions;
        default: assert(false);
    }
    return nullptr;
}

inline bool Model::out_of_bounds(int row, int col){
    return row < 0 || row >= SIZE || col < 0 || col >= SIZE;
}

Model::Model(bool joint, bool exit_on_out_of_bounds, bool decoupled_action_space) {
    this->joint = joint;
    this->exit_on_out_of_bounds = exit_on_out_of_bounds;
    this->decoupled_action_space = decoupled_action_space;

    //calculate the 512 entries of the action lookup table
    for (int i = 0; i < 512; i++) {
        std::bitset<9> crosses(i);
        std::vector<std::tuple<int, int, int> > line_actions;
        int hole_pos = -1;
        for (int start = 0; start <= 4; start++) {
            //check for each possible start/end point combination
            int end = start + 4 ;
            int num_holes = 0;
            for (int pos = start; pos <= end; pos ++) {
                if (!crosses[pos]) {
                    hole_pos = pos;
                    num_holes++;
                }
            }
            if (num_holes == 1)
                line_actions.emplace_back(start, end, hole_pos);
        }
        actions_lookup_table.push_back(line_actions);
    }

    //Action codes and index offset
    action_to_dir_code_and_offset = {
        {{1, 0}, {0, -1, 0}}, //horizontal
        {{-1, 0}, {0, 0, 0}},

        {{0, 1}, {1, 0, -1}}, //vertical
        {{0, -1}, {1, 0, 0}},

        {{1, 1}, {2, 0, 0}}, //diagonal left
        {{-1, -1}, {2, -1, -1}},

        {{1, -1}, {3, -1, 0}}, //diagonal right
        {{-1, 1}, {3, 0, -1}},
    };
}

bool Gamestate::operator==(const ABS::Gamestate& other) const{

    //For speedup make a size check first
    if (first_stone_idx != dynamic_cast<const Gamestate*>(&other)->first_stone_idx ||
        num_verticals != dynamic_cast<const Gamestate*>(&other)->num_verticals ||
        num_horizontals != dynamic_cast<const Gamestate*>(&other)->num_horizontals ||
        num_diagonals_left != dynamic_cast<const Gamestate*>(&other)->num_diagonals_left ||
        num_diagonals_right != dynamic_cast<const Gamestate*>(&other)->num_diagonals_right)
        return false;

    //If size test failed, do heavy but exact comparison
    return crosses ==  dynamic_cast<const Gamestate*>(&other)->crosses &&
    horizontals == dynamic_cast<const Gamestate*>(&other)->horizontals &&
    verticals == dynamic_cast<const Gamestate*>(&other)->verticals &&
    diagonals_left == dynamic_cast<const Gamestate*>(&other)->diagonals_left &&
    diagonals_right == dynamic_cast<const Gamestate*>(&other)->diagonals_right;
}

size_t Gamestate::hash() const
{
    size_t crosses_hash = hash_crosses(crosses);
    size_t horizontals_hash = hash_horizontals(horizontals);
    size_t verticals_hash = hash_verticals(verticals);
    size_t diagonals_left_hash = hash_diagonals_left(diagonals_left);
    size_t action_hash = 0;
    for (int action : avail_actions)
        action_hash = (action_hash << 2) ^ action;

    return crosses_hash ^ horizontals_hash ^ verticals_hash ^ diagonals_left_hash ^ action_hash ^ first_stone_idx;
}

void Model::printState(ABS::Gamestate* uncasted_state) {
    auto* state = dynamic_cast<Gamestate*>(uncasted_state);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << (state->crosses.test(i*SIZE + j) ? "X " : ". ");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

ABS::Gamestate* Model::getInitialState(int num) {
        std::pair<int,int> offset = {SIZE/2 - 4, SIZE/2 - 4};
        std::vector<std::pair<int,int>> initial_crosses;

        for (int i = 0; i < 4; i++) {
            //vertical lines
            initial_crosses.emplace_back(3+i, 0);
            initial_crosses.emplace_back(3+i, 9);
            initial_crosses.emplace_back(i, 3);
            initial_crosses.emplace_back(i, 6);
            initial_crosses.emplace_back(6+i, 3);
            initial_crosses.emplace_back(6+i, 6);

            //horizontal lines
            initial_crosses.emplace_back(0, 3+i);
            initial_crosses.emplace_back(9, 3+i);
            initial_crosses.emplace_back(3, i);
            initial_crosses.emplace_back(6, i);
            initial_crosses.emplace_back(3, 6+i);
            initial_crosses.emplace_back(6, 6+i);
        }

        //Add offset to initial crosses
        for (auto& pos : initial_crosses) {
            pos.first += offset.first;
            pos.second += offset.second;
        }

        //Set crosses
        auto state = new Gamestate();

        for (auto& pos : initial_crosses)
            state->crosses.set(pos_to_cell_index(pos.first,pos.second), true);

        //Brute force determine actions
        auto dirs = std::vector<std::pair<int,int>>{{1,0},{0,1},{1,1},{1,-1}};
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {

                std::vector<int> holes = {0,0,0,0};
                auto potential_line_indices = std::vector(dirs.size(), std::vector<int>());
                std::vector<int> hole_positions = {0,0,0,0};

                for (int k = 0; k < 5; k++) {
                    for (size_t h = 0; h<holes.size(); h++) {
                        if (holes[h] > 1)
                            continue;
                        int new_row = i + k*dirs[h].first;
                        int new_col = j + k*dirs[h].second;
                        potential_line_indices[h].push_back(line_segment_to_index(new_row - dirs[h].first, new_col - dirs[h].second, new_row, new_col));
                        if (out_of_bounds(new_row,new_col))
                            holes[h] = 2;
                        else if (!state->crosses.test(pos_to_cell_index(new_row,new_col))) {
                            holes[h]++;
                            hole_positions[h] = pos_to_cell_index(new_row,new_col);
                        }
                    }
                }
                for (size_t h = 0; h<holes.size(); h++) {
                    if (holes[h] == 1) {
                        int end_row = i + 4*dirs[h].first;
                        int end_col = j + 4*dirs[h].second;
                        int action_code = encode_action(i,j,end_row,end_col);
                        state->avail_actions.insert( action_code);
                        state->holes_to_actions[hole_positions[h]].push_back(action_code);
                        auto [dir_code, row_offset, col_offset] = action_to_dir_code_and_offset[dirs[h]];
                        for (int idx : potential_line_indices[h])
                            (*line_to_actions(state,dir_code))[idx].push_back(action_code);
                    }
                }
            }
        }

        return state;
}
/*
ABS::Gamestate* Model::getInitialState(std::mt19937& rng){
    return getInitialState(0);
}*/

ABS::Gamestate* Model::getInitialState(std::mt19937& rng){
            std::pair<int,int> offset = {SIZE/2 - 4, SIZE/2 - 4};
        std::vector<std::pair<int,int>> initial_crosses;

        for (int i = 0; i < 4; i++) {
            //vertical lines
            initial_crosses.emplace_back(3+i, 0);
            initial_crosses.emplace_back(3+i, 9);
            initial_crosses.emplace_back(i, 3);
            initial_crosses.emplace_back(i, 6);
            initial_crosses.emplace_back(6+i, 3);
            initial_crosses.emplace_back(6+i, 6);

            //horizontal lines
            initial_crosses.emplace_back(0, 3+i);
            initial_crosses.emplace_back(9, 3+i);
            initial_crosses.emplace_back(3, i);
            initial_crosses.emplace_back(6, i);
            initial_crosses.emplace_back(3, 6+i);
            initial_crosses.emplace_back(6, 6+i);
        }

        //Add offset to initial crosses
        for (auto& pos : initial_crosses) {
            pos.first += offset.first;
            pos.second += offset.second;
        }

        //Set crosses
        auto state = new Gamestate();

        for (auto& pos : initial_crosses)
            state->crosses.set(pos_to_cell_index(pos.first,pos.second), true);

        if (init_avail_actions.empty()){
            //Brute force determine actions
            auto dirs = std::vector<std::pair<int,int>>{{1,0},{0,1},{1,1},{1,-1}};
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {

                    std::vector<int> holes = {0,0,0,0};
                    auto potential_line_indices = std::vector(dirs.size(), std::vector<int>());
                    std::vector<int> hole_positions = {0,0,0,0};

                    for (int k = 0; k < 5; k++) {
                        for (size_t h = 0; h<holes.size(); h++) {
                            if (holes[h] > 1)
                                continue;
                            int new_row = i + k*dirs[h].first;
                            int new_col = j + k*dirs[h].second;
                            potential_line_indices[h].push_back(line_segment_to_index(new_row - dirs[h].first, new_col - dirs[h].second, new_row, new_col));
                            if (out_of_bounds(new_row,new_col))
                                holes[h] = 2;
                            else if (!state->crosses.test(pos_to_cell_index(new_row,new_col))) {
                                holes[h]++;
                                hole_positions[h] = pos_to_cell_index(new_row,new_col);
                            }
                        }
                    }
                    for (size_t h = 0; h<holes.size(); h++) {
                        if (holes[h] == 1) {
                            int end_row = i + 4*dirs[h].first;
                            int end_col = j + 4*dirs[h].second;
                            int action_code = encode_action(i,j,end_row,end_col);
                            state->avail_actions.insert( action_code);
                            state->holes_to_actions[hole_positions[h]].push_back(action_code);
                            auto [dir_code, row_offset, col_offset] = action_to_dir_code_and_offset[dirs[h]];
                            for (int idx : potential_line_indices[h])
                                (*line_to_actions(state,dir_code))[idx].push_back(action_code);
                        }
                    }
                }
            }
            init_avail_actions = state->avail_actions;
            init_holes_to_actions = state->holes_to_actions;
            init_verticals_to_actions = state->verticals_to_actions;
            init_horizontals_to_actions = state->horizontals_to_actions;
            init_diagonals_left_to_actions = state->diagonals_left_to_actions;
            init_diagonals_right_to_actions = state->diagonals_right_to_actions;
        }else{
            state->avail_actions = init_avail_actions;
            state->holes_to_actions = init_holes_to_actions;
            state->verticals_to_actions = init_verticals_to_actions;
            state->horizontals_to_actions = init_horizontals_to_actions;
            state->diagonals_left_to_actions = init_diagonals_left_to_actions;
            state->diagonals_right_to_actions = init_diagonals_right_to_actions;
        }

        return state;
}

int Model::getNumPlayers() {
    return 1;
}

ABS::Gamestate* Model::copyState(ABS::Gamestate* uncasted_state) {
    auto state = dynamic_cast<Gamestate*>(uncasted_state);
    auto new_state = new Gamestate();
    *new_state = *state; //default copy constructor should work
    return new_state;
}

inline int sign(int x) {
    return (x > 0) - (x < 0);
}

std::vector<int> Model::getActions_(ABS::Gamestate* uncasted_state)  {

    if (decoupled_action_space){
        Gamestate* state = dynamic_cast<Gamestate*>(uncasted_state);
        if (state->first_stone_idx == -1){
            std::set<int> first_poss = {};
            for (int a : state->avail_actions){
                int r1,c1,r2,c2;
                decode_action(a,r1,c1,r2,c2);
                first_poss.insert(pos_to_cell_index(r1,c1));
                first_poss.insert(pos_to_cell_index(r2,c2));
            }
            std::vector<int> first_poss_vec = {first_poss.begin(), first_poss.end()};
            return first_poss_vec;
        }else{
            std::vector<int> actions;
            for (int a : state->avail_actions){
                int r1,c1,r2,c2;
                decode_action(a,r1,c1,r2,c2);
                if (state->first_stone_idx == pos_to_cell_index(r1,c1))
                    actions.push_back(pos_to_cell_index(r2,c2));
                if (state->first_stone_idx == pos_to_cell_index(r2,c2))
                    actions.push_back(pos_to_cell_index(r1,c1));
            }
            return actions;
        }
    }
    else
        return std::vector<int>(dynamic_cast<Gamestate*>(uncasted_state)->avail_actions.begin(), dynamic_cast<Gamestate*>(uncasted_state)->avail_actions.end());

    /*
     * Can be used for debug to compare if the smart action calculation yields the same results as the trivial brute force variant
     */
    // std::set<int> actions = {};
    // auto state = dynamic_cast<Gamestate*>(uncasted_state);
    // auto dirs = std::vector<std::pair<int,int>>{{1,0},{0,1},{1,1},{1,-1}};
    // for (int i = 0; i < SIZE; i++) {
    //     for (int j = 0; j < SIZE; j++) {
    //
    //         std::vector<int> holes = {0,0,0,0};
    //
    //         for (int k = 0; k < 5; k++) {
    //             for (size_t h = 0; h<holes.size(); h++) {
    //                 if (holes[h] > 1)
    //                     continue;
    //                 int new_row = i + k*dirs[h].first;
    //                 int new_col = j + k*dirs[h].second;
    //                 if (out_of_bounds(new_row,new_col))
    //                     holes[h] = 2;
    //                 else if (!state->crosses.test(pos_to_cell_index(new_row,new_col))) {
    //                     holes[h]++;
    //                 }
    //             }
    //         }
    //         for (size_t h = 0; h<holes.size(); h++) {
    //             if (holes[h] == 1) {
    //                 int end_row = i + 4*dirs[h].first;
    //                 int end_col = j + 4*dirs[h].second;
    //
    //                 std::set<std::pair<int,int>> used_tiles;
    //                 for (int k = 0; k < 5; k++) {
    //                     int new_row = i + k*dirs[h].first;
    //                     int new_col = j + k*dirs[h].second;
    //                     used_tiles.insert({new_row,new_col});
    //                 }
    //                 //test if any lines are retraced
    //                 bool intersection = false;
    //                 for (auto line : state->drawn_lines)
    //                 {
    //                     std::pair<int,int> dir = {sign(line.second.first - line.first.first), sign(line.second.second - line.first.second)};
    //                     if (dir != dirs[h])
    //                         continue;
    //                     std::set<std::pair<int,int>> cmp_used_tiles = {};
    //                     int cmp_start_row = line.first.first;
    //                     int cmp_start_col = line.first.second;
    //                     int cmp_end_row = line.second.first;
    //                     int cmp_end_col = line.second.second;
    //                     //test if lines intersect
    //                     for (int k = 0; k < 5; k++) {
    //                         int new_row = cmp_start_row + k*dir.first;
    //                         int new_col = cmp_start_col + k*dir.second;
    //                         cmp_used_tiles.insert({new_row,new_col});
    //                     }
    //
    //                     std::vector<int> intersection_result;
    //                     for (auto& tile : used_tiles) {
    //                         if (cmp_used_tiles.find(tile) != cmp_used_tiles.end() && ((tile != std::pair{cmp_start_row,cmp_start_col} && tile != std::pair{cmp_end_row,cmp_end_col}) || (tile != std::pair{i,j} && tile != std::pair{end_row,end_col})))
    //                             intersection_result.push_back(1);
    //                     }
    //                     if (!intersection_result.empty()){
    //                         intersection = true;
    //                         break;
    //                     }
    //
    //                 }
    //
    //                 if (!intersection){
    //                     int action_code = encode_action(i,j,end_row,end_col);
    //                     actions.insert(action_code);
    //                 }
    //             }
    //         }
    //     }
    // }
    //
    // std::vector<int> a_vec = {actions.begin(), actions.end()};
    //
    // auto b_vec = std::vector<int>(dynamic_cast<Gamestate*>(uncasted_state)->avail_actions.begin(), dynamic_cast<Gamestate*>(uncasted_state)->avail_actions.end());
    //
    // if (a_vec != b_vec)
    // {
    //     printState(state);
    //     std::cout << "A: ";
    //     for (int a : a_vec)
    //         std::cout << a << " ";
    //     std::cout << std::endl;
    //     std::cout << "B: ";
    //     for (int a : b_vec)
    //         std::cout << a << " ";
    //     std::cout << std::endl;
    //     int a,b,c,d;
    //     decode_action(117012,a,b,c,d);
    //     std::cout << a << " " << b << " " << c << " " << d << std::endl;
    //     exit(1);
    // }
    //
    // return a_vec;
}

inline int Model::line_segment_to_index(int row_from, int col_from, int row_to, int col_to) {
    std::pair<int,int> dir = {sign(row_to - row_from), sign(col_to - col_from)};
    auto [dir_code, row_offset, col_offset] = action_to_dir_code_and_offset[dir];
    return (row_to + row_offset)* SIZE + (col_to + col_offset);
}

void decode_cell(int cell, int &row, int &col) {
    row = cell / SIZE;
    col = cell % SIZE;
}

std::pair<std::vector<double>,double> Model::applyAction_(ABS::Gamestate* uncasted_state, int action, std::mt19937& rng, std::vector<std::pair<int,int>>* decision_outcomes) {
    auto* state = dynamic_cast<Gamestate*>(uncasted_state);

    if (decoupled_action_space && state->first_stone_idx == -1){
        state->first_stone_idx = action;
        return {{0}, 1};
    }

    int row_end, col_end, col_start, row_start;
    if (decoupled_action_space){
        decode_cell(state->first_stone_idx, row_start, col_start);
        decode_cell(action, row_end, col_end);
        if(!state->avail_actions.contains(encode_action(row_start,col_start,row_end,col_end))){
            int tmp = row_start;
            row_start = row_end;
            row_end = tmp;
            tmp = col_start;
            col_start = col_end;
            col_end = tmp;
        }
        state->first_stone_idx = -1;
    }else
        decode_action(action, row_start, col_start, row_end, col_end);

    //FOR DEBUG
    //state->drawn_lines.push_back({{row_start,col_start},{row_end,col_end}});

    //update state
    int empty_row = -1;
    int empty_col = -1;
    std::pair<int,int> action_dir = std::make_pair(sign(row_end - row_start), sign(col_end - col_start));
    auto [dir_code, row_offset, col_offset] = action_to_dir_code_and_offset[action_dir];
    for (int i = 0; i < (joint? 5 : 6); i++) {
        int new_row = row_start + i*action_dir.first;
        int new_col = col_start + i*action_dir.second;

        //set crosses
        int cell_idx = pos_to_cell_index(new_row,new_col);
        if (i != 5 && !state->crosses.test(cell_idx)) {
            //std::cout << empty_row << " ___ " << i << std::endl;
            assert (empty_row == -1 && empty_col == -1);
            empty_row = new_row;
            empty_col = new_col;
            for (int a : state->holes_to_actions.at(cell_idx))
                state->avail_actions.erase(a);
            state->holes_to_actions.erase(cell_idx);
            state->crosses.set(cell_idx, true);
        }

        //update tiles

        if ( joint && (i == 0))
            continue;

        int idx = line_segment_to_index(new_row - action_dir.first, new_col - action_dir.second, new_row, new_col);
        if (i != 0 && i != 5) { //in this case, we only want to remove neighboring lines from the action set
            increment_line_num(state, dir_code);
            get_line_set(state, dir_code)->set(idx, true);
        }
        auto l_to_actions = line_to_actions(state, dir_code);
        if ( (i != 0 && i != 5) || l_to_actions->contains(idx)){
            for (int a : l_to_actions->at(idx))
                state->avail_actions.erase(a);
            l_to_actions->erase(idx);
        }
    }

    assert (empty_row != -1 && empty_col != -1);

    //Add new actions
    auto dirs = std::vector<std::pair<int,int>>{{1,0},{0,1},{1,1},{1,-1}};
    for (auto& dir : dirs) {

        std::bitset<9> crosses;
        crosses.set(4, true);
        for (int i = 1; i < 5; i++) {
            int new_row_pos = empty_row + i*dir.first;
            int new_col_pos = empty_col + i*dir.second;
            crosses.set(4+i,out_of_bounds(new_row_pos,new_col_pos)? false : state->crosses.test(pos_to_cell_index(new_row_pos,new_col_pos)));

            int new_row_neg = empty_row - i*dir.first;
            int new_col_neg = empty_col - i*dir.second;
            crosses.set(4-i,out_of_bounds(new_row_neg,new_col_neg)? false : state->crosses.test(pos_to_cell_index(new_row_neg,new_col_neg)));
        }

        auto line_actions = actions_lookup_table[crosses.to_ullong()];

        auto [dir_code, row_offset, col_offset] = action_to_dir_code_and_offset[dir]; //precomputed to satisfy the 1-hole condition

        //Filter for legal actions by testing if they retrace any line portion
        for (auto& line_action : line_actions) {

            //translation from 1D line space to 2D grid space
            auto [start,end, hole_pos] = line_action;
            assert (hole_pos >= start && hole_pos <= end);
            int start_row = empty_row + (start-4) * dir.first;
            int start_col = empty_col + (start-4) * dir.second;
            int end_row = empty_row + (end-4) * dir.first;
            int end_col = empty_col + (end-4) * dir.second;
            int hole_row = empty_row + (hole_pos-4) * dir.first;
            int hole_col = empty_col + (hole_pos-4) * dir.second;

            bool oob_start = out_of_bounds(start_row,start_col);
            bool oob_end = out_of_bounds(end_row,end_col);
            if ( (oob_start || oob_end) && exit_on_out_of_bounds){
                printState(state);
                throw std::runtime_error("Encountered an action that is legal on an infinite board but not legal in this finite setting.");
            }
            if (oob_start || oob_end)
                continue;

            //Retrace check
            int dir_row = sign(end_row - start_row);
            int dir_col = sign(end_col - start_col);
            bool valid_action = true;
            std::vector<int> line_indices = {};
            for (int i = (joint? 1 : 0); i < (joint? 5 : 6); i++) {
                int row_it = start_row + i*dir_row;
                int col_it = start_col + i*dir_col;
                if ((i == 0 || i == 5) && (out_of_bounds(row_it,col_it) || out_of_bounds(row_it - dir_row,col_it - dir_col)))
                    continue;
                int idx = line_segment_to_index(row_it - dir_row, col_it - dir_col, row_it, col_it);
                if (i!= 0 && i != 5)
                    line_indices.push_back(idx);
                if (get_line_set(state, dir_code)->test(idx)) {
                    valid_action = false;
                    break;
                }
            }

            if (valid_action) {
                int action_code = encode_action(start_row, start_col, end_row, end_col);
                state->avail_actions.insert( action_code);
                state->holes_to_actions[pos_to_cell_index(hole_row,hole_col)].push_back(action_code);
                for (int idx : line_indices)
                    (*line_to_actions(state, dir_code))[idx].push_back(action_code);
            }
        }
    }

    //Terminal check
    if (state->avail_actions.empty())
        state->terminal = true;

    return {{1.0}, 1.0};
}

//------------------------ For heuristic -------------------------------------

// 0: H- (horizontal left -*)
// 1: H+ (horizontal right *-)
// 2: V- (vertical up |)
// 3: V+ (vertical down |)
// 4: DL- (diagonal left up \*)
// 5: DL+ (diagonal right down *\)
// 6: DR- (diagonal right up */)
// 7: DR+ (diagonal left down /*)
enum HalfDir : int{
    H_NEG=0, H_POS=1, V_NEG=2, V_POS=3, DL_NEG=4, DL_POS=5, DR_NEG=6, DR_POS=7
};

inline int half_index_from_dir(int dir_code, int sign){
    switch(dir_code){
        case 0: return (sign<0)? H_NEG : H_POS;   // horizontal
        case 1: return (sign<0)? V_NEG : V_POS;   // vertical
        case 2: return (sign<0)? DL_NEG: DL_POS;  // diag
        case 3: return (sign<0)? DR_NEG: DR_POS;  // diag /
        default: return 0;
    }
}

// Counts deg of a cross
std::array<int,8> Model::halfDegreeAt(const Gamestate* s, int row, int col) {
    std::array<int,8> deg{}; // init 0

    static const std::pair<int,int> dirs[4] = {{1,0},{0,1},{1,1},{1,-1}};

    for(int dcode=0; dcode<4; ++dcode){
        auto d = dirs[dcode];
        int dr = d.first, dc = d.second;

        // (row-dr,col-dc) -> (row,col)
        {
            int r1 = row - dr, c1 = col - dc;
            int r2 = row, c2 = col;
            if(!out_of_bounds(r1,c1) && !out_of_bounds(r2,c2)){
                int idx = line_segment_to_index(r1,c1,r2,c2);
                if(get_line_set(const_cast<Gamestate*>(s), dcode)->test(idx)){
                    deg[half_index_from_dir(dcode, -1)] += 1;
                }
            }
        }

        //(row,col) -> (row+dr,col+dc)
        {
            int r1 = row, c1 = col;
            int r2 = row + dr,  c2 = col + dc;
            if(!out_of_bounds(r1,c1) && !out_of_bounds(r2,c2)){
                int idx = line_segment_to_index(r1,c1,r2,c2);
                if(get_line_set(const_cast<Gamestate*>(s), dcode)->test(idx)){
                    deg[half_index_from_dir(dcode, +1)] += 1;
                }
            }
        }
    }
    /* // DEBUG
    std::cout << "[";
    for (int i = 0; i < 8; ++i) {
        std::cout << deg[i] << (i+1<8 ? "," : "");
    }
    std::cout << "] sum=" << std::accumulate(deg.begin(), deg.end(), 0) << std::endl;
    */
    return deg;
}



std::array<std::array<int,8>,5> Model::halfDegreeVectorForActionAfter(const Gamestate* s, int action){
    int rs, cs, re, ce; // rowstart etc.
    decode_action(action, rs, cs, re, ce);

    int dr = sign(re - rs); // determines direction in row (see HalfDir)
    int dc = sign(ce - cs);
    assert(!(dr==0 && dc==0));

    auto it = action_to_dir_code_and_offset.find(std::make_pair(dr,dc));
    assert(it != action_to_dir_code_and_offset.end());
    int dir_code = std::get<0>(it->second); // 0: horizontal, 1: vertical, 2: diagonal (both)
    int hpos = half_index_from_dir(dir_code, +1); // half degree pos direction (see HalfDir)
    int hneg = half_index_from_dir(dir_code, -1); // half degree nef direction (see HalfDir)
    // 5 crosses
    int r[5], c[5];
    for(int k=0; k<5; ++k){
        r[k] = rs + k*dr; // row pos of cross k (or hole)
        c[k] = cs + k*dc; // col pos of cross k (or hole)
        assert(!out_of_bounds(r[k], c[k]));
    }

    // Baseline half degree (haf degree without new line drawn)
    std::array<std::array<int,8>,5> degs{};
    for(int k=0; k<5; ++k){
        degs[k] = halfDegreeAt(s, r[k], c[k]);
    }
    // New (virtual) segment of 5 crosses (with new line drawn)
    // segments: (0-1), (1-2), (2-3), (3-4)
    for(int k=0; k<4; ++k){
        degs[k][hpos] += 1; // k to k+1 (positive half degree)
        degs[k+1][hneg] += 1; // von k+1 to k (negative half degree)
    }

    /* // DEBUG PRINT
    std::cout << "=== halfDegreeVectorForActionAfter ===\n";
    std::cout << "Action (" << rs << "," << cs << ") -> (" << re << "," << ce << ")\n";
    for(int k=0; k<5; ++k){
        int sum = std::accumulate(degs[k].begin(), degs[k].end(), 0);
        std::cout << " Cell " << k << " @(" << r[k] << "," << c[k] << "): [";
        for (int h=0; h<8; ++h) {
            std::cout << degs[k][h] << (h+1<8 ? "," : "");
        }
        std::cout << "] sum=" << sum << "\n";
    }
    std::cout << "=====================================" << std::endl;
    */
    return degs;
}
// Name potentialScore is a bit misleading, sum of deg, real potential heuristic score is calculated in NestedMctsAgent.cpp
int Model::potentialScore(ABS::Gamestate* s, int action){
    auto* js = dynamic_cast<J5::Gamestate*>(s);
    auto degs = halfDegreeVectorForActionAfter(js, action);
    int sum = 0;
    for (auto& cell : degs) for (int h : cell) sum += h;
    //std::cout << action << " potentialScore: "<< sum << std::endl;
    return sum;
}

// check if two crosses are connected
inline bool Model::edgeTest(ABS::Gamestate *s, int row1, int col1, int row2, int col2, int dcode) {
    auto state = dynamic_cast<Gamestate*>(s);
    int idx = line_segment_to_index(row1,col1,row2,col2);
    return get_line_set(state, dcode)->test(idx);
}
// calculates window for cross candidate (in thesis refered as CrossRay)
int Model::crossWindow(ABS::Gamestate *s, int action) {
    auto state = dynamic_cast<Gamestate*>(s);
    assert(state != nullptr);

    int rs, cs, re, ce;
    decode_action(action, rs, cs, re, ce);
    int dr = sign(re - rs);
    int dc = sign(ce - cs);
    assert(!(dr==0 && dc==0));

    // get dir_code: 0: horizontal, 1: vertical, 2: diagonal (left up), 3: diagonal (right up)
    auto it = action_to_dir_code_and_offset.find({dr, dc});
    assert(it != action_to_dir_code_and_offset.end());
    int dir_code = std::get<0>(it->second);

    // find hole in segment of 5
    int row_hole = -1;
    int col_hole = -1;
    for(int k=0; k<5; ++k) {
        int r = rs + k*dr;
        int c = cs + k*dc;
        assert(!out_of_bounds(r, c));
        if (!state->crosses.test(pos_to_cell_index(r, c))){
            row_hole = r;
            col_hole = c;
            break;
        }
    }
    assert(row_hole != -1 && col_hole != -1);

    const std::pair<int,int> dirs[4] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    bool skip_pos = false; // (dr,dc)
    bool skip_neg = false; // (-dr,-dc)

    /// skip the direction from hole to its future line segment, i. e.:
    /// x o x x x  "skip horizontal in negative (left side) and positive (right side) (half) directions"
    // neighbour in positive direction?
    int rpos = row_hole + dr, cpos = col_hole + dc;
    if (!out_of_bounds(rpos, cpos) && state->crosses.test(pos_to_cell_index(rpos, cpos))) {
        skip_pos = true;
    }
    // neighbour in negative direction?
    int rneg = row_hole - dr, cneg = col_hole - dc;
    if (!out_of_bounds(rneg, cneg) && state->crosses.test(pos_to_cell_index(rneg, cneg))) {
        skip_neg = true;
    }

    int result = 0;

    for (int dcode = 0; dcode < 4; ++dcode) {
        for (const int sgn : {-1, +1}) {
            // vector of half directions
            int vr = sgn * dirs[dcode].first;
            int vc = sgn * dirs[dcode].second;

            // moment of skipping (mentioned above)
            if (dcode == dir_code) {
                if (sgn == +1 && skip_pos) continue; // hole -> (dr,dc)
                if (sgn == -1 && skip_neg) continue; // hole -> (-dr,-dc)
            }

            // look in all directions (4 steps look)
            for (int step = 1; step <= 4; ++step) {
                int r = row_hole + step * vr;
                int c = col_hole + step * vc;
                if (out_of_bounds(r, c)) break;

                if (state->crosses.test(pos_to_cell_index(r, c))) {
                    ++result; // count cross

                    // if found cross has already a connection with another cross in same direction -> break
                    int r2 = r + vr, c2 = c + vc;
                    if (!out_of_bounds(r2, c2)) {
                        if (edgeTest(state, r, c, r2, c2, dcode)) {
                            break;
                        }
                    }
                }else {
                    break; // break after first hole
                }
            }
        }
    }


    return result;
}

inline double centerProximity(int row, int col) {
    double center_row = (SIZE/2) + 0.5;
    double center_col = (SIZE/2) + 0.5;

    double dr = row - center_row;
    double dc = col - center_col;

    double manhattan_distance= std::abs(dr) + std::abs(dc);

    return double{1}/(manhattan_distance);
}

// returns score for touching
double Model::touch(ABS::Gamestate *s, int action){
    auto* state = dynamic_cast<Gamestate*>(s);
    assert(state != nullptr);

    int rs, cs, re, ce;
    decode_action(action, rs, cs, re, ce);
    int dr = sign(re - rs);
    int dc = sign(ce - cs);
    assert(!(dr==0 && dc==0));

    auto it = action_to_dir_code_and_offset.find({dr, dc});
    assert(it != action_to_dir_code_and_offset.end());
    int dir_code = std::get<0>(it->second);

    double best = 0.0;

    // Touch at START: (rs-dr,cs-dc) -> (rs,cs)
    {
        int r1 = rs - dr, c1 = cs - dc;
        int r2 = rs, c2 = cs;
        if (!out_of_bounds(r1,c1) && !out_of_bounds(r2,c2)){
            if (edgeTest(state, r1, c1, r2, c2, dir_code)){
                best = std::max(best, 10.0 * centerProximity(r2,c2)); // or (r1,c1)
            }
        }
    }

    // Touch at END: (re,ce) -> (re+dr,ce+dc)
    {
        int r1 = re, c1 = ce;
        int r2 = re + dr, c2 = ce + dc;
        if (!out_of_bounds(r1,c1) && !out_of_bounds(r2,c2)){
            if (edgeTest(state, r1, c1, r2, c2, dir_code)){
                best = std::max(best, 10.0 * centerProximity(r1,c1)); // or (r2,c2)
            }
        }
    }

    return best;
}

// Manhattan Distance of previous move and potential next move.
double Model::mDistance(ABS::Gamestate *s, int prev_action, int fut_action) {
    const auto state = dynamic_cast<Gamestate*>(s);
    int prev_rs, prev_cs, prev_re, prev_ce;
    int fut_rs, fut_cs, fut_re, fut_ce;

    decode_action(prev_action, prev_rs, prev_cs, prev_re, prev_ce);
    const int prev_dr = sign(prev_re - prev_rs);
    const int prev_dc = sign(prev_ce - prev_cs);

    decode_action(fut_action, fut_rs, fut_cs, fut_re, fut_ce);
    const int fut_dr = sign(fut_re - fut_rs);
    const int fut_dc = sign(fut_ce - fut_cs);

    // find hole in segment of 5 (previous action)
    int prev_row_hole = -1;
    int prev_col_hole = -1;
    for(int k=0; k<5; ++k) {
        int pr = prev_rs + k*prev_dr;
        int pc = prev_cs + k*prev_dc;
        //std::cout << k << " " << pr << " " << pc << std::endl;
        assert(!out_of_bounds(pr, pc));
        if (!state->crosses.test(pos_to_cell_index(pr, pc))){
            prev_row_hole = pr;
            prev_col_hole = pc;
            break;
        }
    }
    assert(prev_row_hole != -1 && prev_col_hole != -1);

    // find hole in segment of 5 (potential next action)
    int fut_row_hole = -1;
    int fut_col_hole = -1;
    for(int k=0; k<5; ++k) {
        int fr = fut_rs + k*fut_dr;
        int fc = fut_cs + k*fut_dc;
        assert(!out_of_bounds(fr, fc));
        if (!state->crosses.test(pos_to_cell_index(fr, fc))){
            fut_row_hole = fr;
            fut_col_hole = fc;
            break;
        }
    }
    assert(fut_row_hole != -1 && fut_col_hole != -1);

    // calculate manhattan distance between hole of prev_action and hole of fut_action
    const double manhattan_distance = std::abs(prev_row_hole - fut_row_hole) + std::abs(prev_col_hole - fut_col_hole);

    return manhattan_distance;
}
/*
std::vector<int> Model::newCreatedActions(ABS::Gamestate *s, int row_hole, int col_hole , int dr, int dc, int dir_code) {
    const auto state = dynamic_cast<Gamestate*>(s);
    assert(state != nullptr);
    const std::pair<int,int> dirs[4] = {{1,0}, {0,1}, {1,1}, {1,-1}};
    bool skip_pos = false; // (dr,dc)
    bool skip_neg = false; // (-dr,-dc)

    /// skip the direction from hole to its future line segment, i. e.:
    /// x o x x x  "skip horizontal in negative (left side) and positive (right side) (half) directions"
    // neighbour in positive direction?
    int rpos = row_hole + dr, cpos = col_hole + dc;
    if (!out_of_bounds(rpos, cpos) && state->crosses.test(pos_to_cell_index(rpos, cpos))) {
        skip_pos = true;
    }
    // neighbour in negative direction?
    int rneg = row_hole - dr, cneg = col_hole - dc;
    if (!out_of_bounds(rneg, cneg) && state->crosses.test(pos_to_cell_index(rneg, cneg))) {
        skip_neg = true;
    }

    std::vector<int> newActions;

    for (int dcode = 0; dcode < 4; ++dcode) {
        for (const int sgn : {-1, +1}) {
            // vector of half directions
            int vr = sgn * dirs[dcode].first;
            int vc = sgn * dirs[dcode].second;

            // moment of skipping (mentioned above)
            if (dcode == dir_code) {
                if (sgn == +1 && skip_pos) continue; // hole -> (dr,dc)
                if (sgn == -1 && skip_neg) continue; // hole -> (-dr,-dc)
            }

            // look in all directions (4 steps look)
            for (int step = 1; step <= 4; ++step) {
                int r = row_hole + step * vr;
                int c = col_hole + step * vc;
                if (out_of_bounds(r, c)) break;

                if (state->crosses.test(pos_to_cell_index(r, c))) {
                    ++result; // count cross

                    // if found cross has already a connection with another cross in same direction -> break
                    int r2 = r + vr, c2 = c + vc;
                    if (!out_of_bounds(r2, c2)) {
                        if (edgeTest(state, r, c, r2, c2, dcode)) {
                            break;
                        }
                    }
                }else {
                    break; // break after first hole
                }
            }
        }
    }

    return 0;
}
*/
Model::MobilityDelta Model::mobilityIf(ABS::Gamestate* s, int action) {
    auto* state = dynamic_cast<Gamestate*>(s);
    assert(state != nullptr);
    int rs,cs,re,ce;
    decode_action(action, rs, cs, re, ce);
    const int dr = sign(re - rs);
    const int dc = sign(ce - cs);

    // find hole in 5 segment
    int hole_r=-1, hole_c=-1, hole_idx=-1;
    for(int k=0; k<5; ++k) {
        int r = rs + k*dr; // move in row direction
        int c = cs + k*dc; // move in col direction
        if (out_of_bounds(r,c)) return {};
        int idx = pos_to_cell_index(r,c);
        if (!state->crosses.test(idx)) { // check if cross is set -> if not: hole
            hole_r = r;
            hole_c = c;
            hole_idx= idx ;
            break;
        }
    }
    if (hole_idx==-1) return {}; // no hole

    // simulating move, collect idx
    auto dir_pair = std::make_pair(dr,dc);
    auto [dir_code, row_off, col_off] = action_to_dir_code_and_offset.at(dir_pair);
    std::vector<int> would_set_indices;
    for (int i=(joint?1:0); i<(joint?5:6); ++i){ // similar to applyAction: x-x-x-x-x 4 edges blocked in joint version and -x-x-x-x-x- 6 edges blocked in disjoint
        int r = rs + i*dr, c = cs + i*dc;
        if ((i == 0||i == 5) && (out_of_bounds(r,c) || out_of_bounds(r-dr,c-dc))) continue;
        int idx = line_segment_to_index(r-dr, c-dc, r, c);
        if (i != 0 && i != 5) would_set_indices.push_back(idx);
    }

    MobilityDelta mobility; // struct with all information to mobility

    // set of removed moves/actions to avoid duplicates
    std::unordered_set<int> rmv;

    // "remove" actions at hole
    if (const auto it = state->holes_to_actions.find(hole_idx); it != state->holes_to_actions.end()){
        rmv.insert(it->second.begin(), it->second.end());
    }

    // "remove" edges that overlap with simulated move/action in same direction
    if (!would_set_indices.empty()) {
        const auto line_map = line_to_actions(const_cast<Gamestate*>(state), dir_code); // <idx, vector of actions in dir_code direction >
        for (const int& seg : would_set_indices) {
            if (auto it2 = line_map->find(seg); it2 != line_map->end()){
                rmv.insert(it2->second.begin(), it2->second.end());
            }
        }
    }

    // "remove" simulated action (current action)
    rmv.insert(action);

    // count all actions from rmv which are available/possible
    for (const int& a : rmv) {
        if (state->avail_actions.count(a)){
            ++mobility.removed;
            mobility.removed_actions.push_back(a);
        }
    }

    // determine all new created actions/moves (X_X) (like applyAction)
    auto dirs = std::vector<std::pair<int,int>>{{1,0},{0,1},{1,1},{1,-1}};

    for (const auto& dir : dirs) {
        std::bitset<9> crosses;
        crosses.set(4, true);
        for (int i=1;i<5;++i){
            int rpos = hole_r + i*dir.first,  cpos = hole_c + i*dir.second;
            crosses.set(4+i, out_of_bounds(rpos,cpos) ? false : state->crosses.test(pos_to_cell_index(rpos,cpos)));
            int rneg = hole_r - i*dir.first,  cneg = hole_c - i*dir.second;
            crosses.set(4-i, out_of_bounds(rneg,cneg) ? false : state->crosses.test(pos_to_cell_index(rneg,cneg)));
        }
        const auto& cand = actions_lookup_table[crosses.to_ullong()];
        auto [dc_code, ro, co] = action_to_dir_code_and_offset.at(dir);

        for (auto [start,end,hpos] : cand) {
            int sr = hole_r + (start - 4)*dir.first;
            int sc = hole_c + (start - 4)*dir.second;
            int er = hole_r + (end - 4)*dir.first;
            int ec = hole_c + (end - 4)*dir.second;
            if (out_of_bounds(sr,sc) || out_of_bounds(er,ec)) continue;

            // retrace check
            bool oob_start = out_of_bounds(sr,sc);
            bool oob_end = out_of_bounds(er,ec);
            if ( (oob_start || oob_end) && exit_on_out_of_bounds){
                printState(s);
                throw std::runtime_error("Encountered an action that is legal on an infinite board but not legal in this finite setting.");
            }
            if (oob_start || oob_end)
                continue;

            //Retrace check
            bool valid = true;
            for (int i=(joint?1:0); i<(joint?5:6); ++i){
                int rr = sr + i*dir.first;
                int cc = sc + i*dir.second;
                // check outside edges left and right (not inner edges of 5 cross segment)
                if (( i== 0||i == 5 ) && (out_of_bounds(rr,cc) || out_of_bounds(rr - dir.first,cc - dir.second))) continue;

                int seg = line_segment_to_index(rr - dir.first, cc - dir.second, rr, cc);

                // check inner edges
                bool segment_taken = get_line_set(const_cast<Gamestate*>(state), dc_code)->test(seg);
                if (!segment_taken && dc_code == dir_code){
                    if (std::find(would_set_indices.begin(), would_set_indices.end(), seg) != would_set_indices.end())
                        segment_taken = true;
                }
                if (segment_taken){ valid = false; break; }
            }
            if (!valid) continue;


            int cand_code = encode_action(sr, sc, er, ec);
            if (!state->avail_actions.count(cand_code)) {
                ++mobility.added;
                mobility.added_actions.push_back(cand_code);
            }
        }
    }

    return mobility;
}


// end for heuristic
