def organize_output(output, new_output):
    '''Update output by taking items from new_output.

        output: dictionary
        new_output: tuple
    '''
    for e_id, e_output in output.items():
        e_output["last"] = False

    for e_output in new_output:
        output[e_id] = e_output
        output[e_id]["last"] = True

def split_state(state, num_elevators, total_floors):
    '''Return list of individual states for each elevator'''
    up_calls = state[0][0 : total_floors]
    down_calls = state[0][total_floors: 2 * total_floors]

    states = []
    for e_id in range(num_elevators):
        s = []
        req_begin = 2 * total_floors
        mult = e_id * 2
        if mult > 0:
            req_begin *= mult
        req_calls = state[0][req_begin: req_begin + total_floors]

        e_begin = req_begin + total_floors
        e_floor = state[0][e_begin: e_begin + total_floors]

        s.extend(up_calls)
        s.extend(down_calls)
        s.extend(req_calls)
        s.extend(e_floor)
        states.append(s)

    return states