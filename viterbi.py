import numpy as np

states = ('1', '2', '3')
start_probability = {'1': 0.32352941, '2': 0.26470588, '3': 0.41176471}

A = {'1': {'1': 0, '2': 0.1, '3': 0.9},
     '2': {'1': 0.6, '2': 0.1, '3': 0.3},
     '3': {'1': 0.4, '2': 0.5, '3': 0.1}
     }

B = {
    '1': {'W': 0.6, 'X': 0.1, 'Y': 0.2, 'Z': 0.1},
    '2': {'W': 0.1, 'X': 0.7, 'Y': 0.1, 'Z': 0.1},
    '3': {'W': 0.2, 'X': 0, 'Y': 0.5, 'Z': 0.3}
}

files = ["input_test.txt" ,"input_1.txt", "input_2.txt", "input_3.txt", "input_4.txt", "input_5.txt"]


def viterbi(states, obs_sequence, start, transition, emission):
    T = len(obs_sequence)
    storage_matrix = [{}]
    path = {}

    # 1. Initialization
    for i in states:
        obs_1 = obs_sequence[0]
        delta = np.log(start[i]) + np.log(emission[i][obs_1])
        storage_matrix[0][i] = delta
        path[i] = [i]

    # 2. Recursion
    for t in range(1, T):
        storage_matrix.append({})
        viterbi_path = {}

        for j in states:
            obs = emission[j][obs_sequence[t]]
            (prob, state) = max((storage_matrix[t - 1][s] + np.log(transition[s][j]) + np.log(obs), s) for s in states)
            storage_matrix[t][j] = prob
            viterbi_path[j] = path[state] + [j]

        path = viterbi_path

    (prob, state) = max((storage_matrix[t][state], state) for state in states)
    return "".join(path[state])


def read_files(file):
    print("Reading file: " + file)
    f = open(file, "r")
    contents = f.read()
    return list(contents)


def write_results(data, index):
    with open("output" + str(index) + ".txt", "w") as new_file:
        results = viterbi(states, data, start_probability, A, B)
        print("Writing results on file...")
        new_file.write(results)

    return results

def run():
    for i in range(len(files)):
        data = read_files(files[i])
        write_results(data, i)
    print("Completed running tests.")


run()
