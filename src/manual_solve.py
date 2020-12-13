#!/usr/bin/python

import os, sys
import json
import numpy as np
import re


def solve_9172f3a0(x):
    """
    This puzzle provides a 3x3 array of numbers. The goal is to expand the 3x3 into a 9x9 where each number in the 3x3
    is now represented by a 3x3 in the 9x9.

    [[ 1 ] [ 2 ] [ 3 ]
     [ 4 ] [ 5 ] [ 6 ]
     [ 7 ] [ 8 ] [ 9 ]]

    Becomes....

    [[ 1 ] [ 1 ] [ 1 ] [ 2 ] [ 2 ] [ 2 ] [ 3 ] [ 3 ] [ 3 ]
     [ 1 ] [ 1 ] [ 1 ] [ 2 ] [ 2 ] [ 2 ] [ 3 ] [ 3 ] [ 3 ]
     [ 1 ] [ 1 ] [ 1 ] [ 2 ] [ 2 ] [ 2 ] [ 3 ] [ 3 ] [ 3 ]
     [ 4 ] [ 4 ] [ 4 ] [ 5 ] [ 5 ] [ 5 ] [ 6 ] [ 6 ] [ 6 ]
     [ 4 ] [ 4 ] [ 4 ] [ 5 ] [ 5 ] [ 5 ] [ 6 ] [ 6 ] [ 6 ]
     [ 4 ] [ 4 ] [ 4 ] [ 5 ] [ 5 ] [ 5 ] [ 6 ] [ 6 ] [ 6 ]
     [ 7 ] [ 7 ] [ 7 ] [ 8 ] [ 8 ] [ 8 ] [ 9 ] [ 9 ] [ 9 ]
     [ 7 ] [ 7 ] [ 7 ] [ 8 ] [ 8 ] [ 8 ] [ 9 ] [ 9 ] [ 9 ]
     [ 7 ] [ 7 ] [ 7 ] [ 8 ] [ 8 ] [ 8 ] [ 9 ] [ 9 ] [ 9 ]]

     {The Procrastination Matrix!}

     This puzzle is solved as follows:


    For each number in x, create a 3x3 array of that number and store each array in another bigger array.
    Split the bigger array into 3 smaller arrays. This creates an array which looks like the solution but not the same
    shape (shape = 3, 3, 3, 3).

    To convert this array into the shape of the solution, the arrays are then concatenated together twice.
        conc1.shape = (3, 9, 3)
        conc2.shape = (9, 9)

    """

    return np.concatenate(np.concatenate(
        np.array_split([np.full([3, 3], num) for num in [ele for row in x for ele in row]], 3), axis=1), axis=1)


def solve_f8b3ba0a(x):
    """
    This puzzle requires ranking three colours in order of how often they appear in the rows. Most occurences is first
    and so on. Black is empty space and the colour of the rows in which the colours appear on does not count towards the
    count e.g. yellow, blue and green squares appear on rows of red squares, the yellow, blue and green squares are
    ranked and the red squares are ignored.

    Solved as follows:
    Counts all the occurrences of each number and stores the counts in a counter object. The counter object stores
    them in order of largest count to smallest. The keys from the counter object, excluding the numbers representing the
    background colour and placeholder colour, are passed to a numpy array which is then reshaped to the solution"""

    from collections import Counter

    c = Counter([z for y in x for z in y])
    return np.array([n for n in c][2:]).reshape(-1, 1)


def solve_42a50994(x):
    """
    This puzzle requires all squares with no neighbour in the 3x3 around it to be removed from the grid. 0's are empty
    space and any number is a coloured square
    It is solved with the following:

    Neighbouring_sum finds all the valid neighbours surrounding a given point. The below gris is an example.

    [ y ] [ y ] [ y ] [ y ] [ y ]
    [ y ] [ x ] [ x ] [ x ] [ y ]
    [ y ] [ x ] [ V ] [ x ] [ y ]
    [ y ] [ x ] [ x ] [ x ] [ y ]
    [ y ] [ y ] [ y ] [ y ] [ y ]

    Where V is the point in question, the x's are the neighbours and the y's are non-neighbours.

    The following is an example if V happens to be at the edge of the grid/array:

    [ y ] [ y ] [ y ]
    [ x ] [ x ] [ y ]
    [ V ] [ x ] [ y ]
    [ x ] [ x ] [ y ]
    [ y ] [ y ] [ y ]

    Neighbouring_sum then sums the values of all the neighbours and returns this value. If this value is 0, then the
    point that was passed to neighbouring_sum is set to 0 as this point has no neighbours.
    """
    X, Y = x.shape

    neighbouring_sum = lambda x_, y_: np.sum(np.array([x[_x][_y]
                                                       for _x in range(x_ - 1, x_ + 2)
                                                       for _y in range(y_ - 1, y_ + 2)
                                                       if (X > _x >= 0
                                                           and Y > _y >= 0
                                                           and (x_, y_) != (_x, _y))]))

    for r in range(X):
        for c in range(Y):
            if x[r][c] != 0 and neighbouring_sum(r, c) == 0:
                x[r][c] = 0

    return x


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))
    print('\n')


if __name__ == "__main__": main()
