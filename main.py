import numpy as np
import cvxopt as co
import cvxpy as cp

import numpy.testing as npt


co.solvers.options['show_progress'] = False
co.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_ERR'}

# https://www.analyzemath.com/linear_programming/linear_prog_applications.html


def example1_cvxpy():
    """ 
    A = 8
    B = 14

    Profit:
    A = 2
    B = 3

    A + B < 2000
    8A + 14B < 20000

    max 2A + 3B
    """

    A = cp.Variable()
    B = cp.Variable()

    constr = [
        A + B <= 2000,
        8*A + 14*B <= 20000,
        A >= 0,
        B >= 0,
    ]
    cp.Problem(cp.Maximize(2*A + 3 * B),
               constraints=constr).solve(solver='GLPK')
    return [A.value, B.value]


def example1_cvxopt():
    G = co.matrix(np.array([
        [-1, 0],
        [0, -1],
        [1, 1],
        [8, 14]
    ], dtype=float))

    c = co.matrix([-2.0, -3.0])

    h = co.matrix([-0.0, -0.0, 2000.0, 20000.0])

    sol = co.solvers.lp(c, G, h, solver='glpk')

    return np.array(sol['x']).flatten()


def example4_cvxpy():
    """
    F1 2%
    F2 4%
    F3 5%

    F3 < 3000
    F2 < 2*F1

    Max 2 F1 + 4 F2 + 5 F3
    """

    f1 = cp.Variable()
    f2 = cp.Variable()
    f3 = cp.Variable()

    constr = [
        f1 >= 0,
        f2 >= 0,
        f3 >= 0,
        f2 <= 2 * f1,
        f3 <= 3000,
        f1 + f2 + f3 <= 20000
    ]

    cp.Problem(cp.Maximize(2 * f1 + 4 * f2 + 5 * f3),
               constraints=constr).solve(solver='GLPK')
    return [f1.value, f2.value, f3.value]


def example4_cvxopt():
    G = co.matrix(np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [-2, 1, 0],  # f2 <= 2 * f1 --> -2 *f1 + f2 <= 0
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=float))

    c = co.matrix([-2.0, -4.0, -5.0])

    h = co.matrix([-0.0, -0.0, -0.0, -0.0, 3000.0, 20000.0])

    # default solver results in a diff @ 3 decimal
    sol = co.solvers.lp(c, G, h, solver='glpk')
    return np.array(sol['x']).flatten()


def example4_cvxopt_explit_z():
    """
    Saw in some examples that the objective is explictly listed as one of the
    variables. Not exactly see why it is done this way, as it makes the system
    of equitions longer and harder to understand... but anyways, 
    listed here so I can remember how it works. This is essentially doing minmax
    """
    G = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [-2, 1, 0],  # f2 <= 2 * f1 --> -2 *f1 + f2 <= 0
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=float)

    G = np.hstack((
        np.zeros((G.shape[0], 1)),
        G
    ))
    G = np.vstack((
        G,
        # maximize z st: z <= 2 F1 + 4 F2 + 5 F3
        # is the same as:
        # minimize z st: -z >= -2f1 - 4f2 - 5f3 ---> z - 2f1 - 4f2 - 5f3 <= 0
        np.array([1, -2, -4, -5])
    ))
    G = co.matrix(G)

    # resulting G
    #  first col is the objective, it does not participate in any of the constraits, other than
    #  the actual minimize objective listed in the very last row.
    #   |
    #   v
    # [[ 0. -1.  0.  0.]
    # [ 0.  0. -1.  0.]
    # [ 0.  0.  0. -1.]
    # [ 0. -2.  1.  0.]
    # [ 0.  0.  0.  1.]
    # [ 0.  1.  1.  1.]
    # [ 1. -2. -4. -5.]]

    c = co.matrix([-1.0, 0.0, 0.0, 0.0])  # minimize z

    h = co.matrix([-0.0, -0.0, -0.0, -0.0, 3000.0, 20000.0, -0.0])

    # default solver results in a diff @ 3 decimal
    sol = co.solvers.lp(c, G, h, solver='glpk')
    return sol['x'][0], np.array(sol['x'][1:]).flatten()


def rock_paper_scissors_cvxpy():
    r = cp.Variable()
    p = cp.Variable()
    s = cp.Variable()
    z = cp.Variable()  # obj

    constr = [
        r >= 0,
        p >= 0,
        s >= 0,
        r + p + s == 1,
        # rps rules using maxmin
        # max Z, s.t.:
        z <= +0*r - 1*p + 1*s,
        z <= +1*r + 0*p - 1*s,
        z <= -1*r + 1*p + 0*s,
    ]

    cp.Problem(cp.Maximize(z), constraints=constr).solve(solver='GLPK')
    print('rock paper scissors solution using cvxpy')
    print('expected value of the game: ', z.value)
    print('best stragegy: ', np.array([r.value, p.value, s.value]).flatten())

    # matrix form
    rpsrule = np.array([[0, -1, 1],
                        [1, 0, -1],
                        [-1, 1, 0]], dtype=float)
    rps = cp.Variable(3)
    z1 = cp.Variable()

    constr1 = [
        rps >= 0,
        sum(rps) == 1,
        # rps rules using maxmin
        # max Z, s.t.:
        z1 <= rpsrule @ rps
    ]
    cp.Problem(cp.Maximize(z1), constraints=constr1).solve(solver='GLPK')
    print('rock paper scissors solution using cvxpy Matrix')
    print('expected value of the game: ', z1.value)
    print('best stragegy: ', np.array([rps.value]).flatten())


def rock_paper_scissors_cvxopt():
    rpsrule = np.array([[0, -1, 1],
                        [1, 0, -1],
                        [-1, 1, 0]], dtype=float)

    G = co.matrix(np.vstack((
        # negate 'rpsrule' or not would generate the same result.
        # Without negating it is calcuating the probablity for column player.
        # Since this is a zero sum game, the stragegy for both row and column player would be identical.
        np.hstack((np.ones((3, 1)), - rpsrule)),
        np.hstack((np.zeros((3, 1)), - np.eye(3))),  # each P >= 0
    )))

    c = co.matrix([-1.0, 0.0, 0.0, 0.0])
    h = co.matrix(np.zeros(G.size[0]))

    # sum P == 1
    A = co.matrix(np.array([[0.0, 1.0, 1.0, 1.0]]))
    b = co.matrix([1.0])

    sol = co.solvers.lp(c, G, h, A, b, solver='glpk')

    print('rock paper scissors solution using cvxopt')
    print('expected value of the game: ', sol['x'][0])
    print('best stragegy: ', np.array(sol['x'][1:]).flatten())


if __name__ == "__main__":
    ex1_coeff = np.array([2, 3])
    npt.assert_almost_equal(
        np.dot(ex1_coeff, example1_cvxpy()),
        np.dot(ex1_coeff, example1_cvxopt()), decimal=10)

    ex4_coeff = np.array([2, 4, 5])
    npt.assert_almost_equal(
        np.dot(ex4_coeff, example4_cvxpy()),
        np.dot(ex4_coeff, example4_cvxopt()), decimal=10)

    sol = example4_cvxopt()
    obj = np.dot(ex4_coeff, sol)
    obj_z, sol_z = example4_cvxopt_explit_z()

    print('normal vs explicit z:')
    print(f'sol: \n{sol}\n{sol_z}\n')
    print(f'obj: \n{obj}\n{obj_z}')

    print()
    rock_paper_scissors_cvxopt()
    print()
    rock_paper_scissors_cvxpy()
