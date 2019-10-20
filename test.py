from pyModelChecking import *
from pyModelChecking.LTL import *


parser = Parser()
psi =X('one')

functions = {0: ['one'], 1: ['one'], 2: ['one'], 3: ['one'], 4: ['undefined'], 5: ['undefined'], 6: ['undefined'], 7: ['one'], 8: ['one'], 9: ['one'], 10: ['undefined'], 11: ['undefined'], 12: ['undefined'], 13: ['undefined'], 14: ['undefined'], 15: ['undefined'], 16: ['undefined'], 17: ['one'], 18: ['one'], 19: ['one'], 20: ['one'], 21: ['one'], 22: ['one'], 23: ['undefined'], 24: ['one'], 25: ['one'], 26: ['one'], 27: ['one'], 28: ['one'], 29: ['one'], 30: ['one'], 31: ['undefined'], 32: ['one'], 33: ['one'], 34: ['one'], 35: ['one'], 36: ['one'], 37: ['one'], 38: ['one'], 39: ['one']}
relations =  [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 40)]
K=Kripke(R=relations,L=functions)

print(modelcheck(K,psi))