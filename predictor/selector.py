import random

ins_num = 1
ins_prices = [100]
ins_mems = [8]
ins_cores = [2]

def predictNetMem(mod):
    raise NotImplementedError

def predictNetTime(mod):
    raise NotImplementedError

def getAllCkptMods(mod):
    raise NotImplementedError


def getInsMinCost(ins_price, ins_mem, ins_core, mod, T):
    # cost = training time * price of instance
    # subject to
    # Training time < T
    # peak memory usage < memory of an instance

    min_cost = 10000000
    mods = getAllCkptMods(mod)
    min_ckpt_mod = None
    # enumerate different ckpt policies
    for cur_ckp_mod in mods:
        min_ckpt_mod = None
        mem = predictNetMem(cur_ckp_mod)
        if mem > ins_mem:
            # use too much memory
            continue
        t = predictNetTime(cur_ckp_mod)
        cur_cost = t * ins_price
        min_cost = min_cost if min_cost < cur_cost else cur_cost
        min_ckpt_mod = min_ckpt_mod if min_cost < cur_cost else cur_ckp_mod

    if min_ckpt_mod is None:
        print("[Error] Cannot find instance meet the training time constraint")
    return min_cost, min_ckpt_mod


def getInstance(mod, T):
    min_cost = 0
    min_ins_idx = -1
    for i in range(ins_num):
        ins_cost, _ = getInsMinCost(ins_prices[i], ins_mems[i], ins_cores[i], mod, T)
        if ins_cost < min_cost:
            min_cost = ins_cost
            min_ins_idx = i
    return min_ins_idx

def calMemNoCkpt(mod):
    return 0

if __name__ == "main":
    mod = None
    # our method
    min_idx = getInstance(mod, 10 * 3600)

    # random selection
    rand_idx = random.randint(0, ins_num)

    # max selection
    max_idx = ins_num - 1

    # just fit selection
    mod_mem_no_ckpt = calMemNoCkpt(mod)
    fit_idx = [x for x in ins_mems if x > mod_mem_no_ckpt][0]





