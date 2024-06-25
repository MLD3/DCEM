from functools import partial, wraps, update_wrapper

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.metrics import roc_auc_score, average_precision_score

MAX_TRIES_AFTER_OOM = 10
RETRY_S = 30

def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)] #sort the predictions first
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n): # visit the examples in increasing order of predictions.
        y_i = y_true[i]
        nfalse += (1 - y_i) # negative (RIGHT) examples seen so far
        auc += y_i * nfalse # Each time we see a positive (LEFT) example we add the number of negative examples we've seen so far
    try:
        auc /= (nfalse * (n - nfalse)) # catch div by zero
    except ZeroDivisionError:
        print("ZeroDivsionError caught in fast_auc() - setting AUC to -1 as there are zero pairs - AUC is vacuous")
        return -1
    return auc
'''
cross_auc for the Ra0 > Rb1 error
function takes in scores for (a,0), (b,1)
'''
def cross_auc(R_a_0, R_b_1):
    scores = np.hstack([R_a_0, R_b_1])
    y_true = np.zeros(len(R_a_0)+len(R_b_1))
    y_true[0:len(R_a_0)] = 1 # Pr[ LEFT > RIGHT]; Y = 1 is the left (A0)
    return fast_auc(y_true, scores)

def linear_warmup_multiplier(curr, warmup, rampup_time=16):
    return np.clip((curr - warmup) / rampup_time, 0.0, 1.0)

def retry_if_oom(f, *args, retries=MAX_TRIES_AFTER_OOM, delay=RETRY_S, backoff=1, **kwargs):
    t = delay
    for i in range(retries):
        try:
            func_results = f(*args, **kwargs)
            break
        except RuntimeError as e:
            if 'out of memory' in str(e): # see https://discuss.pytorch.org/t/how-to-clean-gpu-memory-after-a-runtimeerror/28781/3
                if i != MAX_TRIES_AFTER_OOM - 1:
                    while t:
                        print(f"WARNING: Ran out of memory on attempt #{i}. Retrying in {t}s...", end="\r")
                        time.sleep(1)
                        t -= 1
                    t *= backoff
                    continue
                else:
                    print(f"ERROR: Ran out of memory the maximum number of times ({MAX_TRIES_AFTER_OOM}). Reraising error.\n")
                    raise e
            else:
                raise e
    return func_results

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func