import numpy as np
import random


memory_size = 10
memory = [0] * memory_size
memory_ptr = 0 

def recall(batch):
    global memory, memory_ptr
    N, M, P = len(batch), len(memory), memory_ptr
    if N >= M:
        memory = batch[N-M:N]
        memory_ptr = 0

    elif N < M:
        if P + N <= M:
            memory[P:P+N] = batch
            memory_ptr = P + N
            idx = list(range(P,P+N))
        elif M < P + N:
            memory[P:M] = batch[0:M-P]
            memory[0:P+N-M] = batch[M-P:N]
            memory_ptr = P+N-M
            idx = list(range(P,M)) + list(range(0,P+N-M))
        else:
            raise Exception
    else:
        raise Exception
    return idx

def get_batch(size):
    return [random.randint(0,10) for _ in range(size)]


for i in range(3):
    print(f'round {i}')    
    batch_size = random.randint(4, 5)
    batch = get_batch(batch_size)
    print(batch)
    idx = recall(batch)
    print(memory)
    print(idx)


