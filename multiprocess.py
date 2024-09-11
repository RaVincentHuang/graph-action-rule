import random
import multiprocessing
from tqdm import tqdm


if __name__ == '__main__':
    def write_file(i):
        with open(f"test/{i}", 'w') as file:
            file.write("Hello, world!")
    
    pool = multiprocessing.Pool(processes=16)
    with tqdm(total=100) as pbar:
        for i in range(100):
            pool.apply_async(write_file, (i,))
            pbar.update(1)
    pool.close()
    pool.join()
