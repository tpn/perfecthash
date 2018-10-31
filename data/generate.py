from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

# Toggle to true to enable writing .png and .txt files.
write_png = False
write_txt = False

# Turn off interactive mode to disable plots being displayed
# prior to saving them to disk.
plt.ioff()

def save_array_plot_to_png_file(filename, a):
    plt.plot(a)
    plt.savefig(filename)
    print("Generated %s." % filename)

def save_array_to_text_file(filename, a):
    with open(filename, 'wb') as f:
        for i in a:
            f.write(str(i).zfill(9).encode('ascii'))
            f.write(b'\n')
    print("Generated %s." % filename)

def save_array_to_binary_file(filename, a):
    fp = np.memmap(filename, dtype='uint32', mode='w+', shape=a.shape)
    fp[:] = a[:]
    del fp
    print("Generated %s." % filename)

def save_array(prefix, a):
    l = len(a)
    binary_filename = '%s-%d.keys' % (prefix, l)
    save_array_to_binary_file(binary_filename, a)
    if write_png:
        png_filename = '%s-%d.png' % (prefix, l)
        save_array_plot_to_png_file(png_filename, a)
    if write_txt:
        text_filename = '%s-%d.txt' % (prefix, l)
        save_array_to_text_file(text_filename, a)

def gen_random_unique_array(size):
    a = np.random.randint(0, (1 << 32)-1, size * 2, dtype='uint32')
    a = np.unique(a)[:size]
    return a if len(a) == size else None

def gen_linear_array(size):
    a = np.linspace(0, size, num=size, dtype='uint32')
    return a

def gen_curved_unique_array(size):
    dsize = size * 32
    m = np.arange(start=10000, stop=500000, step=1)
    r = np.random.choice(m, size=dsize)
    n = np.random.normal(0, 0.1, dsize)
    b = r * n
    f = np.floor(np.abs(b))
    i = f.astype(np.uint32)
    u = np.unique(i)
    a = u[:size]
    return a if len(a) == size else None

def gen_spiked_unique_array(size):
    dsize = size * 32
    g = np.random.geometric(p=0.7, size=dsize)
    l = np.random.logseries(0.6, size=dsize)
    m = np.arange(start=10000, stop=500000, step=1)
    r = np.random.choice(m, size=dsize)
    n = np.random.normal(0, 0.1, dsize)
    b = g * l * r * n
    f = np.floor(np.abs(b))
    i = f.astype(np.uint32)
    u = np.unique(i)
    a = u[:size]
    return a if len(a) == size else None

normal_sizes = (
    2000,
    4000,
    4050,
    5000,
    10000,
    15000,
    17000,
    25000,
    31000,
    33000,
    50000,
    63000,
    65500,
    75000,
    100000,
    121000,
    125000,
    150000,
    200000,
    225000,
    245000,
    255000,
    265000,
    389161,
    472374,
)

large_sizes = (
    618201,
    789291,
    976562,
   1038181,
   1893815,
   2000000,
   3000000,
   4000000,
)

even_larger_sizes = (
   5910810,
   6182344,
   7742919,
   10000000,
)

# Toggle this depending on which sizes you want.
sizes = normal_sizes
#sizes = large_sizes
#sizes = even_larger_sizes

functions = (
    ('linear', gen_linear_array),
    ('random', gen_random_unique_array),
    #('curved', gen_curved_unique_array),
    #('spiked', gen_spiked_unique_array),
)

def main():
    for (name, func) in functions:
        for size in sizes:
            a = func(size)
            if a is None:
                continue
            save_array(name, a)

if __name__ == '__main__':
    main()
