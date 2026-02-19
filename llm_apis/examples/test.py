
def func(a, b):
    yield a
    
    print(b)
    yield b[0]

x = []

for a in func('asdf', x):
    print(a)
    x.append('1234')

