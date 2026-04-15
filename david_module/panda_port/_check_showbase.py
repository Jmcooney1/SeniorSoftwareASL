from direct.showbase.ShowBase import ShowBase

print('ShowBase methods containing open/window/graphics:')
names = [name for name in dir(ShowBase) if any(k in name.lower() for k in ('open', 'window', 'graphics'))]
for n in sorted(names):
    print(' -', n)

print('\nHas attribute openDefaultWindow?:', hasattr(ShowBase, 'openDefaultWindow'))
