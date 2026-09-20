import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board.pattern import Stabiliser, readings_from, EMPTY

def frames(s, reading, n, t0, step=1/30):
    out=[]
    for k in range(n):
        out += s.update(reading, t0 + k*step)
    return out, t0 + n*step

ok = True
def check(name, got, want):
    global ok
    if got != want:
        ok = False
        print(f'  FAIL {name}: got {got!r}, wanted {want!r}')
    else:
        print(f'  pass {name}')

print('a ball must persist before it counts')
s=Stabiliser(); t=0.0
one=[EMPTY]*64; one[5]='pink'
c,t = frames(s, one, 3, t)            # 0.1 s
check('not yet at 0.1 s', c, [])
check('pattern untouched', s.pattern[5], EMPTY)
check('but it is pending', s.pending(), {5:'pink'})
c,t = frames(s, one, 4, t)            # past 0.2 s
check('settles after 0.2 s', c, [(5, EMPTY, 'pink')])
check('pattern holds it', s.pattern[5], 'pink')

print('\na single flickering frame is ignored')
s=Stabiliser(); t=0.0
flick=[EMPTY]*64; flick[7]='blue'
for k in range(20):                    # alternate every frame for 0.66 s
    s.update(flick if k%2 else [EMPTY]*64, t); t+=1/30
check('never settles', s.pattern[7], EMPTY)

print('\nlosing the board holds the pattern, never clears it')
s=Stabiliser(); t=0.0
c,t = frames(s, one, 10, t)
check('ball is in', s.pattern[5], 'pink')
c,t = frames(s, [None]*64, 60, t)      # two seconds blind
check('no changes while blind', c, [])
check('ball still there', s.pattern[5], 'pink')
check('reports blind', s.blind, True)

print('\nremoving a ball works the same way round')
c,t = frames(s, [EMPTY]*64, 10, t)
check('removal settles', s.pattern[5], EMPTY)

print('\na transient board-wide change never lands')
s=Stabiliser(); t=0.0
c,t = frames(s, [EMPTY]*64, 5, t)
everything=['yellow']*64
c,t = frames(s, everything, 20, t)     # 0.66 s, longer than a normal hold
check('nothing let through yet', c, [])
check('flagged as suspect', s.blocked, True)
check('pattern intact', set(s.pattern), {EMPTY})
c,t = frames(s, [EMPTY]*64, 10, t)     # it goes away again
check('and it leaves no trace', set(s.pattern), {EMPTY})

print('\nbut a board that really is full settles, just slowly')
s=Stabiliser(); t=0.0
full=['pink']*20 + [EMPTY]*44
c,t = frames(s, full, 20, t)           # 0.66 s
check('not at 0.66 s', c, [])
c,t = frames(s, full, 15, t)           # past 1.0 s
check('settles by 1.2 s', len(c), 20)
check('pattern has them', len(s.filled()), 20)

print('\nbut a handful at once is fine, that is just playing')
s=Stabiliser(); t=0.0
few=[EMPTY]*64
for i in (1,2,3,4,5): few[i]='green'
c,t = frames(s, few, 10, t)
check('five settle together', sorted(i for i,_,_ in c), [1,2,3,4,5])
check('not blocked', s.blocked, False)

print('\ngeometry failure routes to no data')
check('unplaced gives all None', readings_from([{'colour':'pink'}]*64, False), [None]*64)
check('placed passes through', readings_from([{'colour':'pink'}]*64, True)[:2], ['pink','pink'])

print('\nALL PASS' if ok else '\nFAILURES ABOVE')
sys.exit(0 if ok else 1)
