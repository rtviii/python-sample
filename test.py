

n = 1024
vals = [-1,0,1]
index = 0
for x in range(6000):
    if not (x + 1) & (n - 1):
        index+=1
        vals[ index%len(vals)]
        

        

        
    