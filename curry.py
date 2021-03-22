


def outer(x):
    print("outer x : {}".format(x))
    def inner(y):
        print("inner y:{}".format(y))
        return y+x
    return inner
    
print(outer(2)(3))

