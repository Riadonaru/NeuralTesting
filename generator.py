def meow():
    i = 0
    yield i
    i += 1
    yield i
    i += 1
    yield i
    i += 1
    yield i


res = meow()
next(res)
next(res)
next(res)


print(next(meow()))
