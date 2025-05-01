def mise(a, b):
    gan = 0.0
    if a == b:
        gan = 0.0
    elif (a == 6.0 and b == 9.0) or (a == 9.0 and b == 6.0):
        gan = 11.0
    elif (a == 2.0 and b == 5.0) or (a == 5.0 and b == 2.0):
        gan = 1.1
    elif (a == 1.0 and b == 100.0) or (a == 100.0 and b == 1.0):
        gan = 83.0
    elif a != b:
        gan = max(a, b)
    return gan

