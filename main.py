import csv
import random

x = []
y = []

BATCH_SIZE = 500
LEARNING_RATE = 0.00000006
EPOCHS = 50

def load_data(path: str):
    with open(path,"r") as file:
        reader = csv.reader(file)
        next(reader)
        for idx,row in enumerate(reader):
            x.append((1,int(row[1]))) #index 0 is the coefficient thing for the bias
            y.append(int(row[0]))


load_data("Housing.csv")
theta = [random.uniform(-0.01,0.01) for i in range(len(x[0]))]

def hypothesis(x: tuple[int], thetas: list[int]) -> int: #input 1 single input example and the list of all thetas
    assert len(x) == len(thetas), "Input x and Thetas must have the same length"

    return sum(x[i]*thetas[i] for i in range(len(x)))

def error(h: int, y: int) -> float:
    return 0.5*(h-y)**2

def minibatch(size: int, max: int) -> set[int]:
    s = set()
    while len(s)<size:
        s.add(random.randint(0,max-1))
    return s

for ii in range(EPOCHS):
    batch = minibatch(BATCH_SIZE, len(x))
    batch = tuple(batch)
    thetas2 = theta.copy()
    a=LEARNING_RATE
    for idx, value in enumerate(theta):
        thetas2[idx] -= a * sum([  (hypothesis(x[i],theta)-y[i]) * x[i][idx] for i in batch  ])/len(batch)

    theta = thetas2
    print(theta)


print("Done")
print(hypothesis(x[0],theta))
