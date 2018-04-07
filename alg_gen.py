import math as m
import numpy as np

def obj_func(args):
    x,y = args
    a = 20
    b = 0.2
    c = 2*np.pi
    d = 5.7
    f = 0.8
    n = 2
    return (1./f)*(-a*m.exp(-b*m.sqrt((1/n)*(x*x+y*y)))-m.exp((1/n)*(m.cos(c*x)+m.cos(c*y)))+a+m.exp(1)+d)

def nbits(a, b, dx):
    c=abs((b-a)/dx+1)
    d=2
    i=1
    while(c>d):
        d*=2
        i+=1
    return (i,(b-a)/(d-1))

def gen_population(P, N, B):
    return np.random.randint(2, size=(P, N*B))

def decode_population(pop, P, N, B, a, b, dx):
    arr = []
    for p in pop:
        r = []
        for x in range(N):
            t = p[x*B:(x+1)*B]
            r.insert(x, 0)
            for y in range(B):
                r[x] += t[y] * 2 ** (B-y-1)
        arr.append(r)
    return arr

def evaluate_population(pop, func, P, N, a, b, dx):
    arr = []
    for p in pop:
        args = []
        for n in range(N):
            args.append(a+p[n]*dx)
        arr.append(-obj_func(args))
    return arr

def get_best(pop, evaluated_pop):
    max = 0
    for x in range(len(evaluated_pop)):
        if(evaluated_population[max] < evaluated_pop[x]):
            max = x
    return max

def avg(pop, evaluated_pop):
    avg = 0
    for x in range(len(evaluated_pop)):
        avg+= evaluated_pop[x]         
    return avg/len(evaluated_pop)

def adjust(pop_eval):
    min = 0
    for x in range(len(pop_eval)):
        if(pop_eval[min] > pop_eval[x]):
            min = x
    minabs = abs(pop_eval[min])
    for x in range(len(pop_eval)):
        pop_eval[x] += minabs

def roulette(pop, pop_eval):
    max = 0
    ranges = []
    new_pop = []

    for x in pop_eval:
        ranges.append((max, max + x))
        max += x
    for i in range(len(pop)):
        rn = np.random.random() * max
        for j in range(len(ranges)):
            if(rn >= ranges[j][0] and rn <= ranges[j][1]):
                new_pop.append(pop[j])
                break

    return new_pop

def cross(pop, pk):
    crossed_pop = []
    halfsize = int(len(pop[0]) / 2)

    for i in range(int(len(pop) / 2)):
        if(np.random.random() < pk):
            crossed_pop.append(list(pop[i*2][:halfsize]) + list(pop[i*2 + 1][halfsize:]))
            crossed_pop.append(list(pop[i*2 + 1][:halfsize]) + list(pop[i*2][halfsize:]))
        else:
            crossed_pop.append(pop[i*2])
            crossed_pop.append(pop[i*2 + 1])

    return crossed_pop

def mutate(pop, pm):
    size = len(pop[0]) - 1
    for i in range(len(pop)):
        if(np.random.random() < pm):
            j = np.random.randint(0, size)
            pop[i][j] = 0 if pop[i][j] == 1 else 1


pk, pm = 0.3, 0.01
it= 100
a, b = -1.5, 1.5
P, N = 300, 2
avg_list=[]
max_list = []
bits, dx = nbits(a, b, 0.0001)
population = gen_population(P, N, bits)
decoded_population = decode_population(population, P, N, bits, a, b, dx)
evaluated_population = evaluate_population(decoded_population, obj_func, P, N, a, b, dx)
best = get_best(decoded_population, evaluated_population)
print(decoded_population)
print(evaluated_population)
print(decoded_population[best], evaluated_population[best])
print('\n')

print('{} {}'.format('Najlepszy osobnik','    Srednia'))
for i in range(it):
    adjust(evaluated_population)
    population = roulette(population, evaluated_population)
    population = cross(population, pk)
    mutate(population, pm)
    decoded_population = decode_population(population, P, N, bits, a, b, dx)
    evaluated_population = evaluate_population(decoded_population, obj_func, P, N, a, b, dx)
    best = get_best(decoded_population, evaluated_population)
    average = avg(decoded_population, evaluated_population)
    avg_list.append(average)
    max_list.append(best)
    print('{} {}'.format(evaluated_population[best],average))

print(decoded_population[best], evaluated_population[best])
import matplotlib.pyplot as plt
plt.plot([i for i in range(it)],avg_list,'ro')
plt.ylabel('srednia')
plt.xlabel('ilosc iteracji')
plt.show()