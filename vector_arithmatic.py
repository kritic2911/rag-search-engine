def add_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

def subtract_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]

def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError
    return sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])

def magnitude(vec):
    return (sum([v ** 2 for v in vec])) ** 0.5

def cosine_sim(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError
    mag1, mag2 = magnitude(vec1), magnitude(vec2)

    if mag1 * mag2 == 0:
        return 0

    return dot(vec1, vec2) / (mag1 * mag2)
    
vec1 = [0.8, 0.5,  0.5]
vec2 = [.5, .4, .6]

print(add_vectors(vec1, vec2))
print(subtract_vectors(vec1, vec2))
print(dot(vec1, vec2))