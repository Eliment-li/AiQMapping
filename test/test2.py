import math




def test_reward_function():
    distance = [3,10,9,8,7,6,5,4,3]
    default = distance[0]
    last = default
    total = 0
    for i,v in enumerate(distance):

        k1 = (default - v) / default
        k2 = (last - v) / last

        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * (1 + k1)
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * (1 - k1)
        else:
            reward = 0

        total = total*0.999 + reward
        print(f'{i}= {reward.__round__(4)}, total={total.__round__(2)}')

        last = v

test_reward_function()