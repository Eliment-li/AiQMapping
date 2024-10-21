import math




def test_reward_function():
    #distance = [100,26.11,19.06,23.06,19.06]
    #distance = [24.36,26.11,19.06,23.06,19.06]
    distance = [2,1,2,1]
    default = distance[0]
    last = default
    total = 0
    for i,v in enumerate(distance):

        k1 = (default - v) / default
        k2 = (last - v) / last
        if k1==0:
            k1=0.5
        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * math.fabs(k1)
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * math.fabs( k1)
        else:
            reward = 0

        total = total*0.999 + reward
        print(f'{i}= {reward.__round__(4)}, total={total.__round__(2)}')

        last = v

test_reward_function()
