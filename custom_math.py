

# from https://www.reddit.com/r/gamemaker/comments/6nk3us/help_is_there_a_gamemaker_equivalent_to_the_map/
"""
Maps a value in [pre_min, pre_max] to another in [post_min, post_max]
"""
def map(value, pre_min, pre_max, post_min, post_max):
    prop = (value - pre_min) / (pre_max - pre_min);
    return (prop * (post_max - post_min)) + post_min;

def interpolate_np(ndarr1, ndarr2, t):
    weighted_1 = ndarr1 * t
    weighted_2 = ndarr2 * (1-t)
    return weighted_1 + weighted_2
