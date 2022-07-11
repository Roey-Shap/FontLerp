

# from https://www.reddit.com/r/gamemaker/comments/6nk3us/help_is_there_a_gamemaker_equivalent_to_the_map/
"""
Maps a value in [pre_min, pre_max] to another in [post_min, post_max]
"""
def map(value, pre_min, pre_max, post_min, post_max):
    prop = (value - pre_min) / (pre_max - pre_min);
    return (prop * (post_max - post_min)) + post_min;