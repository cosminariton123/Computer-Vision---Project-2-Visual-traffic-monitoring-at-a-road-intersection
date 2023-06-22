
def rect_to_roirect(rect):
    
    x, y, v, w = rect

    return x, y, v - x, w - y


def roirect_to_rect(roirect):
    
    x, y, width, height = roirect

    return x, y, x + width, y + height