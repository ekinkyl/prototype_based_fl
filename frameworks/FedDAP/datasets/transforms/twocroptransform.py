class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform_left, base_transform_right):
        self.base_transform_left = base_transform_left
        self.base_transform_right = base_transform_right

    def __call__(self, x):
        q = self.base_transform_left(x)
        k = self.base_transform_right(x)
        return [q, k]