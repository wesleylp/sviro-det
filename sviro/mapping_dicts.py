CLASS2DET_DICT = {
    # mapping from classification (image name) to object detection (label)
    'empty': {
        'cls': [0],  # empty
        'det': 0
    },
    'infant_seat': {
        'cls': [1, 5],  # infant_seat_with_child and empty_infant_seat, respectively
        'det': 1  # infant_seat
    },
    'child_seat': {
        'cls': [2, 6],  # child_seat_with_child and empty_child_seat, respectively
        'det': 2  # child_seat
    },
    'person': {
        'cls': [1, 2, 3],  # infant, child and adult, respectively
        'det': 3  # person
    },
    'everyday_object': {
        'cls': [4],  # everyday_object
        'det': 4  # everyday_object
    },
}

# these are the labels used in the dataset
DET_IDXS_LABELS = {0: 'empty', 1: 'infant_seat', 2: 'child_seat', 3: 'person', 4: 'everyday_object'}
