import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_PIXEL_VALUE = 255.0


def _normalize():
    return A.Normalize(
        mean=IMAGENET_MEAN,
        std=tuple(s * MAX_PIXEL_VALUE for s in IMAGENET_STD),
        max_pixel_value=MAX_PIXEL_VALUE,
    )


def _flip():
    return A.OneOf(
        [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
        ],
        p=0.5,
    )


def seg_transform(split, size):
    if split == "train":
        return A.Compose(
            [
                A.RandomRotate90(p=0.5),
                _flip(),
                A.Resize(size, size),
                _normalize(),
                A.ToTensorV2(transpose_mask=True),
            ]
        )
    if split == "val":
        return A.Compose(
            [
                A.Resize(size, size),
                _normalize(),
                A.ToTensorV2(transpose_mask=True),
            ]
        )
    raise ValueError(f"Unrecognized `split` found: {split}.")
