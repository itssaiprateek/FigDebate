"""Decode only one bounded batch; caller-owned images are never closed."""
from itertools import islice


def decoded_batches(samples, size=32, decoder=None):
    if not isinstance(size, int) or size < 1:
        raise ValueError('Batch size must be a positive integer')
    if decoder is None:
        from dataset.loaders import decode_image
        decoder = decode_image
    source = iter(samples)
    while batch := list(islice(source, size)):
        batch = [dict(sample) for sample in batch]
        owned = []
        try:
            for sample in batch:
                if sample.get('image') is None:
                    sample['image'] = decoder(sample['raw']['image_bytes'])
                    owned.append(sample['image'])
            yield batch
        finally:
            for image in owned:
                image.close()
