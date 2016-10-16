import h5py


def load_h5():
    with h5py.File('data.h5', 'r') as hf:
        for item in hf.items():
            print(item)
        metadata = hf.get('metadata')

        for item in metadata.iteritems():
            print(item.)

load_h5()