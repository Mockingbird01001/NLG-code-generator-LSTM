import pickle, os

# Loading and saving files

def read_txt(path):
    return open(path, 'r', encoding='utf-8').read()


def read_dir(path='data/data_model/batch_{}/', index=0):
    text = ''
    dir_path = path.format(index)
    for filename in os.listdir(dir_path):
        text += read_txt(dir_path + filename)
    return text
        

def save_txt(text, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

        
def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_pickle(variable, path):
    with open(path, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)