import numpy as np

def load_glove_file(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    #this is the glove model, each word is a key and the value is its vector representation
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

if __name__ == '__main__':
    glove_model = load_glove_file("/Users/denizsonmez/Downloads/glove.6B/glove.6B.200d.txt")

    #testing if we got the embeddings right
    print(glove_model['me'])