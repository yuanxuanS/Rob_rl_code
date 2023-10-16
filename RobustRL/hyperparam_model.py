
def hyper_model(edge_features, z):
    multi = edge_features * z
    print(multi.sum(axis=1))