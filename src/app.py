'''
PCA Visualization

Refresh command: 

python setup.py install && python .\examples\PCA.py --vis --port 2634 --n_component 4

'''
import os
import argparse
from pathlib import Path
from visTorch import visboard
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import re
import numpy as np

def load_surfaces(datfile_path):
    surfaces = []
    for datfile in Path(datfile_path).glob('*.dat'):
        # print(f"{p.name}:\n")
        current_surface = []

        with open(datfile) as f:
            lines = [l.strip() for l in f.readlines()]

        for line in lines:
            try:
                xy = [float(str(i)) for i in re.split("\s", line) if len(str(i)) > 0]
            except Exception as e:
                print("Problem decoding line \"%s\"" % line)
                raise e
            if len(xy) > 1:
                current_surface.append([float(xy[0]), float(xy[1])])
        
        surfaces.append(current_surface)

    print ("Got %s surfaces" % len(surfaces))

    # verify all surfaces are of the same length
    len_verify = []
    for surface in surfaces:
        len_verify.append(len(surface))
    if len(set(len_verify)) > 1:
        raise Exception("Surfaces are not of the same length: %s" % set(len_verify))
    else:
        print ("All surfaces are of the same length (%s)" % len_verify[0])

    return surfaces

n_component = 6

# print(os.getcwd())
dataset_dir = os.path.join(os.getcwd(), 'dataset', 'picked_uiuc')
# dataset_dir = './dataset/picked_uiuc'
# print(dataset_dir)

# dataset
print('====Loading dataset...====')
data = load_surfaces(dataset_dir) 
data = np.array(data)
print('====Dataset loaded.====')

pca_model = PCA(n_components=n_component, svd_solver='full')

# Preprocess data
print('====Preprocessing data...====')
data_2d = np.array([surf.flatten() for surf in data])
scaler = StandardScaler()
data_2d_scaled = scaler.fit_transform(data_2d)
pca_model.fit(data_2d_scaled)
# data_pca = pca_model.transform(data_2d_scaled)
print('====Preprocessing Done...====')


# if args.visualize:
# initialize visualization app
vis_board = visboard()
vis_board.add_pca(pca_model, 
                data_2d_scaled, 
                latent_options={'n': pca_model.n_components_, 'min': -30, 'max': 30, 'step': 0.01},
                pre_process=scaler)
vis_board.prepare_server()
app = vis_board.app
server = app.server

if __name__ == '__main__':

    app.run_server()
