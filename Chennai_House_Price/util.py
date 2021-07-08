import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, INT_SQFT, DIST_MAINROAD, N_BEDROOM, N_BATHROOM, N_ROOM):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = INT_SQFT
    x[1] = DIST_MAINROAD
    x[2] = N_BEDROOM
    x[3] = N_BATHROOM
    x[4] = N_ROOM
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0]/100000)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artifacts/columns1.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[7:]  # first 5 columns are sqft, Distance From Mainroad, N_bedroom, N_bathroom, N_room

    global __model
    if __model is None:
        with open('./artifacts/Chennai_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    # other location