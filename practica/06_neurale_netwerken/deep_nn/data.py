# create crosses and dots data
img01 = {
        "img": [[1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]],
        "label": [1, 0]
        }  
        
img02 = {
        "img": [[1, 0, 1],
                [0, 1, 0],
                [0, 0, 1]],
        "label": [1, 0]
        }

img03 = {
        "img": [[0, 0, 1],
                [0, 1, 0],
                [1, 0, 1]],
        "label": [1, 0]
        }

img04 = {
        "img": [[1, 0, 0],
                [0, 1, 0],
                [1, 0, 1]],
        "label": [1, 0]
        }

img05 = {
        "img": [[1, 0, 1],
                [0, 1, 0],
                [1, 0, 0]],
        "label": [1, 0]
        } 

img06 = {
        "img": [[1, 0, 1],
                [0, 0, 0],
                [1, 0, 1]],
        "label": [1, 0]
        }

img07 = {
        "img": [[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]],
        "label": [0, 1]
        }

img08 = {
        "img": [[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]],
        "label": [0, 1]
        }

img09 = {
        "img": [[1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]],
        "label": [0, 1]
        }

img10 = {
        "img": [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]],
        "label": [0, 1]
        }

img11 = {
        "img": [[1, 1, 1],
                [1, 0, 1],
                [0, 1, 0]],
        "label": [0, 1]
        }

img12 = {
        "img": [[1, 1, 0],
                [1, 0, 1],
                [1, 1, 1]],
        "label": [0, 1]
        }


img13 = {
        "img": [[1, 1, 1],
                [1, 0, 1],
                [0, 1, 1]],
        "label": [0, 1]
        }

img14 = {
        "img": [[1, 1, 1],
                [1, 1, 1],
                [0, 1, 1]],
        "label": [0, 1]
        }

def get_training_data():
    # create training data
    X = [img01["img"], img02["img"], img03["img"], img04["img"], img07["img"], img08["img"], img09["img"], img10["img"], img11["img"]]
    y = [img01["label"], img02["label"], img03["label"], img04["label"], img07["label"], img08["label"], img09["label"], img10["label"], img11["label"]]
    return X, y

def get_test_data():
    # create test data
    X = [img05["img"], img06["img"], img12["img"], img13["img"], img14["img"]]
    y = [img05["label"], img06["label"], img12["label"], img13["label"], img14["label"]]
    return X, y