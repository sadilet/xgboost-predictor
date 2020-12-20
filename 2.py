import datetime

import xgboost
import numpy as np

import xgboost_predictor


if __name__ == "__main__":
    data = []
    with open("tests/resources/data/agaricus.txt.0.test", "r") as f:
        for line in f.readlines():
            row = [0] * 126
            for i in line.split(" ")[1:]:
                f, v = i.split(":")
                row[int(f)] = int(v)
            data.append(tuple(row))


    booster1 = xgboost.Booster({"nthread": 1})
    booster1.load_model('tests/resources/model/gblinear/v40/binary-logistic.model')
    data1 = xgboost.DMatrix(np.array(data, dtype=np.float32))
    start = datetime.datetime.now()
    a = booster1.predict(data1)
    print(datetime.datetime.now() - start)
    
    booster2 = xgboost_predictor.load_model('tests/resources/model/gblinear/v40/binary-logistic.model')
    data2 = np.array(data, dtype=np.float32)

    start = datetime.datetime.now()
    b = booster2.predict_many(data2)
    print((datetime.datetime.now() - start))

    # print(a)
    # print(b)
    # print(a, b)


"""
[1,2,3,4, 1,2,3,4, 1,2,3,4]

(0 * 3) + 0 = 0
(0 * 3) + 1 = 1
(0 * 3) + 2 = 2
(0 * 3) + 3 = 3

(1 * 3) + 0 = 3 +
(1 * 3) + 1 = 4
(1 * 3) + 2 = 5
(1 * 3) + 3 = 6

(2 * 3) + 0 = 6 + 2
(2 * 3) + 1 = 7
(2 * 3) + 2 = 8
(2 * 3) + 3 = 9
"""