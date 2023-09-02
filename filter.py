import numpy as np

X_train = np.load('old_data/Dataset_new/PTB-XL_X_trainf.npy', allow_pickle=True)
y_train = np.load('old_data/Dataset_new/PTB-XL_y_trainf.npy', allow_pickle=True)
X_test = np.load('old_data/Dataset_new/PTB-XL_X_testf.npy', allow_pickle=True)
y_test = np.load('old_data/Dataset_new/PTB-XL_y_testf.npy',allow_pickle=True)

bad_idx = []
for i, y in enumerate(y_train):
    classes = ['NORM', 'CD', 'STTC', 'MI', 'HYP']
    isin = False
    for c in classes:
        if c in y:
            isin = True
            break
    if not isin:
        bad_idx.append(i)
X_train = np.delete(X_train, bad_idx, axis = 0)
y_train = np.delete(y_train, bad_idx, axis = 0)

bad_idx = []
for i, y in enumerate(y_test):
    classes = ['NORM', 'CD', 'STTC', 'MI', 'HYP']
    isin = False
    for c in classes:
        if c in y:
            isin = True
            break
    if not isin:
        bad_idx.append(i)
X_test = np.delete(X_test, bad_idx, axis = 0)
y_test = np.delete(y_test, bad_idx, axis = 0)

np.save('Dataset2/PTB-XL_X_train.npy', X_train)
np.save('Dataset2/PTB-XL_y_train.npy', y_train)
np.save('Dataset2/PTB-XL_X_test.npy', X_test)
np.save('Dataset2/PTB-XL_y_test.npy', y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)