import torch
from Class3_kNNFaceSet import *

train_set = kNNFaceSet('/home/wangling/faceImages/project/known')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)  # , num_workers=8
print('OK! ', len(train_set), len(train_loader))

test_set = kNNFaceSet('/home/wangling/faceImages/project/unknown')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)  # , num_workers=8
print('OK! ', len(test_set), len(test_loader))

right_num = 0
for step, (x_test, y_test) in enumerate(test_loader):
    # print('in line: ', step, x_test.size(), y_test.size())
    x_test = x_test.reshape(-1)

    min_dis = 99999999.0
    min_y = -1

    for step1, (x_train, y_train) in enumerate(train_loader):
        # print('in line: ', step, x_train.size(), y_train.size())
        x_train = x_train.reshape(-1)

        cur_dis = np.abs(x_train - x_test)
        cur_dis = np.power(cur_dis, 2)
        cur_dis = cur_dis.sum(0)
        cur_dis = np.power(cur_dis, 0.5)

        if cur_dis < min_dis:
            min_dis = cur_dis
            min_y = y_train

    if y_test == min_y:
        right_num += 1
    print(y_test, min_y)

print(100.0 * right_num / len(test_set), r'%')
