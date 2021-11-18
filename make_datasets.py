import json
import os
import random
import math
import shutil
import sys
from MeshPly import MeshPly
from miso_utils import *

'''##############################################################################################
# DESCRIPTION
 데이터셋의 필요한 속성은 실행한 폴더를 기준으로 ./data 경로에 파일들을 생성하며, 이미지 파일은 용량이 크기 때문에 NAS /dataset의 하위 경로에서 읽어오도록 한다.

<OPTIONS>
dataset_base_dir : 객체 3D 데이터 경로(라벨링데이터, 원천데이터 폴더를 포함)]
data : 객체 3D 대상명
train_ratio : train/test dataset의 비율


<OUTPUT>
LINEMOD 데이터 구성을 위해, 다음 파일들을 생성합니다.
- test.txt
- train,txt
- [폴더명].data
##############################################################################################'''

dataset_base_dir="/mnt/hackerton/dataset/Dataset/07.투명객체3D/"
data = "010304.소스용기4"
#각 데이터 셋마다의 클래스 리스트
data_ts_list = [
    "010201.스프레이병1",
    "010202.스프레이병2",
    "010204.스프레이병4",
    "010304.소스용기4",
    "010305.소스용기5",
    "010312.소스용기12",
    "010313.소스용기13",
    "010321.소스용기21",
    "040207.계량컵7",
    "040209.계량컵9"
]
data_big_list=[
    "030102.쉐이커",
    "060201.무우",
    "070205.물컵",
    "070308.물조리개",
    "070608.각티슈",
    "070702.라이터",
    "070708.휴지통",
    "070710.향초",
    "090206.하모니카",
    "100211.무드등"
]
def main():
    dataset_ts_base_dir="/mnt/hackerton/dataset/Dataset/07.투명객체3D/"
    dataset_big_base_dir="/mnt/hackerton/dataset/Dataset/08.대용량객체3D/"
    flag = 'nothing'
    if len(sys.argv) == 3:
        if sys.argv[2] == 'TS':
            dataset_ts_base_dir = sys.argv[1]
            flag = 'ts'
        elif sys.argv[2] == 'BIG':
            dataset_big_base_dir = sys.argv[1]
            flag = 'big'
    train_ratio = 0.8  # test_ratio = 1 -train_ratio
    # 0:대용량 1:투명
    type_file = [".Images", ".TR"]

    ##투명
    # 원천데이터
    origin_ts_data_dir = []
    origin_ts_image_dir = []
    origin_ts_threed_shape_data_dir = []
    #라벨링
    labeling_ts_dir = []
    labeling_ts_threed_json_dir = []
    ##대용량
    #원천데이터
    origin_big_data_dir = []
    origin_big_image_dir = []
    origin_big_threed_shape_data_dir = []
    #라벨링
    labeling_big_dir = []
    labeling_big_threed_json_dir = []

    #각 생성 데이터별 경로 이름 지정
    for i in range(10):
        origin_ts_data_dir[i] = dataset_ts_base_dir + "/" + data_ts_list[i] + "/" + data_ts_list[i] + ".원천데이터"
        origin_ts_image_dir[i] = origin_ts_data_dir[i] + "/" + data_ts_list[i] + type_file[1]
        origin_ts_threed_shape_data_dir[i] = origin_ts_data_dir[i] + "/" + data_ts_list[i] + ".3D_Shape"
        origin_big_data_dir[i] = dataset_big_base_dir + "/" + data_big_list[i] + "/" + data_big_list[i] + ".원천데이터"
        origin_big_image_dir[i] = origin_big_data_dir[i] + "/" + data_big_list[i] + type_file[0]
        origin_big_threed_shape_data_dir[i] = origin_big_data_dir[i] + "/" + data_big_list[i] + ".3D_Shape"
        labeling_ts_dir = dataset_ts_base_dir[i] + "/" + data_ts_list[i] + "/" + data_ts_list[i] + ".라벨링데이터"
        labeling_ts_threed_json_dir = labeling_ts_dir[i] + "/" + data_ts_list[i] + ".3D_json"
        labeling_big_dir = dataset_big_base_dir[i] + "/" + data_big_list[i] + "/" + data_big_list[i] + ".라벨링데이터"
        labeling_big_threed_json_dir = labeling_big_dir[i] + "/" + data_big_list[i] + ".3D_json"

    if not (os.path.isdir(os.path.join('cfg'))):
    os.makedirs(os.path.join('cfg'))
    
    if not (os.path.isdir("data")):
        os.makedirs(os.path.join("data"))
    #투명
    for i in range(10):
        if not (os.path.isdir(os.path.join("data",data_ts_list[i]))):
            os.makedirs(os.path.join("data",data_ts_list[i]))
        if not (os.path.isdir(os.path.join("data",data_big_list[i]))):
            os.makedirs(os.path.join("data",data_big_list[i]))
    dataProperty_ts_list = []
    dataProperty_big_list = []
    for i in range(10):
        dataProperty_ts_list[i] = DataProperty()
        dataProperty_ts_list[i].setTrain('data/' + data_ts_list[i] + '/train.txt')
        dataProperty_ts_list[i].setValid('data/' + data_ts_list[i] + '/test.txt')
        dataProperty_ts_list[i].setMesh('data/' + data_ts_list[i] + '/' + data_ts_list[i].split('.')[0] + '.ply')
        dataProperty_ts_list[i].setBackup('backup/' + data_ts_list[i] )
        dataProperty_ts_list[i].setTrRange('data/' + data_ts_list[i] + '/training_range.txt')
        dataProperty_ts_list[i].setName(data_ts_list[i].split('.')[1])

        dataProperty_big_list[i] = DataProperty()
        dataProperty_big_list[i].setTrain('data/' + data_big_list[i] + '/train.txt')
        dataProperty_big_list[i].setValid('data/' + data_big_list[i] + '/test.txt')
        dataProperty_big_list[i].setMesh('data/' + data_big_list[i] + '/' + data_big_list[i].split('.')[0] + '.ply')
        dataProperty_big_list[i].setBackup('backup/' + data_big_list[i] )
        dataProperty_big_list[i].setTrRange('data/' + data_big_list[i] + '/training_range.txt')
        dataProperty_big_list[i].setName(data_big_list[i].split('.')[1])

#for batch


#------------------------------------------
# ratio





##################################
# data 파일 생성
'''
train  = LINEMOD/ape/train.txt
valid  = LINEMOD/ape/test.txt
backup = backup/ape
mesh = LINEMOD/ape/ape.ply
tr_range = LINEMOD/ape/training_range.txt
name = ape
diam = 0.103
gpus = 0
width = 640
height = 480
fx = 572.4114 
fy = 573.5704
u0 = 325.2611
v0 = 242.0489
'''


class DataProperty:
    def setTrain(self, train):
        self.train = train

    def setValid(self, valid):
        self.valid = valid

    def setMesh(self, mesh):
        self.mesh = mesh

    def setBackup(self, backup):
        self.backup = backup

    def setTrRange(self, tr_range):
        self.tr_range = tr_range

    def setName(self, name):
        self.name = name

    def setDiam(self, diam):
        self.diam = diam

    def setGpus(self, gpus):
        self.gpus = gpus

    def setWidth(self, width):
        self.width = width

    def getWidth(self):
        return self.width

    def setHeight(self, height):
        self.height = height

    def getHeight(self):
        return self.height

    def setFx(self, fx):
        self.fx = fx

    def setFy(self, fy):
        self.fy = fy

    def setU0(self, u0):
        self.u0 = u0

    def setV0(self, v0):
        self.v0 = v0

    def toString(self):
        ret = 'train = ' + self.train + '\n'
        ret += 'valid = ' + self.valid + '\n'
        ret += 'mesh = ' + self.mesh + '\n'
        ret += 'backup = ' + self.backup + '\n'
        ret += 'tr_range = ' + self.tr_range + '\n'
        ret += 'name = ' + self.name + '\n'
        ret += 'diam = ' + str(self.diam) + '\n'
        ret += 'gpus = ' + str(self.gpus) + '\n'
        ret += 'width = ' + self.width + '\n'
        ret += 'height = ' + self.height + '\n'
        ret += 'fx = ' + self.fx + '\n'
        ret += 'fy = ' + self.fy + '\n'
        ret += 'u0 = ' + self.u0 + '\n'
        ret += 'v0 = ' + self.v0

        return ret




## ply파일 복사
#투명 
name_list_vec = []
labels_ts_txt_data_dir=[]
width = []
height = []

for i in ranage(10):
    ply_filename = data_ts_list[i].split('.')[0] + '.ply'
    shutil.copy(origin_ts_threed_shape_data_dir[i] + "/" + ply_filename, "data/" + data_ts_list[i] + "/" + ply_filename)
    mesh = MeshPly("data/" + data_ts_list[i] + "/" + ply_filename)
    diam = calc_pts_diameter(np.array(mesh.vertices))
    dataProperty_ts_list[i].setDiam(diam)
    dataProperty_ts_list[i].setGpus("0,1")
    name_list = os.listdir(labeling_ts_threed_json_dir[i])
    name_list_vec[i] = name_list
    with open(labeling_threed_json_dir + '/' + name_list[0], 'r') as f:
        data2 = json.load(f)
        dataProperty_ts_list[i].setWidth(data2['metaData']['Resolution x'])
        dataProperty_ts_list[i].setHeight(data2['metaData']['Resolution y'])
        dataProperty_ts_list[i].setFx(data2['metaData']['Fx'])
        dataProperty_ts_list[i].setFy(data2['metaData']['Fy'])
        dataProperty_ts_list[i].setU0(data2['metaData']['PPx'])
        dataProperty_ts_list[i].setV0(data2['metaData']['PPy'])
    with open(os.path.join('data', data_ts_list[i] + ".data"), 'w') as file:
        file.write(dataProperty_ts_list[i].toString())
    labels_ts_txt_data_dir[i] = os.path.join('data', data_ts_list[i], 'labels')
    if not (os.path.isdir(labels_ts_txt_data_dir[i])):
        os.makedirs(os.path.join(labels_ts_txt_data_dir[i]))
    width[i] = float(dataProperty_ts_list[i].getWidth())
    height[i] = float(dataProperty_ts_list[i].getHeight())
##################################






for i in name_list:
    with open(labeling_threed_json_dir + '/' + i, 'r') as f:
        data2 = json.load(f)

        location = data2['labelingInfo'][0]['3DBox']['location'][0]
        k = open(labels_txt_data_dir + '/' + i[-21:-5] + '.txt', 'w')
        k.write('3 ')
        k.write(str(float(location['x9'])/width))
        k.write(' ')
        k.write(str(float(location['y9'])/height))
        k.write(' ')
        k.write(str(float(location['x4'])/width))
        k.write(' ')
        k.write(str(float(location['y4'])/height))
        k.write(' ')
        k.write(str(float(location['x1'])/width))
        k.write(' ')
        k.write(str(float(location['y1'])/height))
        k.write(' ')
        k.write(str(float(location['x8'])/width))
        k.write(' ')
        k.write(str(float(location['y8'])/height))
        k.write(' ')
        k.write(str(float(location['x5'])/width))
        k.write(' ')
        k.write(str(float(location['y5'])/height))
        k.write(' ')
        k.write(str(float(location['x3'])/width))
        k.write(' ')
        k.write(str(float(location['y3'])/height))
        k.write(' ')
        k.write(str(float(location['x2'])/width))
        k.write(' ')
        k.write(str(float(location['y2'])/height))
        k.write(' ')
        k.write(str(float(location['x7'])/width))
        k.write(' ')
        k.write(str(float(location['y7'])/height))
        k.write(' ')
        k.write(str(float(location['x6'])/width))
        k.write(' ')
        k.write(str(float(location['y6'])/height))
        k.write(' ')
        k.write(str(float(location['x-range'])/width))
        k.write(' ')
        k.write(str(float(location['y-range'])/height))
        k.write(' ')

data_root = os.path.join('data', data, 'labels')
loader_root = os.path.join('data', data)

name_list2 = os.listdir(data_root)
image_list = [name for name in name_list2 if name[-4:] == '.txt']
train_len = math.floor(len(image_list) * train_ratio)

train_name = random.sample(image_list, train_len)
valid_name = [name for name in image_list if name not in train_name]

with open(os.path.join(loader_root, 'train.txt'), 'w') as file:
    for i in range(len(train_name)):
        file.write(os.path.join(origin_image_dir, train_name[i].replace('.txt', '.png') + '\n'))

with open(os.path.join(loader_root, 'test.txt'), 'w') as file:
    for i in range(len(valid_name)):
        file.write(os.path.join(origin_image_dir, valid_name[i].replace('.txt', '.png') + '\n'))

with open(os.path.join(loader_root, 'training_range.txt'), 'w') as file:
    for i in range(len(train_name)):
        file.write(str(int(train_name[i].replace('.txt', '').split('_')[1])) + '\n')



if __name__ == '__main__':
    main() 
