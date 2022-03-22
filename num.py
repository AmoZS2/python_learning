# import 
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
 
# �^�[�Q�b�g�f�o�C�X�̎w�� 
plugin = IEPlugin(device='MYRIAD')
 
# ���f���̓ǂݍ��݂Ɠ��o�̓f�[�^�̃L�[�擾�i�猟�o�j 
net_face  = IENetwork(model='intel/face-detection-retail-0005/FP16/face-detection-retail-0005.xml', weights='intel/face-detection-retail-0005/FP16/face-detection-retail-0005.bin')
exec_net_face  = plugin.load(network=net_face)
input_blob_face = next(iter(net_face.inputs))
out_blob_face  = next(iter(net_face.outputs))
 
# ���f���̓ǂݍ��݂Ɠ��o�̓f�[�^�̃L�[�擾�ilandmarks�j 
net_landmarks = IENetwork(model='intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml', weights='intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin')
exec_net_landmarks = plugin.load(network=net_landmarks)
input_blob_landmarks = next(iter(net_landmarks.inputs))
out_blob_landmarks = next(iter(net_landmarks.outputs))
 
# �J�������� 
cap = cv2.VideoCapture(0)
 
#================================================== 
# ���C�����[�v 
#================================================== 
while True:
    # �L�[�����ŏI�� 
    key = cv2.waitKey(1)
    if key != -1:
        break
 
    # �J�����摜�ǂݍ��� 
    ret, frame = cap.read()
 
    # ���̓f�[�^�t�H�[�}�b�g�֕ϊ� 
    img = cv2.resize(frame, (300, 300)) # Height��Width�ύX 
    img = img.transpose((2, 0, 1))      # HWC > CHW 
    img = np.expand_dims(img, axis=0)   # CHW > BCHW 
 
    # ���_���s 
    out = exec_net_face.infer(inputs={input_blob_face: img})
 
    # �o�͂���K�v�ȃf�[�^�̂ݎ��o�� 
    out = out[out_blob_face]
 
    # �s�v�Ȏ������팸 
    out = np.squeeze(out)
 
    # ���o���ꂽ���ׂĂ̊�̈�ɑ΂��ĂP������ 
    for detection in out:
        # conf�l�̎擾 
        confidence = float(detection[2])
 
        # �o�E���f�B���O�{�b�N�X���W����͉摜�̃X�P�[���ɕϊ� 
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
 
        # conf�l��0.5���傫���ꍇ�̂�Landmarks���_�ƃo�E���f�B���O�{�b�N�X�\�� 
        if confidence > 0.5:
           # �猟�o�̈�̓J�����͈͓��ɕ␳����B����min�͕␳���Ȃ��ƃG���[�ɂȂ� 
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > frame.shape[1]:
                xmax = frame.shape[1]
            if ymax > frame.shape[0]:
                ymax = frame.shape[0]
 
            #-------------------------------------------------- 
            #  �f�B�[�v���[�j���OLandmarks���� 
            #-------------------------------------------------- 
            # ��̈�̂ݐ؂�o�� 
            img_face = frame[ ymin:ymax, xmin:xmax ]
 
            # ���̓f�[�^�t�H�[�}�b�g�֕ϊ� 
            img = cv2.resize(img_face, (48, 48)) # Height��Width�ύX 
            img = img.transpose((2, 0, 1))       # HWC > CHW 
            img = np.expand_dims(img, axis=0)    # CHW > BCHW 
 
            # ���_���s 
            out = exec_net_landmarks.infer(inputs={input_blob_landmarks: img})
 
            # �o�͂���K�v�ȃf�[�^�̂ݎ��o�� 
            out = out[out_blob_landmarks]
 
            # �s�v�Ȏ������팸 
            out = np.squeeze(out)
 
            # Landmarks���o�ʒu��circle�\�� 
            for i in range(0, 10, 2):
                x = int(out[i] * img_face.shape[1]) + xmin
                y = int(out[i+1] * img_face.shape[0]) + ymin
                cv2.circle(frame, (x, y), 10, (89, 199, 243), thickness=-1)
 
            # �o�E���f�B���O�{�b�N�X�\�� 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(89, 199, 243), thickness=3)
 
    # �摜�\�� 
    cv2.imshow('frame', frame)
 
#================================================== 
# �I������ 
#================================================== 
cap.release()
cv2.destroyAllWindows()