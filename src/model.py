'''
This file contains the 
1. Face Detection Model 
2. Head Pose Estimation
3. Facial Landmark Estimation
4. Gaze Estimation
'''
import os
import cv2
import math
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class FaceDetection:
    '''
    Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', threshold=0.6):
        '''
        Intialization of model.
        '''
        self.device = device
        self.threshold = threshold
        self.core = IECore()
        self.network = self.core.read_network(model=str(model_name), weights=str(os.path.splitext(model_name)[0] + ".bin"))
        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

    def load_model(self):
        '''
        Loading the model.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, image, prob_threshold):
        '''
        Model prediction.
        '''
        img_processed = self.preprocess_input(image.copy())
        outputs = self.exec_network.infer({self.input: img_processed})
        faces_coordinates = self.preprocess_output(outputs, prob_threshold)
        if (len(faces_coordinates) == 0):
            return 0, 0
        faces_coordinates = faces_coordinates[0]
        height = image.shape[0]
        width = image.shape[1]
        faces_coordinates = faces_coordinates * \
            np.array([width, height, width, height])
        faces_coordinates = faces_coordinates.astype(np.int32)
        cropped_face_image = image[faces_coordinates[1]:faces_coordinates[3], faces_coordinates[0]:faces_coordinates[2]]
        return cropped_face_image, faces_coordinates
    
    
    def preprocess_input(self, image):
        '''
        preprocessing the input frame and image.
        '''
        net_input_shape = self.network.inputs[self.input].shape
        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose(2, 0, 1)
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs, prob_threshold):
        '''
        preprocessing the output frame.
        '''
        faces_coordinates = []
        output = outputs[self.output][0][0]
        for box in output:
            conf = box[2]
            if conf >= prob_threshold:
                xmin = box[3]
                ymin = box[4]
                xmax = box[5]
                ymax = box[6]
                faces_coordinates.append([xmin, ymin, xmax, ymax])

        return faces_coordinates    
        
         
    def check_model(self):
        '''
        Checking the supported and unsupported layer.
        '''
        supported_layers = self.core.query_network(
            network=self.network, device_name=self.device)
        unsupported_layers = [
            layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" +
                  str(unsupported_layers))
            exit(1)
        print("All layers are supported")


class FacialLandmarksDetection:
    '''
    Facial Landmark Detection Class
    '''

    def __init__(self, model_name, device='CPU', threshold=0.6):
        '''
        Intialization of facial landmark detection class.
        '''
        self.device = device
        self.threshold = threshold

        model_bin = os.path.splitext(model_name)[0] + ".bin"
        try:
            self.network = IENetwork(model_name, model_bin)
        except Exception as e:
            print(
                "Cannot initialize the network. Please enter correct model path. Error : %s", e)

        self.core = IECore()
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.width = None
        self.height = None

    def load_model(self):
        '''
        Loading the model.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, image):
        """
        prediction on input frames.
        """
        self.width = image.shape[1]
        self.height = image.shape[0]
        p_frame = self.preprocess_input(image)

        outputs = self.exec_network.infer({self.input_name: p_frame})

        left_eye, right_eye = self.preprocess_outputs(
            outputs[self.output_name])

        # cropped image for left eye
        y_left_eye = int(left_eye[1])
        x_left_eye = int(left_eye[0])
        cropped_left_eye = image[(
            y_left_eye - 20):(y_left_eye + 20), (x_left_eye - 20):(x_left_eye + 20)]

        # cropped image for Right eye
        y_right_eye = int(right_eye[1])
        x_right_eye = int(right_eye[0])
        cropped_right_eye = image[(
            y_right_eye - 20):(y_right_eye + 20), (x_right_eye - 20):(x_right_eye + 20)]

        # eye coords
        eyes_coords = [[(x_left_eye - 20, y_left_eye - 20), (x_left_eye + 20, y_left_eye + 20)],
                       [(x_right_eye - 20, y_right_eye - 20), (x_right_eye + 20, y_right_eye + 20)]]

        cv2.rectangle(image, (eyes_coords[0][0][0], eyes_coords[0][0][1]),
                      (eyes_coords[0][1][0], eyes_coords[0][1][1]), (255, 0, 0), 2)
        cv2.rectangle(image, (eyes_coords[1][0][0], eyes_coords[1][0][1]),
                      (eyes_coords[1][1][0], eyes_coords[1][1][1]), (255, 0, 0), 2)

        return cropped_left_eye, cropped_right_eye, eyes_coords

    def check_model(self):
        '''
        Checking the supported and unsupported layers of the model.
        '''
        supported_layers = self.core.query_network(
            network=self.network, device_name=self.device)
        unsupported_layers = [
            layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" +
                  str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_input(self, image):
        '''
        preprocessing the input frame.
        '''
        try:
            image = image.astype(np.float32)
            n, c, h, w = self.input_shape
            image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1))
            image = image.reshape(n, c, h, w)
        except Exception as e:
            print("Error While preprocessing Image in " + str(e))
        return image

    def preprocess_outputs(self, outputs):
        """
        preprocessing the output frame.
        """
        left_eye = (outputs[0][0][0][0] * self.width,
                    outputs[0][1][0][0] * self.height)
        right_eye = (outputs[0][2][0][0] * self.width,
                     outputs[0][3][0][0] * self.height)
        return left_eye, right_eye


class GazeEstimation:
    '''
    GazeEstimation class
    '''

    def __init__(self, model_name, device='CPU', threshold=0.6):
        '''
        intialize the Gaze Estimation class.
        '''
        self.device = device

        self.threshold = threshold
        self.core = IECore()
        self.network = self.core.read_network(
            model=str(model_name), weights=str(os.path.splitext(model_name)[0] + ".bin"))

        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

        self.input_shape = self.network.inputs[self.input].shape

    def load_model(self):
        '''
        Loading the model.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def check_model(self):
        '''
        Checking supported and unsupported layers.
        '''
        supported_layers = self.core.query_network(
            network=self.network, device_name=self.device)
        unsupported_layers = [
            layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" +
                  str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_input(self, image):
        '''
        Preprocessing the input frame and images.
        '''
        try:
            image = image.astype(np.float32)
            n, c = self.input_shape
            image = cv2.resize(image, (60, 60))
            image = image.transpose((2, 0, 1))
            image = image.reshape(n, c, 60, 60)
        except Exception as e:
            print("Error While preprocessing Image in " + str(e))
        return image

    def predict(self, left_eye, right_eye, head_pose_angles, cropped_face, eyes_coords):
        """
        Inference on the input frame.
        """
        left_eye_image = self.preprocess_input(left_eye)
        right_eye_image = self.preprocess_input(right_eye)

        outputs = self.exec_network.infer({"head_pose_angles": head_pose_angles,
                                           "left_eye_image": left_eye_image,
                                           "right_eye_image": right_eye_image
                                           })

        x = round(outputs[self.output][0][0], 4)
        y = round(outputs[self.output][0][1], 4)
        z = outputs[self.output][0][2]

        center_x_left_eye = int(
            (eyes_coords[0][1][0] - eyes_coords[0][0][0]) / 2 + eyes_coords[0][0][0])
        center_y_left_eye = int(
            (eyes_coords[0][1][1] - eyes_coords[0][0][1]) / 2 + eyes_coords[0][0][1])
        new_x_left_eye = int(center_x_left_eye + x * 90)
        new_y_left_eye = int(center_y_left_eye + y * 90 * -1)
        cv2.arrowedLine(cropped_face, (center_x_left_eye, center_y_left_eye),
                        (new_x_left_eye, new_y_left_eye), (0, 255, 0), 2)

        center_x_right_eye = int(
            (eyes_coords[1][1][0] - eyes_coords[1][0][0]) / 2 + eyes_coords[1][0][0])
        center_y_right_eye = int(
            (eyes_coords[1][1][1] - eyes_coords[1][0][1]) / 2 + eyes_coords[1][0][1])
        new_x_right_eye = int(center_x_right_eye + x * 90)
        new_y_right_eye = int(center_y_right_eye + y * 90 * -1)
        cv2.arrowedLine(cropped_face, (center_x_right_eye, center_y_right_eye), (new_x_right_eye, new_y_right_eye),
                        (0, 255, 0), 2)

        return x, y, z

    def preprocess_output(self, outputs, head_position):
        '''
        Preprocessing the output frame.
        '''
        roll = head_position[2]
        gaze_vector = outputs / cv2.norm(outputs)

        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)

        x = gaze_vector[0] * cosValue * gaze_vector[1] * sinValue
        y = gaze_vector[0] * sinValue * gaze_vector[1] * cosValue
        return (x, y)


class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation.
    '''

    def __init__(self, model_name, device='CPU', threshold=0.6):
        '''
        intialize model parameters
        '''
        self.device = device
        self.threshold = threshold
        self.core = IECore()
        self.network = self.core.read_network(
            model=str(model_name), weights=str(os.path.splitext(model_name)[0] + ".bin"))

        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

    def load_model(self):
        '''
        Loading the model.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.preprocess_image = self.preprocess_input(image)
        self.results = self.exec_network.infer(
            inputs={self.input: self.preprocess_image})
        self.output_list = self.preprocess_output(self.results)
        return self.output_list

    def check_model(self):
        '''
        Check supported and unsupported layers.
        '''
        supported_layers = self.core.query_network(
            network=self.network, device_name=self.device)
        unsupported_layers = [
            layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Check extention of these unsupported layers =>" +
                  str(unsupported_layers))
            exit(1)
        print("All layers are supported")

    def preprocess_input(self, image):
        '''
        preprocessing the input images and frames.
        '''
        image = image.astype(np.float32)
        net_input_shape = self.network.inputs[self.input].shape
        p_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose(2, 0, 1)
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs):
        '''
        preprocessing the output of the model.
        '''
        yaw = outputs["angle_y_fc"][0, 0]
        pitch = outputs["angle_p_fc"][0, 0]
        roll = outputs["angle_r_fc"][0, 0]

        return [yaw, pitch, roll]
