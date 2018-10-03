import cv2 as cv
import argparse
import sys
import numpy as np
import os.path


def set_constants():
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    inpWidth = 416  # Width of network's input image
    inpHeight = 416
    return confThreshold, nmsThreshold, inpWidth, inpHeight


def read_image_or_video():
    """
    Read input from image or video or from webcam.
    This file read the image from videoCapture as well.
    Initialize the output writer.
    :return:
    """
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()

    vid_writer = None

    if (args.image):
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)
        outputFile = args.image[:-4] + '_yolo_out_py.jpg'
    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile = args.video[:-4] + '_yolo_out_py.avi'
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                    (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    else:
        # Webcam input
        cap = cv.VideoCapture(0)
        outputFile = "yolo_out_py.avi"
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                    (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    return cap, args, outputFile, vid_writer


def load_classes_file():
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes


def specify_and_read_model():
    """
    Load configuration and weight files for the model, and load the network using them
    :return: Network
    """
    modelConfiguration = "yolov3.cfg"  # Network configuration
    modelWeights = "yolov3.weights"  # Contains the pre-trained network's weights
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net


def get_outputs_names(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    names = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return names


def draw_pred(frame, classes, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
# Non-maxima suppression is controlled by a parameter nmsThreshold
def post_process(frame, classes, outs, confThreshold, nmsThreshold):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # classIds = []
    # confidences = []
    # boxes = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_pred(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)
    return frame


def create_4D_blob(frame, inpWidth, inpHeight):
    """Create a 4D blob from a frame"""
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    return blob


def set_input(blob, net):
    """
    Set proper input for running the model forward.
    :param frame: Input frame
    :param inpWidth: Expected input width for the model
    :param inpHeight: Expected input height for the model
    :return:
    """
    net.setInput(blob)
    return net


def run_caffe_net_forward(net, names):
    """
    :return: The network output is a list of predicted bounding boxes
    """
    outs = net.forward(names)
    return outs


def put_efficiency_information(frame, net):
    """
    Put efficiency information at the top right of the output. The function getPerfProfile returns the overall
    time for inference(t) and the timings for each of the layers(in layersTimes).
    """
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return frame


def save_output(args, frame, outputFile, vid_writer):
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))


def main():
    # Set constants
    confThreshold, nmsThreshold, inpWidth, inpHeight = set_constants()

    # Get input and initialize output writer, both image and video are imported by videocapture,
    cap, args, outputFile, vid_writer = read_image_or_video()

    # Load classes name
    classes = load_classes_file()

    # Read model using local files
    net = specify_and_read_model()

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        # Create 4D blob
        blob = create_4D_blob(frame, inpWidth, inpHeight)

        # Set input for network
        net = set_input(blob, net)

        # Get names of all the output layers
        names = get_outputs_names(net)

        # Run the model
        outs = run_caffe_net_forward(net, names)

        # Remove the bounding boxes with low confidence using non-maxima suppression,
        # then draw rectangular boxes with labels on objects.
        frame = post_process(frame, classes, outs, confThreshold, nmsThreshold)

        # Put efficiency information on the top-right corner of the output.
        frame = put_efficiency_information(frame, net)

        # Save output
        save_output(args, frame, outputFile, vid_writer)


if __name__ == "__main__":
    main()


