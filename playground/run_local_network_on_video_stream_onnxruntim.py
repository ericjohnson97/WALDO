import cv2
import numpy as np
import onnxruntime as rt
import argparse
import re
import time
import torch

def get_resolution_from_model_path(model_path):
    # Regex to find resolution in model path
    rect_match = re.search(r"_rect_(\d+)_(\d+)_", model_path)
    if rect_match:
        return int(rect_match.group(1)), int(rect_match.group(2))
    square_match = re.search(r"(\d+)px", model_path)
    if square_match:
        res = int(square_match.group(1))
        return res, res
    return None, None

def scale_based_on_bbox(bbox):
    # Compute the diagonal length of the bounding box
    diag_length = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)

    # Linearly scale the text size and thickness based on the diagonal length
    text_size = max(0.4, diag_length / 300)

    return text_size

def resize_and_pad(frame, expected_width, expected_height):
    # Resize and pad the frame to the expected dimensions
    height, width, _ = frame.shape
    ratio = min(expected_width / width, expected_height / height)
    new_width, new_height = int(width * ratio), int(height * ratio)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    padded_frame = np.zeros((expected_height, expected_width, 3), dtype=np.uint8)
    y_offset = (expected_height - new_height) // 2
    x_offset = (expected_width - new_width) // 2
    padded_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
    return padded_frame

def process_frame(frame, sess, input_name):
    colors = {
        'car': (255, 0, 0),  # Blue
        'van': (255, 0, 0),  # Cyan
        'truck': (0, 255, 0),  # Green
        'building': (0, 42, 92),  # Brown
        'human': (203, 192, 255),  # Pink
        'gastank': (0, 255, 255),  # Yellow
        'digger': (0, 0, 255),  # Red
        'container': (255, 255, 255),  # White
        'bus': (128, 0, 128),  # Purple
        'u_pole': (255, 0, 255),  # Magenta
        'boat': (0, 0, 139),  # Dark red
        'bike': (144, 238, 144),  # Light green
        'smoke': (0, 230, 128),  # Grey
        'solarpanels': (0, 0, 0),  # Black
        'arm': (0, 0, 0),  # Black
        'plane': (255, 255, 255)  # White
    }
    names = ['car', 'van', 'truck', 'building', 'human', 'gastank', 'digger', 'container', 'bus', 'u_pole', 'boat', 'bike', 'smoke',
             'solarpanels', 'arm', 'plane']

    # Prepare the frame data for inference
    image = frame.copy()
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255

    # Update the input name in the dictionary to match the model's expected input
    inp = {input_name: im}
    outputs = sess.run(None, inp)[0]
    thickness = 1

    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        box = np.array([x0, y0, x1, y1])
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 1)
        name = names[cls_id] + ' ' + str(score)
        color = colors[names[cls_id]]

        text_size = max(0.4, np.sqrt((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) / 300)

        if score > 0.5:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
            cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Display RTSP stream with ONNX model processing")
    parser.add_argument('--rtsp_url', type=str, required=True, help='RTSP stream URL')
    parser.add_argument('--model_path', type=str, required=True, help='Path to ONNX model')
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    print(f"Using CUDA: {cuda}")


    # Initialize ONNX runtime session with CUDA execution
    print("Available providers:", rt.get_available_providers())
    providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

    sess = rt.InferenceSession(args.model_path, providers=providers)
    print("Session providers:", sess.get_providers())
    input_name = sess.get_inputs()[0].name
    expected_width, expected_height = get_resolution_from_model_path(args.model_path)

    cap = cv2.VideoCapture(args.rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open stream")
        return

    frame_count = 0
    fps_timer_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()

            frame = resize_and_pad(frame, expected_width, expected_height)
            frame = process_frame(frame, sess, input_name)

            cv2.imshow('Processed Video', frame)

            frame_count += 1
            if time.time() - fps_timer_start >= 1.0:  # Check if one second has passed
                print(f"FPS: {frame_count}")
                frame_count = 0
                fps_timer_start = time.time()  # Reset the FPS timer

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()